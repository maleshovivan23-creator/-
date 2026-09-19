"""Персонализация: LoRA-адаптеры, защита от forgetting, четыре петли адаптации.

«Ядро заморожено. Меняются только маленькие надстройки — несколько мегабайт».

Защита от забывания реализована тремя способами из спецификации:
  1. Ортогональная инициализация новых адаптеров к старым.
  2. Replay — подмешивание старых примеров.
  3. Регуляризация важных весов (EWC-подобная).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..functional import cross_entropy
from ..nn import Module, Parameter
from ..optim import Adam
from ..tensor import Tensor, get_rng, no_grad


class LoRALayer(Module):
    """Низкоранговая добавка: W_eff = W + (B @ A) * scale.

    Обучаются только A и B — это r*(in+out) параметров вместо in*out.
    """

    def __init__(self, in_f: int, out_f: int, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.in_f, self.out_f, self.rank = in_f, out_f, rank
        self.scale = alpha / rank
        # A ~ N(0, σ), B = 0 -> в начале адаптер не меняет поведение
        self.A = Parameter(get_rng().standard_normal((in_f, rank)).astype(np.float32)
                           * (1.0 / math.sqrt(in_f)))
        self.B = Parameter(np.zeros((rank, out_f), dtype=np.float32))

    def forward(self, x: Tensor, in_d: Optional[int] = None,
                out_d: Optional[int] = None) -> Tensor:
        in_d = in_d or self.in_f
        out_d = out_d or self.out_f
        return (x @ self.A[:in_d]) @ self.B[:, :out_d] * self.scale

    def delta(self, in_d: Optional[int] = None, out_d: Optional[int] = None) -> np.ndarray:
        in_d = in_d or self.in_f
        out_d = out_d or self.out_f
        return (self.A.data[:in_d] @ self.B.data[:, :out_d]) * self.scale

    def n_params(self) -> int:
        return self.A.size + self.B.size

    def orthogonalize_to(self, others: Sequence["LoRALayer"]) -> None:
        """Ортогонализовать A к подпространствам прошлых адаптеров (Грам-Шмидт).

        Новый адаптер не пересекается со старыми -> меньше интерференции.
        """
        if not others:
            return
        basis: List[np.ndarray] = []
        for o in others:
            for col in o.A.data.T:
                v = col.copy()
                for b in basis:
                    v -= np.dot(v, b) * b
                n = np.linalg.norm(v)
                if n > 1e-6:
                    basis.append(v / n)
        newA = self.A.data.copy()
        for j in range(newA.shape[1]):
            v = newA[:, j]
            for b in basis:
                v -= np.dot(v, b) * b
            n = np.linalg.norm(v)
            newA[:, j] = v / n * (1.0 / math.sqrt(self.in_f)) if n > 1e-6 else newA[:, j]
        self.A.data = newA

    def overlap_with(self, other: "LoRALayer") -> float:
        """Косинусная близость подпространств (0 = ортогональны)."""
        def orth(M: np.ndarray) -> np.ndarray:
            q, _ = np.linalg.qr(M)
            return q
        qa, qb = orth(self.A.data), orth(other.A.data)
        return float(np.abs(qa.T @ qb).max())


@dataclass
class Feedback:
    """Сигнал от пользователя: правка, лайк, игнор."""

    ids: List[int]
    reward: float = 1.0        # 1 = нравится, -1 = плохо, 0 = нейтрально
    tag: str = ""


class PersonalAdapter(Module):
    """Набор LoRA-адаптеров поверх замороженного ядра.

    >>> pa = PersonalAdapter(model, rank=4)
    >>> pa.observe(Feedback(ids, reward=1.0))
    >>> pa.consolidate(steps=20)     # раз в несколько часов
    """

    def __init__(self, model, rank: int = 4, alpha: float = 8.0,
                 replay_size: int = 64) -> None:
        super().__init__()
        # ВАЖНО: ядро хранится в обход Module.__setattr__, иначе оно попадёт
        # в _modules и parameters() вернёт его веса — оптимизатор начнёт
        # обучать ядро, хотя спецификация требует «ядро заморожено».
        object.__setattr__(self, "model", model)
        self.rank = rank
        d = model.cfg.n_embd
        # адаптеры на выходных проекциях внимания каждого слоя
        self.loras: Dict[int, LoRALayer] = {}
        for i in range(model.cfg.n_layer):
            lora = LoRALayer(d, d, rank, alpha)
            self.loras[i] = lora
            self._modules[f"lora{i}"] = lora
        self.buffer: Deque[Feedback] = deque(maxlen=replay_size)
        self.replay: Deque[List[int]] = deque(maxlen=replay_size)
        self.fisher: Dict[str, np.ndarray] = {}
        self.anchor: Dict[str, np.ndarray] = {}
        self.updates = 0
        self.enabled = True

    # -------------------------------------------------------------- параметры
    def n_params(self) -> int:
        return sum(l.n_params() for l in self.loras.values())

    def core_params(self) -> int:
        return self.model.num_params()

    def ratio(self) -> float:
        return self.n_params() / max(self.core_params(), 1)

    def core_frozen(self) -> bool:
        """Ядро не должно получать градиенты при обучении адаптера."""
        return all(p.grad is None or not np.abs(p.grad).any()
                   for p in self.model.parameters())

    # ----------------------------------------------------------------- петли
    def observe(self, fb: Feedback) -> None:
        """Быстрая петля: копим сигналы (лайки/правки)."""
        self.buffer.append(fb)

    def add_replay(self, ids: Sequence[int]) -> None:
        """Старые примеры для защиты от забывания."""
        self.replay.append(list(ids))

    def snapshot_importance(self, ids: Sequence[int], block: int) -> None:
        """Оценить важность параметров адаптера (диагональ Фишера) и закрепить."""
        params = dict(self.named_parameters())
        arr = np.array([list(ids)[:block]], dtype=int)
        x, y = arr[:, :-1], arr[:, 1:]
        for p in params.values():
            p.grad = None
        loss = cross_entropy(self.forward_with_adapter(x), y)
        loss.backward()
        for name, p in params.items():
            g = np.zeros_like(p.data) if p.grad is None else p.grad
            self.fisher[name] = self.fisher.get(name, 0.0) + g ** 2
            self.anchor[name] = p.data.copy()
            p.grad = None

    # -------------------------------------------------------------- forward
    def forward_with_adapter(self, idx, width: float = 1.0) -> Tensor:
        """Прогон модели с активными LoRA-добавками."""
        d = self.model.width_dim(width)
        x = self.model.embed(idx, d)
        for i in range(self.model.cfg.n_layer):
            x = self.model.run_block(i, x, d)
            if self.enabled:
                x = x + self.loras[i](x, d, d)
        return self.model.readout(x, d)

    # ---------------------------------------------------------- консолидация
    def consolidate(self, steps: int = 20, lr: float = 1e-3, block: int = 32,
                    ewc_lambda: float = 1.0, replay_ratio: float = 0.5,
                    verbose: bool = False) -> Dict[str, List[float]]:
        """Средняя петля: дообучить адаптеры на накопленных сигналах.

        Ядро заморожено: оптимизатор видит только параметры LoRA.
        """
        if not self.buffer:
            return {"loss": []}
        opt = Adam(self.parameters(), lr=lr)      # ТОЛЬКО параметры адаптера
        hist: List[float] = []
        pos = [f for f in self.buffer if f.reward > 0]
        if not pos:
            return {"loss": []}

        for step in range(steps):
            use_replay = self.replay and np.random.rand() < replay_ratio
            if use_replay:
                ids = list(self.replay[np.random.randint(len(self.replay))])
            else:
                ids = list(pos[np.random.randint(len(pos))].ids)
            ids = ids[:block + 1]
            if len(ids) < 4:
                continue
            arr = np.array([ids], dtype=int)
            x, y = arr[:, :-1], arr[:, 1:]

            opt.zero_grad()
            loss = cross_entropy(self.forward_with_adapter(x), y)
            # EWC-регуляризация: важные веса держатся около якоря
            if self.fisher and ewc_lambda > 0:
                for name, p in self.named_parameters():
                    if name in self.fisher:
                        f = Tensor(self.fisher[name])
                        a = Tensor(self.anchor[name])
                        loss = loss + ((p - a) ** 2 * f).sum() * ewc_lambda
            loss.backward()
            opt.clip_grad_norm(1.0)
            opt.step()
            hist.append(loss.item())
            if verbose and step % 10 == 0:
                print(f"    адаптер шаг {step}: {loss.item():.4f}")

        self.updates += 1
        return {"loss": hist}

    def new_task_adapter(self, layer: int) -> LoRALayer:
        """Создать адаптер для новой задачи, ортогональный существующим."""
        d = self.model.cfg.n_embd
        fresh = LoRALayer(d, d, self.rank)
        fresh.orthogonalize_to([self.loras[layer]])
        return fresh

    # ------------------------------------------------------------------ вес
    def size_bytes(self, bits: float = 16.0) -> int:
        return int(self.n_params() * bits / 8)

    def report(self) -> str:
        return (f"LoRA rank={self.rank}: {self.n_params():,} параметров "
                f"({self.size_bytes() / 1024:.1f} КБ) против ядра "
                f"{self.core_params():,} — {self.ratio() * 100:.2f}% | "
                f"обновлений: {self.updates}")
