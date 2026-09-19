"""MoE: специализированные эксперты и роутер.

«Активны разные 10% для разных задач. Просишь код — работают одни.
Просишь стихи — другие». Роутер — маленькая сеть, которая решает,
каких экспертов включить.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import nn
from ..nn import Parameter
from ..tensor import Tensor, get_rng


def _round_to(d: int, mult: int) -> int:
    return max(mult, int(round(d / mult)) * mult)


class Expert(nn.Module):
    """Один эксперт — SwiGLU-блок с эластичной шириной."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        from .matformer import ElasticLinear

        self.dim, self.hidden = dim, hidden
        self.w_gate = ElasticLinear(dim, hidden)
        self.w_up = ElasticLinear(dim, hidden)
        self.w_down = ElasticLinear(hidden, dim)

    def forward(self, x: Tensor, d_act: int) -> Tensor:
        h_act = _round_to(self.hidden * d_act // self.dim, 8)
        gated = self.w_gate(x, d_act, h_act).silu() * self.w_up(x, d_act, h_act)
        return self.w_down(gated, h_act, d_act)

    def active_params(self, d_act: int) -> int:
        h_act = _round_to(self.hidden * d_act // self.dim, 8)
        return 3 * d_act * h_act


class MoE(nn.Module):
    """Смесь экспертов с top-k роутингом.

    Роутер обучается предсказывать, какой эксперт даст лучший ответ.
    Вспомогательный loss балансирует нагрузку, иначе роутер схлопывается
    в одного «любимого» эксперта.

    >>> moe = MoE(64, n_experts=4, top_k=2)
    >>> y = moe(Tensor(np.random.randn(2, 5, 64)), d_act=64)
    """

    def __init__(self, dim: int, n_experts: int = 4, top_k: int = 2,
                 mlp_ratio: int = 4) -> None:
        super().__init__()
        assert 1 <= top_k <= n_experts, "top_k должен быть в [1, n_experts]"
        self.dim, self.n_experts, self.top_k = dim, n_experts, top_k
        hidden = mlp_ratio * dim
        self.experts = [Expert(dim, hidden) for _ in range(n_experts)]
        for i, e in enumerate(self.experts):
            self._modules[f"expert{i}"] = e
        # роутер — маленькая матрица dim -> n_experts
        self.router = Parameter(
            get_rng().standard_normal((dim, n_experts)).astype(np.float32) * 0.02
        )
        self.last_load: Optional[np.ndarray] = None
        self.last_aux: float = 0.0
        self._forced: Optional[List[int]] = None

    # ------------------------------------------------------------ управление
    def force_experts(self, idx: Optional[List[int]]) -> None:
        """Принудительно активировать конкретных экспертов (для отладки/капсул)."""
        self._forced = idx

    def route(self, x: Tensor, d_act: int) -> np.ndarray:
        """Распределение по экспертам (без вычисления самих экспертов)."""
        logits = (x @ self.router[:d_act]).data
        z = logits - logits.max(-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(-1, keepdims=True)

    # --------------------------------------------------------------- forward
    def forward(self, x: Tensor, d_act: int, n_active: Optional[int] = None,
                force: Optional[List[int]] = None) -> Tensor:
        """force — временно активировать конкретных экспертов (капсулы/отладка)."""
        if force is not None:
            self._forced = list(force)
        k = int(np.clip(n_active or self.top_k, 1, self.n_experts))
        gate_logits = x @ self.router[:d_act]              # (B,T,E)
        probs = gate_logits.softmax(axis=-1)
        pd = probs.data
        b, t, _ = pd.shape

        if self._forced is not None:
            chosen = np.tile(np.array(self._forced[:k], dtype=int), (b, t, 1))
        else:
            chosen = np.argsort(-pd, axis=-1)[..., :k]      # (B,T,k)

        # нормируем веса выбранных экспертов
        sel = np.take_along_axis(pd, chosen, axis=-1)
        sel_norm = sel / np.clip(sel.sum(-1, keepdims=True), 1e-9, None)

        # знаменатель нормировки top-k (константа для градиента — как в Switch/Mixtral)
        denom = np.clip(sel.sum(-1, keepdims=True), 1e-9, None)   # (B,T,1)
        out: Optional[Tensor] = None
        load = np.zeros(self.n_experts, dtype=np.float64)
        for e_idx in range(self.n_experts):
            hit = (chosen == e_idx).any(-1)                  # (B,T) выбран ли эксперт
            if not hit.any():
                continue
            load[e_idx] = float(hit.mean())
            # вес = prob эксперта / сумму вероятностей выбранных; градиент идёт в роутер
            keep = Tensor(hit[..., None].astype(np.float32))
            gate = probs[:, :, e_idx:e_idx + 1] * keep / Tensor(denom.astype(np.float32))
            contrib = self.experts[e_idx](x, d_act) * gate
            out = contrib if out is None else out + contrib

        self.last_load = load
        # aux-loss: поощряем равномерную загрузку (Switch Transformer)
        frac = load / max(load.sum(), 1e-9)
        mean_p = pd.reshape(-1, self.n_experts).mean(0)
        self.last_aux = float(self.n_experts * (frac * mean_p).sum())

        if out is None:                                       # страховка
            out = self.experts[0](x, d_act) * 0.0
        return out

    def aux_loss(self, x: Optional[Tensor] = None,
                 d_act: Optional[int] = None):
        """Балансирующий loss роутера.

        С аргументами — дифференцируемый Tensor для обучения.
        Без аргументов — число: дисбаланс последнего прогона (для отчётов).
        """
        if x is None:
            load = self.last_load
            if load is None or not len(load):
                return 0.0
            target = 1.0 / self.n_experts
            return float(((load / max(load.sum(), 1e-9) - target) ** 2).sum()
                         * self.n_experts)
        probs = (x @ self.router[:d_act]).softmax(axis=-1)
        mean_p = probs.mean(axis=(0, 1))
        target = 1.0 / self.n_experts
        return ((mean_p - target) ** 2).sum() * float(self.n_experts)

    def active_params(self, d_act: int, n_active: Optional[int] = None) -> int:
        k = int(np.clip(n_active or self.top_k, 1, self.n_experts))
        return k * self.experts[0].active_params(d_act) + d_act * self.n_experts

    def total_params(self, d_act: int) -> int:
        return self.n_experts * self.experts[0].active_params(d_act) + d_act * self.n_experts

    # ---------------------------------------------------------- специализация
    def specialization(self, samples: Dict[str, np.ndarray], model, d_act: int,
                       layer: int) -> Dict[str, np.ndarray]:
        """Какие эксперты активируются на разных типах задач."""
        out: Dict[str, np.ndarray] = {}
        for name, ids in samples.items():
            x = model.embed(ids, d_act)
            for i in range(layer):
                x = model.run_block(i, x, d_act)
            h = model.norm_for_moe(x, layer, d_act)
            p = self.route(h, d_act).reshape(-1, self.n_experts).mean(0)
            out[name] = p
        return out
