"""Test-time steering vectors: направление мышления без изменения весов.

«Активировать вектор "математика" — значит сдвинуть все активации
в сторону математического мышления. Без изменения весов».

Вектор строится как разность средних активаций на двух наборах примеров
(контрастная пара), затем прибавляется к активациям нужного слоя.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..tensor import Tensor, no_grad
from .bytes import ByteTokenizer


@dataclass
class SteeringVector:
    """Направление в пространстве активаций."""

    name: str
    layer: int
    vector: np.ndarray          # (dim,)
    strength: float = 1.0

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])

    @property
    def direction(self) -> np.ndarray:
        """Единичный вектор направления."""
        n = float(np.linalg.norm(self.vector))
        return self.vector / n if n > 1e-9 else self.vector

    def scaled(self, d_act: int, alpha: Optional[float] = None) -> np.ndarray:
        a = self.strength if alpha is None else alpha
        return self.vector[:d_act] * a

    def __str__(self) -> str:
        return (f"«{self.name}» слой {self.layer} dim={self.dim} "
                f"|v|={np.linalg.norm(self.vector):.3f} сила={self.strength:.2f}")


class SteeringLibrary:
    """Набор векторов: «математика», «поэзия», «простота», «детальность»…

    >>> lib = SteeringLibrary(model)
    >>> lib.learn("простота", positive=[...], negative=[...], layer=2)
    >>> with lib.active("простота", 1.5):
    ...     out = model(ids)
    """

    def __init__(self, model, tokenizer: Optional[ByteTokenizer] = None) -> None:
        self.model = model
        self.tok = tokenizer or ByteTokenizer()
        self.vectors: Dict[str, SteeringVector] = {}

    def _as_ids(self, sample) -> List[int]:
        """Принимаем и текст, и готовые id — так пользоваться удобнее."""
        if isinstance(sample, str):
            return self.tok.encode(sample)
        return [int(t) for t in sample]

    def _resolve_layer(self, layer: int) -> int:
        """Отрицательный индекс считается от конца, как в Python."""
        n = self.model.cfg.n_layer
        li = layer + n if layer < 0 else layer
        return int(np.clip(li, 0, n - 1))

    # ----------------------------------------------------------- построение
    @no_grad()
    def _mean_activation(self, samples: Sequence[Sequence[int]], layer: int,
                         d_act: int) -> np.ndarray:
        acc = np.zeros(d_act, dtype=np.float64)
        n = 0
        for sample in samples:
            ids = self._as_ids(sample)
            if not ids:
                continue
            arr = np.array([ids[:self.model.cfg.block_size]], dtype=int)
            x = self.model.embed(arr, d_act)
            for i in range(layer + 1):
                x = self.model.run_block(i, x, d_act)
            acc += x.data[0].mean(axis=0).astype(np.float64)
            n += 1
        return (acc / max(n, 1)).astype(np.float32)

    def learn(self, name: str, positive: Sequence[Sequence[int]],
              negative: Sequence[Sequence[int]], layer: int = 2,
              strength: float = 1.0, normalize: bool = True) -> SteeringVector:
        """Контрастный вектор: среднее(positive) − среднее(negative)."""
        d = self.model.width_dim(1.0)
        layer = self._resolve_layer(layer)
        pos = self._mean_activation(positive, layer, d)
        neg = self._mean_activation(negative, layer, d)
        v = pos - neg
        if normalize:
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                v = v / norm
        sv = SteeringVector(name, layer, v.astype(np.float32), strength)
        self.vectors[name] = sv
        return sv

    def add(self, sv: SteeringVector) -> None:
        self.vectors[sv.name] = sv

    def names(self) -> List[str]:
        return list(self.vectors)

    def __getitem__(self, name: str) -> SteeringVector:
        return self.vectors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.vectors

    def __len__(self) -> int:
        return len(self.vectors)

    # ------------------------------------------------------------ применение
    def apply(self, name: str, strength: Optional[float] = None) -> None:
        """Включить вектор для последующих forward."""
        if name not in self.vectors:
            raise KeyError(f"вектор «{name}» не выучен. Есть: {', '.join(self.vectors) or '—'}")
        sv = self.vectors[name]
        self.model.set_steering(sv.layer, sv.vector,
                                sv.strength if strength is None else strength)

    def clear(self) -> None:
        self.model.clear_steering()

    def active(self, name: str, strength: Optional[float] = None):
        """Контекстный менеджер: вектор действует только внутри блока."""
        lib = self

        class _Ctx:
            def __enter__(self):
                lib.apply(name, strength)
                return lib

            def __exit__(self, *exc):
                lib.clear()
                return False

        return _Ctx()

    # -------------------------------------------------------------- эффект
    @no_grad()
    def effect_size(self, name: str, probe: Optional[Sequence] = None,
                    strength: Optional[float] = None) -> float:
        """Насколько сильно вектор меняет распределение на выходе (L1, 0..2)."""
        if probe is None:
            probe = [1, 2, 3, 4, 5, 6, 7, 8]
        ids = self._as_ids(probe)
        arr = np.array([ids[:self.model.cfg.block_size]], dtype=int)
        base = self.model(arr).softmax(axis=-1).data[0, -1]
        self.apply(name, strength)
        steered = self.model(arr).softmax(axis=-1).data[0, -1]
        self.clear()
        return float(np.abs(base - steered).sum())
