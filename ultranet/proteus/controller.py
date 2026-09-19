"""Мета-контроллер: пересобирает Протея под текущее железо и задачу.

Реализует Часть 2 спецификации: профилирование -> сборка -> мышление ->
ранний выход. Решение принимается по реальному бюджету байтов, а не «на глаз»:
конфигурация проверяется на то, что упакованные веса влезают в память.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .devices import Device, DeviceState
from .matformer import MatConfig, MatFormer


@dataclass
class Plan:
    """Решение мета-контроллера."""

    width: float
    n_layers: int
    active_params: int
    weight_bytes: int
    exit_threshold: float
    reason: str
    fits: bool = True

    def __str__(self) -> str:
        kb = self.weight_bytes / 1024
        size = f"{kb / 1024:.1f} МБ" if kb >= 1024 else f"{kb:.0f} КБ"
        return (f"width={self.width:.2f} слоёв={self.n_layers} "
                f"активно={self.active_params:,} ({size}) порог={self.exit_threshold:.2f} "
                f"[{self.reason}]")


class MetaController:
    """Выбирает максимальную конфигурацию, влезающую в бюджет устройства.

    >>> mc = MetaController(model)
    >>> plan = mc.plan(DeviceState(get_device("esp32")))
    >>> plan.fits
    True
    """

    # 2 бита на вес + fp32-скейлы: реальная стоимость тернарной упаковки
    BITS_PER_WEIGHT = 2.0

    def __init__(self, model: MatFormer, bits_per_weight: float = BITS_PER_WEIGHT) -> None:
        self.model = model
        self.bits = bits_per_weight
        self.candidates = self._build_candidates()

    def _build_candidates(self) -> List[Tuple[float, int]]:
        cfg = self.model.cfg
        widths = sorted({round(self.model.width_dim(w) / cfg.n_embd, 4)
                         for w in list(cfg.widths) + [0.125, 0.25, 0.5, 0.75, 1.0]})
        layers = sorted({max(1, int(cfg.n_layer * f)) for f in (0.25, 0.5, 0.75, 1.0)})
        out = [(w, l) for w in widths for l in layers if w > 0]
        out.sort(key=lambda wl: self.model.active_params(wl[0], wl[1]))
        return out

    def weight_bytes(self, width: float, n_layers: int) -> int:
        return int(self.model.active_params(width, n_layers) * self.bits / 8)

    # --------------------------------------------------------------- решение
    def plan(self, state: DeviceState, task_complexity: float = 0.5,
             latency_ms: Optional[float] = None) -> Plan:
        """Подобрать конфигурацию под память, батарею, температуру и задачу.

        task_complexity: 0 = тривиально (включить свет), 1 = сложно (анализ).
        """
        budget = state.available_bytes
        throttle = state.throttle
        complexity = float(np.clip(task_complexity, 0.0, 1.0))

        fitting = [(w, l) for (w, l) in self.candidates
                   if self.weight_bytes(w, l) <= budget]

        if not fitting:
            w, l = self.candidates[0]
            return Plan(w, l, self.model.active_params(w, l), self.weight_bytes(w, l),
                        0.5, f"не влезает даже минимум ({budget / 1024:.0f} КБ бюджет)", fits=False)

        # верхняя граница: throttle и сложность задачи ограничивают долю от максимума
        ceiling = max(1, int(len(fitting) * min(1.0, throttle * (0.45 + 0.55 * complexity)) + 0.5))
        w, l = fitting[ceiling - 1]

        reasons = []
        if ceiling < len(fitting):
            if throttle < 1.0:
                reasons.append(f"throttle {throttle:.2f}")
            if complexity < 0.9:
                reasons.append(f"задача {complexity:.2f}")
        else:
            reasons.append("максимум для устройства")
        if len(fitting) < len(self.candidates):
            reasons.append(f"лимит памяти {budget / 1024:.0f} КБ")

        # простые задачи выходят раньше
        threshold = float(np.clip(0.55 + 0.4 * complexity, 0.5, 0.97))
        return Plan(w, l, self.model.active_params(w, l), self.weight_bytes(w, l),
                    threshold, ", ".join(reasons) or "по умолчанию")

    def explain(self, state: DeviceState, task_complexity: float = 0.5) -> str:
        p = self.plan(state, task_complexity)
        total = self.model.num_params()
        frac = p.active_params / max(total, 1) * 100
        return (f"{state}\n  -> {p}\n"
                f"  -> активно {frac:.1f}% от полной модели ({total:,} параметров)")
