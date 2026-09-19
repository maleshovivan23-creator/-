"""Мета-контроллер: пересобирает Протея под текущее железо и задачу.

Реализует Часть 2 спецификации: профилирование -> сборка -> мышление ->
ранний выход. Решение принимается по реальному бюджету байтов, а не «на глаз»:
конфигурация проверяется на то, что упакованные веса влезают в память.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math

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


# ═══════════════════════════════════ RL-мета-контроллер (награда из спеки)
@dataclass
class Experience:
    """Один шаг: состояние, выбранное действие, полученная награда."""

    state: Tuple[float, ...]
    action: int
    reward: float


class RLMetaController:
    """Мета-контроллер, обучающийся на награде  качество − λ·задержка − μ·энергия.

    Политика — линейный softmax по признакам состояния (батарея, температура,
    свободная память, сложность задачи). Обучение — REINFORCE с baseline:

        L = -E[log π(a|s) · A(s, a)],   A = R - b

    Это ровно та формула, что заявлена в спецификации.
    """

    #: веса награды: сколько стоят задержка и энергия относительно качества
    LAMBDA_LATENCY = 0.35
    MU_ENERGY = 0.25

    def __init__(self, model: MatFormer, lam: float = LAMBDA_LATENCY,
                 mu: float = MU_ENERGY, lr: float = 0.05, seed: int = 0) -> None:
        self.base = MetaController(model)
        self.model = model
        self.lam, self.mu, self.lr = lam, mu, lr
        self.actions = self.base.candidates           # (width, n_layers)
        self.n_features = 5
        self.rng = np.random.default_rng(seed)
        # Политика линейна по признакам ПАРЫ (состояние, действие).
        # Таблица θ[признак, действие] не работала: действия (width, layers)
        # не упорядочены, и линейная по состоянию модель не могла выразить
        # оптимум. Здесь действие описано своей стоимостью — политика
        # научается сопоставлять «сколько ресурса» и «сколько его есть».
        self.n_pair = 11
        self.theta = np.zeros(self.n_pair, dtype=np.float64)
        self.baseline = 0.0
        self.temperature = 1.0
        self.history: List[Experience] = []

    # ------------------------------------------------------------ признаки
    def features(self, state: DeviceState, complexity: float) -> np.ndarray:
        return np.array([
            1.0,
            state.battery_pct / 100.0,
            min(state.temperature_c, 100.0) / 100.0,
            float(np.clip(state.available_bytes / max(state.device.weight_budget, 1), 0, 1)),
            float(np.clip(complexity, 0.0, 1.0)),
        ], dtype=np.float64)

    def action_features(self, action: int, state: DeviceState,
                        complexity: float) -> np.ndarray:
        """φ(s, a): стоимость действия в контексте состояния."""
        w, layers = self.actions[action]
        params = self.model.active_params(w, layers)
        biggest = max(self.model.active_params(1.0, self.model.cfg.n_layer), 1)
        frac = params / biggest
        budget_use = self.base.weight_bytes(w, layers) / max(state.available_bytes, 1)
        thr = state.throttle
        bat = state.battery_pct / 100.0
        root = math.sqrt(frac)          # качество в награде растёт как √frac
        return np.array([
            root,                       # выигрыш в качестве
            root * complexity,          # качество там, где задача сложная
            root * thr,                 # качество, когда железо позволяет
            frac,                       # сколько ресурса берём
            frac * frac,                # нелинейная цена
            frac * complexity,          # ресурс под сложную задачу
            frac * thr,                 # ресурс, когда железо не душит
            frac * bat,                 # ресурс при заряде
            frac / max(thr, 0.05),      # штраф за задержку при троттлинге
            min(budget_use, 2.0),       # заполнение бюджета памяти
            float(layers) / max(self.model.cfg.n_layer, 1),   # глубина
        ], dtype=np.float64)

    def _all_action_features(self, state: DeviceState, complexity: float) -> np.ndarray:
        return np.stack([self.action_features(i, state, complexity)
                         for i in range(len(self.actions))])

    def policy(self, state: DeviceState, complexity: float) -> np.ndarray:
        """π(a|s) — распределение по конфигурациям."""
        logits = (self._all_action_features(state, complexity) @ self.theta) / self.temperature
        # недопустимые действия (не влезают в память) исключаются жёстко
        mask = np.array([self.base.weight_bytes(w, l) <= state.available_bytes
                         for w, l in self.actions], dtype=bool)
        if not mask.any():
            mask[0] = True
        logits = np.where(mask, logits, -1e9)
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()

    # -------------------------------------------------------------- награда
    def reward(self, action: int, state: DeviceState, complexity: float) -> float:
        """качество − λ·задержка − μ·энергия, всё в долях 0..1."""
        w, layers = self.actions[action]
        params = self.model.active_params(w, layers)
        biggest = self.model.active_params(1.0, self.model.cfg.n_layer)
        frac = params / max(biggest, 1)

        # качество растёт с размером, но насыщается; сложной задаче нужен размер
        quality = float(np.sqrt(frac)) * (0.45 + 0.55 * complexity)
        # задержка и энергия линейны по активным параметрам и падают от throttle
        latency = frac / max(state.throttle, 0.05)
        energy = frac * (2.0 - state.battery_pct / 100.0)
        return float(quality - self.lam * latency - self.mu * energy)

    # ------------------------------------------------------------- действие
    def act(self, state: DeviceState, complexity: float,
            explore: bool = True) -> int:
        p = self.policy(state, complexity)
        if explore:
            return int(self.rng.choice(len(p), p=p))
        return int(p.argmax())

    def plan(self, state: DeviceState, complexity: float = 0.5) -> Plan:
        """Готовый план по выученной политике (без исследования)."""
        a = self.act(state, complexity, explore=False)
        w, layers = self.actions[a]
        threshold = float(np.clip(0.55 + 0.4 * complexity, 0.5, 0.97))
        return Plan(w, layers, self.model.active_params(w, layers),
                    self.base.weight_bytes(w, layers), threshold,
                    f"RL-политика (награда {self.reward(a, state, complexity):+.3f})")

    # ------------------------------------------------------------- обучение
    def train_step(self, state: DeviceState, complexity: float) -> float:
        """Один шаг REINFORCE с ТОЧНЫМ per-state baseline.

        Скользящее среднее по всем состояниям в качестве baseline давало
        вырожденную политику: «хорошая» награда на ноутбуке выглядела
        отличной на фоне часов, и градиент тянул все состояния к одному
        действию. Здесь baseline = E_π[r | s] считается честно для текущего
        состояния — advantage становится сравнением действий между собой.
        """
        phi = self._all_action_features(state, complexity)
        p = self.policy(state, complexity)
        rewards = np.array([self.reward(i, state, complexity)
                            for i in range(len(p))], dtype=np.float64)
        a = int(self.rng.choice(len(p), p=p))
        r = float(rewards[a])

        baseline = float(np.dot(p, rewards))       # ожидаемая награда в ЭТОМ состоянии
        self.baseline = 0.95 * self.baseline + 0.05 * r
        advantage = r - baseline

        # ∇ log π(a|s) = φ(s,a) − E_π[φ(s,·)]
        grad = phi[a] - p @ phi
        self.theta += self.lr * advantage * grad
        self.history.append(Experience(tuple(self.features(state, complexity)), a, r))
        return r

    def train(self, states: Sequence[DeviceState], steps: int = 400,
              complexities: Sequence[float] = (0.1, 0.5, 0.9)) -> List[float]:
        """Обучение на наборе ситуаций. Возвращает историю наград."""
        out: List[float] = []
        lr0 = self.lr
        for i in range(steps):
            st = states[i % len(states)]
            cx = complexities[i % len(complexities)]
            # отжиг: сначала широкое исследование, к концу — уверенный выбор
            frac_done = i / max(steps - 1, 1)
            # отжиг: сначала широкое исследование, к концу — уверенный выбор
            self.temperature = max(0.05, 1.0 - 0.95 * frac_done)
            # затухание шага: крупные правки вначале, тонкая настройка в конце
            self.lr = lr0 * (1.0 - 0.9 * frac_done)
            out.append(self.train_step(st, cx))
        self.temperature = 0.05
        self.lr = lr0
        return out

    def mean_reward(self, states: Sequence[DeviceState],
                    complexities: Sequence[float] = (0.1, 0.5, 0.9),
                    greedy: bool = True) -> float:
        """Средняя награда текущей политики — метрика качества обучения."""
        vals = []
        for st in states:
            for cx in complexities:
                a = self.act(st, cx, explore=not greedy)
                vals.append(self.reward(a, st, cx))
        return float(np.mean(vals))

    def baseline_reward(self, states: Sequence[DeviceState],
                        complexities: Sequence[float] = (0.1, 0.5, 0.9)) -> float:
        """Награда эвристического MetaController — честный конкурент."""
        vals = []
        for st in states:
            for cx in complexities:
                plan = self.base.plan(st, cx)
                key = (plan.width, plan.n_layers)
                a = self.actions.index(key) if key in self.actions else 0
                vals.append(self.reward(a, st, cx))
        return float(np.mean(vals))
