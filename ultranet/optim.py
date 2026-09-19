"""Оптимизаторы и планировщики скорости обучения."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from .nn import Parameter


class Optimizer:
    def __init__(self, params: Sequence[Parameter], lr: float) -> None:
        self.params: List[Parameter] = [p for p in params if p is not None]
        self.lr = lr
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def clip_grad_norm(self, max_norm: float) -> float:
        total = math.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params if p.grad is not None))
        if total > max_norm and total > 0:
            scale = max_norm / (total + 1e-6)
            for p in self.params:
                if p.grad is not None:
                    p.grad *= scale
        return total

    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    """SGD с моментом, поддержкой Nesterov и weight decay."""

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.9,
                 weight_decay: float = 0.0, nesterov: bool = False) -> None:
        super().__init__(params, lr)
        self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            self.buf[i] = self.momentum * self.buf[i] + g
            upd = g + self.momentum * self.buf[i] if self.nesterov else self.buf[i]
            p.data -= self.lr * upd


class Adam(Optimizer):
    """Adam / AdamW (decoupled weight decay)."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, decoupled: bool = True) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd, self.decoupled = eps, weight_decay, decoupled
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        #: переиспользуемый буфер под знаменатель — чтобы не аллоцировать на шаге
        self._buf = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd and not self.decoupled:
                g = g + self.wd * p.data
            # in-place: раньше каждая строка порождала новый массив на каждый
            # параметр каждый шаг. Для GPT-2.7M это ~50 аллокаций на шаг.
            m, v = self.m[i], self.v[i]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * (g * g)
            if self.wd and self.decoupled:
                p.data -= self.lr * self.wd * p.data
            # буфер переиспользуется: sqrt(v/bc2) + eps
            denom = self._buf[i]
            np.multiply(v, 1.0 / bc2, out=denom)
            np.sqrt(denom, out=denom)
            denom += self.eps
            np.divide(m, denom, out=denom)
            p.data -= (self.lr / bc1) * denom


AdamW = Adam


class RMSprop(Optimizer):
    def __init__(self, params, lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__(params, lr)
        self.alpha, self.eps = alpha, eps
        self.sq = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.sq[i] = self.alpha * self.sq[i] + (1 - self.alpha) * p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.sq[i]) + self.eps)


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum, Google 2023) — знаковые обновления, мало памяти."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            upd = np.sign(self.b1 * self.m[i] + (1 - self.b1) * p.grad)
            if self.wd:
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * upd
            self.m[i] = self.b2 * self.m[i] + (1 - self.b2) * p.grad


class Adagrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-10) -> None:
        super().__init__(params, lr)
        self.eps = eps
        self.sq = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.sq[i] += p.grad ** 2
            p.data -= self.lr * p.grad / (np.sqrt(self.sq[i]) + self.eps)


class Lookahead(Optimizer):
    """Lookahead (Zhang et al.): «медленные» веса догоняют быстрые каждые k шагов."""

    def __init__(self, base: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        self.base, self.k, self.alpha = base, k, alpha
        self.params = base.params
        self.t = 0
        self.slow = [p.data.copy() for p in self.params]

    @property
    def lr(self) -> float:
        return self.base.lr

    @lr.setter
    def lr(self, v: float) -> None:
        self.base.lr = v

    def zero_grad(self) -> None:
        self.base.zero_grad()

    def clip_grad_norm(self, max_norm: float) -> float:
        return self.base.clip_grad_norm(max_norm)

    def step(self) -> None:
        self.base.step()
        self.t += 1
        if self.t % self.k == 0:
            for i, p in enumerate(self.params):
                self.slow[i] += self.alpha * (p.data - self.slow[i])
                p.data = self.slow[i].copy()


class EMA:
    """Экспоненциальное скользящее среднее весов — обычно даёт лучший eval.

    Использует «прогрев» decay = min(decay, (1+t)/(10+t)), поэтому на ранних
    шагах среднее не отстаёт от весов (иначе короткое обучение ломается).
    """

    def __init__(self, params: Sequence[Parameter], decay: float = 0.999,
                 warmup: bool = True) -> None:
        self.params = list(params)
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow = [p.data.copy() for p in self.params]
        self._backup: Optional[List[np.ndarray]] = None

    def _current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        t = self.num_updates
        return min(self.decay, (1.0 + t) / (10.0 + t))

    def update(self) -> None:
        self.num_updates += 1
        d = self._current_decay()
        for i, p in enumerate(self.params):
            self.shadow[i] = d * self.shadow[i] + (1 - d) * p.data

    def apply(self) -> None:
        """Подставить усреднённые веса (с возможностью восстановить)."""
        self._backup = [p.data.copy() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data = s.copy()

    def restore(self) -> None:
        if self._backup is not None:
            for p, b in zip(self.params, self._backup):
                p.data = b
            self._backup = None


class CosineWarmup:
    """Косинусный спад с линейным прогревом (как у GPT)."""

    def __init__(self, optimizer: Optimizer, warmup: int, total: int, min_lr_ratio: float = 0.1) -> None:
        self.opt, self.warmup, self.total = optimizer, max(1, warmup), total
        self.base_lr, self.min_ratio, self.step_num = optimizer.lr, min_lr_ratio, 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / self.warmup
        else:
            prog = min(1.0, (self.step_num - self.warmup) / max(1, self.total - self.warmup))
            lr = self.base_lr * (self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
        self.opt.lr = lr
        return lr


class StepLR:
    """Ступенчатое снижение lr каждые step_size шагов."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        self.opt, self.step_size, self.gamma = optimizer, step_size, gamma
        self.base_lr, self.step_num = optimizer.lr, 0

    def step(self) -> float:
        self.step_num += 1
        self.opt.lr = self.base_lr * self.gamma ** (self.step_num // self.step_size)
        return self.opt.lr


class OneCycleLR:
    """One-cycle политика Лесли Смита: разогрев до max_lr и спад почти до нуля."""

    def __init__(self, optimizer: Optimizer, max_lr: float, total: int,
                 pct_start: float = 0.3, final_div: float = 1e4) -> None:
        self.opt, self.max_lr, self.total = optimizer, max_lr, max(1, total)
        self.warm = max(1, int(total * pct_start))
        self.init_lr = max_lr / 25.0
        self.final_lr = max_lr / final_div
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warm:
            prog = self.step_num / self.warm
            lr = self.init_lr + (self.max_lr - self.init_lr) * 0.5 * (1 - math.cos(math.pi * prog))
        else:
            prog = min(1.0, (self.step_num - self.warm) / max(1, self.total - self.warm))
            lr = self.final_lr + (self.max_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * prog))
        self.opt.lr = lr
        return lr


class ReduceLROnPlateau:
    """Снижает lr, когда метрика перестаёт улучшаться."""

    def __init__(self, optimizer: Optimizer, factor: float = 0.5, patience: int = 3,
                 min_lr: float = 1e-6) -> None:
        self.opt, self.factor, self.patience, self.min_lr = optimizer, factor, patience, min_lr
        self.best, self.bad = float("inf"), 0

    def step(self, metric: float) -> float:
        if metric < self.best - 1e-6:
            self.best, self.bad = metric, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.opt.lr = max(self.min_lr, self.opt.lr * self.factor)
                self.bad = 0
        return self.opt.lr
