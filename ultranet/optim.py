"""Оптимизаторы и планировщики скорости обучения."""
from __future__ import annotations

import math
from typing import List, Sequence

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
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            if self.wd and self.decoupled:
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


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
