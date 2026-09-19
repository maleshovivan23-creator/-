"""Функции потерь и метрики."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def one_hot(idx: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n_classes), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx.astype(int)] = 1.0
    return out


def cross_entropy(logits: Tensor, targets, label_smoothing: float = 0.0) -> Tensor:
    """Кросс-энтропия по логитам. targets — индексы классов (..., ) или one-hot."""
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    t = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    n_classes = logits.shape[-1]
    if t.ndim == logits.ndim and t.shape == logits.shape:
        y = t.astype(np.float32)
    else:
        y = one_hot(t.reshape(-1), n_classes)
    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + label_smoothing / n_classes
    logp = logits.log_softmax(axis=-1)
    return -(logp * Tensor(y)).sum(axis=-1).mean()


def binary_cross_entropy(probs: Tensor, targets) -> Tensor:
    y = Tensor(targets.data if isinstance(targets, Tensor) else np.asarray(targets, dtype=np.float32))
    p = probs.clip(1e-7, 1 - 1e-7)
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean()


def mse_loss(pred: Tensor, target) -> Tensor:
    y = Tensor(target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32))
    return ((pred - y) ** 2).mean()


def accuracy(logits: Tensor, targets) -> float:
    t = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    pred = logits.data.reshape(-1, logits.shape[-1]).argmax(-1)
    return float((pred == t.reshape(-1).astype(int)).mean())
