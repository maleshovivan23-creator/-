"""Функции потерь и метрики."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import Tensor


def one_hot(idx: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n_classes), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx.astype(int)] = 1.0
    return out


def cross_entropy(logits: Tensor, targets, label_smoothing: float = 0.0,
                  ignore_index: Optional[int] = None) -> Tensor:
    """Fused softmax + NLL одним узлом графа.

    Градиент считается аналитически как (softmax(x) - y) / N, поэтому
    проход дешевле и численно устойчивее цепочки log_softmax -> gather.

    targets: индексы классов (любой формы) или one-hot той же формы, что logits.
    ignore_index: позиции с этой меткой исключаются из loss (паддинг).
    """
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    t = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    n_classes = logits.shape[-1]

    if t.ndim == 2 and t.shape == logits.shape:
        y = t.astype(np.float32)
        mask = np.ones((logits.shape[0], 1), dtype=np.float32)
    else:
        flat = t.reshape(-1).astype(int)
        valid = flat != ignore_index if ignore_index is not None else np.ones_like(flat, dtype=bool)
        y = one_hot(np.where(valid, flat, 0), n_classes)
        mask = valid.astype(np.float32)[:, None]

    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + label_smoothing / n_classes

    x = logits.data
    z = x - x.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - lse
    n = max(float(mask.sum()), 1.0)
    loss_val = -(logp * y * mask).sum() / n

    out = logits._make(np.array(loss_val, dtype=np.float32), (logits,), "cross_entropy")
    probs = np.exp(logp)

    def _backward() -> None:
        logits._accum(out.grad * (probs - y) * mask / n)

    out._backward = _backward
    return out


def nll_loss(log_probs: Tensor, targets) -> Tensor:
    """Negative log-likelihood поверх уже посчитанных log-вероятностей."""
    if log_probs.ndim > 2:
        log_probs = log_probs.reshape(-1, log_probs.shape[-1])
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).reshape(-1).astype(int)
    y = one_hot(t, log_probs.shape[-1])
    return -(log_probs * Tensor(y)).sum(axis=-1).mean()


def binary_cross_entropy(probs: Tensor, targets) -> Tensor:
    y = Tensor(targets.data if isinstance(targets, Tensor) else np.asarray(targets, dtype=np.float32))
    p = probs.clip(1e-7, 1 - 1e-7)
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean()


def bce_with_logits(logits: Tensor, targets) -> Tensor:
    """Численно устойчивая BCE прямо по логитам (log-sum-exp трюк)."""
    y = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=np.float32)
    x = logits.data
    loss_val = float(np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))))
    out = logits._make(np.array(loss_val, dtype=np.float32), (logits,), "bce_logits")
    sig = 1.0 / (1.0 + np.exp(-x))

    def _backward() -> None:
        logits._accum(out.grad * (sig - y) / x.size)

    out._backward = _backward
    return out


def focal_loss(logits: Tensor, targets, gamma: float = 2.0, alpha: float = 1.0) -> Tensor:
    """Focal loss — приглушает вклад лёгких примеров при дисбалансе классов."""
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).reshape(-1).astype(int)
    logp = logits.log_softmax(axis=-1)
    y = Tensor(one_hot(t, logits.shape[-1]))
    pt = (logp * y).sum(axis=-1)
    focal = (1 - pt.exp()) ** gamma
    return -(focal.detach() * pt).mean() * alpha


def mse_loss(pred: Tensor, target) -> Tensor:
    y = Tensor(target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32))
    return ((pred - y) ** 2).mean()


def mae_loss(pred: Tensor, target) -> Tensor:
    y = Tensor(target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32))
    return (pred - y).abs().mean()


def huber_loss(pred: Tensor, target, delta: float = 1.0) -> Tensor:
    """Робастная к выбросам смесь MSE (внутри delta) и MAE (снаружи)."""
    y = Tensor(target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32))
    diff = pred - y
    a = diff.abs()
    small = (np.abs(diff.data) <= delta).astype(np.float32)
    quad = (diff ** 2) * 0.5
    lin = (a - 0.5 * delta) * delta
    return (quad * Tensor(small) + lin * Tensor(1 - small)).mean()


# ------------------------------------------------------------------------ метрики
def accuracy(logits: Tensor, targets) -> float:
    t = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    pred = logits.data.reshape(-1, logits.shape[-1]).argmax(-1)
    return float((pred == t.reshape(-1).astype(int)).mean())


def top_k_accuracy(logits: Tensor, targets, k: int = 5) -> float:
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).reshape(-1).astype(int)
    x = logits.data.reshape(-1, logits.shape[-1])
    topk = np.argpartition(-x, min(k, x.shape[-1]) - 1, axis=-1)[:, :k]
    return float(np.mean([t[i] in topk[i] for i in range(len(t))]))


def perplexity(loss_value: float) -> float:
    """Перплексия языковой модели из значения кросс-энтропии."""
    return float(np.exp(min(loss_value, 20.0)))


def confusion_matrix(logits: Tensor, targets, n_classes: Optional[int] = None) -> np.ndarray:
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).reshape(-1).astype(int)
    pred = logits.data.reshape(-1, logits.shape[-1]).argmax(-1)
    n = n_classes or int(max(t.max(), pred.max()) + 1)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (t, pred), 1)
    return cm


def f1_score(logits: Tensor, targets, average: str = "macro") -> float:
    cm = confusion_matrix(logits, targets)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    if average == "macro":
        return float(f1.mean())
    weights = cm.sum(1) / max(cm.sum(), 1)
    return float((f1 * weights).sum())
