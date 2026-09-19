"""Численная проверка градиентов — публичный инструмент отладки своих слоёв.

    >>> import ultranet as un
    >>> layer = un.nn.Linear(4, 3)
    >>> x = un.Tensor(np.random.randn(2, 4), requires_grad=True)
    >>> un.gradcheck(lambda: layer(x).sum(), [x, layer.weight])
    True
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

from .tensor import Tensor


def numeric_grad(fn: Callable[[], Tensor], param: Tensor, eps: float = 1e-3) -> np.ndarray:
    """Градиент центральной разностью: (f(x+e) - f(x-e)) / 2e."""
    grad = np.zeros_like(param.data)
    flat = param.data.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        plus = float(fn().item())
        flat[i] = orig - eps
        minus = float(fn().item())
        flat[i] = orig
        gflat[i] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(fn: Callable[[], Tensor], params: Sequence[Tensor], eps: float = 1e-3,
              atol: float = 2e-2, rtol: float = 2e-2, verbose: bool = False,
              raise_exception: bool = True) -> bool:
    """Сравнить аналитический градиент с численным.

    fn должна возвращать скалярный Tensor и быть чистой функцией от params.
    """
    loss = fn()
    if loss.size != 1:
        raise ValueError("fn должна возвращать скаляр")
    for p in params:
        p.grad = None
    loss.backward()

    ok = True
    report: List[str] = []
    for i, p in enumerate(params):
        analytic = np.zeros_like(p.data) if p.grad is None else p.grad
        numeric = numeric_grad(fn, p, eps)
        denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
        rel = np.abs(analytic - numeric) / denom
        close = np.allclose(analytic, numeric, atol=atol, rtol=rtol)
        ok &= bool(close)
        report.append(
            f"  param[{i}] shape={p.shape} max|Δ|={np.abs(analytic - numeric).max():.2e} "
            f"max_rel={rel.max():.2e} {'OK' if close else 'FAIL'}"
        )

    if verbose or (not ok):
        text = "gradcheck:\n" + "\n".join(report)
        if not ok and raise_exception:
            raise AssertionError(text)
        print(text)
    return ok
