"""Автоград-движок: Tensor с обратным распространением по динамическому графу.

Ядро всей библиотеки. Каждая операция строит узел графа и запоминает,
как протолкнуть градиент назад (замыкание `_backward`).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int, list]

# ------------------------------------------------------------------ случайность
_RNG = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Зафиксировать все источники случайности библиотеки (инициализация, dropout).

    >>> un.manual_seed(42)   # два запуска дадут идентичные веса и маски dropout
    """
    global _RNG
    _RNG = np.random.default_rng(seed)
    np.random.seed(seed)


def get_rng() -> np.random.Generator:
    """Текущий генератор библиотеки (используется слоями и dropout)."""
    return _RNG


# --------------------------------------------------------------- режим градиентов
_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class no_grad:
    """Контекст/декоратор, отключающий построение графа (быстрый инференс).

    >>> with no_grad():
    ...     logits = model(x)
    """

    def __enter__(self) -> "no_grad":
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc) -> bool:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False

    def __call__(self, fn):
        def wrapper(*a, **kw):
            with no_grad():
                return fn(*a, **kw)

        return wrapper


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Свернуть градиент к исходной форме операнда после broadcasting."""
    if grad.shape == shape:
        return grad
    # лишние ведущие оси
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # оси, растянутые из 1
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def _is_basic_index(idx) -> bool:
    """True, если индекс — базовый (срезы/int/None/Ellipsis) и не дублирует элементы."""
    items = idx if isinstance(idx, tuple) else (idx,)
    for i in items:
        if not (isinstance(i, (slice, int, np.integer)) or i is Ellipsis or i is None):
            return False
    return True


class Tensor:
    """N-мерный массив с автоматическим дифференцированием."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Sequence["Tensor"] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # ---------------------------------------------------------------- утилиты
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __repr__(self) -> str:  # pragma: no cover - косметика
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})\n{self.data}"

    def __len__(self) -> int:
        return self.shape[0]

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        self.grad = None

    @staticmethod
    def _ensure(x: ArrayLike) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    def _make(self, data: np.ndarray, parents: Sequence["Tensor"], op: str) -> "Tensor":
        req = _GRAD_ENABLED and any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=req, _children=parents if req else (), _op=op)

    def _accum(self, g: np.ndarray, shared: bool = True) -> None:
        """Накопить градиент.

        shared=True  — `g` может быть чужим буфером (out.grad или его view),
                       поэтому при первом присваивании делается копия.
        shared=False — `g` только что вычислен этой операцией и больше нигде
                       не используется, копию можно не делать (быстрый путь).
        """
        if not self.requires_grad:
            return
        # быстрый путь: форма уже совпадает (самый частый случай)
        if type(g) is np.ndarray and g.shape == self.data.shape:
            if g.dtype != np.float32:
                g = g.astype(np.float32, copy=False)
        else:
            g = _unbroadcast(np.asarray(g, dtype=np.float32), self.shape)
        if self.grad is None:
            # Копия обязательна: некоторые _backward отдают out.grad (или его view)
            # сразу нескольким родителям — без копии градиенты «слипнутся»,
            # и внешняя запись вида `p.grad *= s` повредит чужой тензор.
            self.grad = g.copy() if shared else g
        else:
            self.grad += g  # in-place: накопление без лишней аллокации

    # ------------------------------------------------------------- арифметика
    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._ensure(other)
        out = self._make(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self._accum(out.grad)
            other._accum(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._ensure(other)
        out = self._make(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self._accum(out.grad * other.data, shared=False)
            other._accum(out.grad * self.data, shared=False)

        out._backward = _backward
        return out

    def __pow__(self, p: float) -> "Tensor":
        out = self._make(self.data ** p, (self,), f"**{p}")

        def _backward() -> None:
            self._accum(out.grad * p * self.data ** (p - 1), shared=False)

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = self._ensure(other)
        out = self._make(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            g = out.grad
            a, b = self.data, other.data
            ga = g @ np.swapaxes(b, -1, -2)
            self._accum(_unbroadcast(ga, a.shape), shared=False)

            # Частый случай: batched-вход (B, T, K) на 2D-веса (K, N).
            # Наивно это batched matmul (B, K, N) с последующей суммой по
            # батчу — восьмикратно лишняя работа и лишний большой буфер.
            # Свёртка батча в строки даёт ровно тот же результат одним
            # вызовом BLAS (x2.6 на типичной форме).
            if b.ndim == 2 and a.ndim > 2:
                gb = a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])
            else:
                gb = _unbroadcast(np.swapaxes(a, -1, -2) @ g, b.shape)
            other._accum(gb, shared=False)

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-self._ensure(other))

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self * (self._ensure(other) ** -1.0)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._ensure(other) + (-self)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._ensure(other) * (self ** -1.0)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return self._ensure(other) @ self

    def __getitem__(self, idx) -> "Tensor":
        out = self._make(self.data[idx], (self,), "getitem")
        # Базовая индексация (срезы/int/Ellipsis) не может повторять элементы,
        # поэтому градиент пишется прямым присваиванием — np.add.at на порядок медленнее
        # и нужен только для «фантазийной» индексации массивами.
        basic = _is_basic_index(idx)

        def _backward() -> None:
            g = np.zeros_like(self.data)
            if basic:
                g[idx] = out.grad
            else:
                np.add.at(g, idx, out.grad)
            self._accum(g, shared=False)

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- формы
    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,), "reshape")

        def _backward() -> None:
            self._accum(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def view(self, *shape: int) -> "Tensor":
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 1) -> "Tensor":
        shape = self.shape[:start_dim] + (-1,)
        return self.reshape(*shape)

    def transpose(self, *axes: int) -> "Tensor":
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        out = self._make(self.data.transpose(axes), (self,), "transpose")
        inv = np.argsort(axes)

        def _backward() -> None:
            self._accum(out.grad.transpose(inv))

        out._backward = _backward
        return out

    def swapaxes(self, a: int, b: int) -> "Tensor":
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    # ---------------------------------------------------------------- редукции
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape).copy(), shared=False)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        n = self.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def var(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        mu = self.mean(axis=axis, keepdims=True)
        return ((self - mu) ** 2).mean(axis=axis, keepdims=keepdims)

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        res = self.data.max(axis=axis, keepdims=True)
        out_data = res if keepdims else res.squeeze(axis) if axis is not None else res.reshape(())
        out = self._make(out_data, (self,), "max")
        mask = (self.data == res).astype(np.float32)
        mask /= mask.sum(axis=axis, keepdims=True)

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(mask * g, shared=False)

        out._backward = _backward
        return out

    # -------------------------------------------------------------- нелинейности
    def exp(self) -> "Tensor":
        out = self._make(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self._accum(out.grad * out.data, shared=False)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = self._make(np.log(np.clip(self.data, 1e-12, None)), (self,), "log")

        def _backward() -> None:
            self._accum(out.grad / np.clip(self.data, 1e-12, None), shared=False)

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        return self ** 0.5

    def relu(self) -> "Tensor":
        out = self._make(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            self._accum(out.grad * (self.data > 0), shared=False)

        out._backward = _backward
        return out

    def leaky_relu(self, slope: float = 0.01) -> "Tensor":
        out = self._make(np.where(self.data > 0, self.data, slope * self.data), (self,), "leaky_relu")

        def _backward() -> None:
            self._accum(out.grad * np.where(self.data > 0, 1.0, slope), shared=False)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = self._make(t, (self,), "tanh")

        def _backward() -> None:
            self._accum(out.grad * (1 - t * t), shared=False)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._make(s, (self,), "sigmoid")

        def _backward() -> None:
            self._accum(out.grad * s * (1 - s), shared=False)

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """Приближение tanh-GELU (как в GPT-2)."""
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        # x*x*x вместо x**3: NumPy не сворачивает целую степень для float32
        # и уходит в общий pow — это в ~96 раз медленнее (16.3 мс против 0.17 мс
        # на массиве 8x64x512). Здесь это было 40% времени шага обучения.
        x2 = x * x
        inner = c * (x + 0.044715 * (x2 * x))
        t = np.tanh(inner)
        out_data = 0.5 * x * (1 + t)
        out = self._make(out_data, (self,), "gelu")

        def _backward() -> None:
            dinner = c * (1 + 3 * 0.044715 * x2)
            d = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner
            self._accum(out.grad * d, shared=False)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Fused softmax: один узел графа вместо цепочки exp/sum/div."""
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = self._make(p, (self,), "softmax")

        def _backward() -> None:
            g = out.grad
            self._accum(p * (g - (g * p).sum(axis=axis, keepdims=True)), shared=False)

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """Fused log-softmax — численно устойчив и дешевле по графу."""
        z = self.data - self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
        logp = z - lse
        out = self._make(logp, (self,), "log_softmax")
        p = np.exp(logp)

        def _backward() -> None:
            g = out.grad
            self._accum(g - p * g.sum(axis=axis, keepdims=True), shared=False)

        out._backward = _backward
        return out

    def silu(self) -> "Tensor":
        """SiLU / Swish: x * sigmoid(x) — используется в LLaMA-style MLP."""
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._make(self.data * s, (self,), "silu")

        def _backward() -> None:
            self._accum(out.grad * (s * (1 + self.data * (1 - s))), shared=False)

        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = self._make(np.abs(self.data), (self,), "abs")

        def _backward() -> None:
            self._accum(out.grad * np.sign(self.data), shared=False)

        out._backward = _backward
        return out

    def sin(self) -> "Tensor":
        out = self._make(np.sin(self.data), (self,), "sin")

        def _backward() -> None:
            self._accum(out.grad * np.cos(self.data), shared=False)

        out._backward = _backward
        return out

    def cos(self) -> "Tensor":
        out = self._make(np.cos(self.data), (self,), "cos")

        def _backward() -> None:
            self._accum(-out.grad * np.sin(self.data), shared=False)

        out._backward = _backward
        return out

    def masked_fill(self, mask: np.ndarray, value: float) -> "Tensor":
        """mask=True -> подставить value (используется в causal-attention)."""
        keep = (~mask).astype(np.float32)
        out = self._make(np.where(mask, value, self.data), (self,), "masked_fill")

        def _backward() -> None:
            self._accum(out.grad * keep, shared=False)

        out._backward = _backward
        return out

    def dropout(self, p: float, training: bool = True) -> "Tensor":
        if p <= 0.0 or not training:
            return self
        keep = (_RNG.random(self.shape) >= p).astype(np.float32) / (1.0 - p)
        out = self._make(self.data * keep, (self,), "dropout")

        def _backward() -> None:
            self._accum(out.grad * keep, shared=False)

        out._backward = _backward
        return out

    def clip(self, lo: float, hi: float) -> "Tensor":
        out = self._make(np.clip(self.data, lo, hi), (self,), "clip")
        inside = ((self.data >= lo) & (self.data <= hi)).astype(np.float32)

        def _backward() -> None:
            self._accum(out.grad * inside, shared=False)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ backward
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Обратный проход в топологическом порядке."""
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        if grad is None:
            if self.size != 1:
                raise RuntimeError("backward() без grad возможен только для скаляра")
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=np.float32)

        for node in reversed(topo):
            if node.grad is None:
                continue  # ветка, не получившая градиента (константы, отсечённые пути)
            node._backward()


# --------------------------------------------------------------- конструкторы
def tensor(data: ArrayLike, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False) -> Tensor:
    return Tensor(_RNG.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)


def arange(n: int) -> Tensor:
    return Tensor(np.arange(n, dtype=np.float32))


def stack(items: Iterable[Tensor], axis: int = 0) -> Tensor:
    items = list(items)
    out = Tensor(
        np.stack([t.data for t in items], axis=axis),
        requires_grad=any(t.requires_grad for t in items),
        _children=items,
        _op="stack",
    )

    def _backward() -> None:
        grads = np.split(out.grad, len(items), axis=axis)
        for t, g in zip(items, grads):
            t._accum(np.squeeze(g, axis=axis))

    out._backward = _backward
    return out


def cat(items: Sequence[Tensor], axis: int = -1) -> Tensor:
    items = list(items)
    sizes = [t.shape[axis] for t in items]
    out = Tensor(
        np.concatenate([t.data for t in items], axis=axis),
        requires_grad=any(t.requires_grad for t in items),
        _children=items,
        _op="cat",
    )

    def _backward() -> None:
        idx = np.cumsum(sizes)[:-1]
        for t, g in zip(items, np.split(out.grad, idx, axis=axis)):
            t._accum(g)

    out._backward = _backward
    return out
