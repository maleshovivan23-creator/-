"""Слои и модули (аналог torch.nn), построенные поверх автограда Tensor."""
from __future__ import annotations

import math
import pickle
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .tensor import Tensor, cat, get_rng


class Parameter(Tensor):
    """Тензор-параметр: всегда requires_grad=True."""

    def __init__(self, data) -> None:
        super().__init__(data, requires_grad=True)


# ------------------------------------------------------------------ инициализация
def kaiming(fan_in: int, fan_out: int, gain: float = 2.0) -> np.ndarray:
    return (get_rng().standard_normal((fan_in, fan_out)) * math.sqrt(gain / fan_in)).astype(np.float32)


def xavier(fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return get_rng().uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


class Module:
    """Базовый модуль: рекурсивный обход параметров, режимы train/eval, сохранение."""

    def __init__(self) -> None:
        self._modules: Dict[str, "Module"] = {}
        self._params: Dict[str, Parameter] = {}
        self.training = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, Module) for v in value):
            for i, v in enumerate(value):
                self.__dict__.setdefault("_modules", {})[f"{name}.{i}"] = v
        object.__setattr__(self, name, value)

    def parameters(self) -> List[Parameter]:
        out = list(self._params.values())
        for m in self._modules.values():
            out.extend(m.parameters())
        return out

    def named_parameters(self, prefix: str = "") -> Iterator[tuple]:
        for k, v in self._params.items():
            yield f"{prefix}{k}", v
        for name, m in self._modules.items():
            yield from m.named_parameters(f"{prefix}{name}.")

    def num_params(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.data.copy() for k, v in self.named_parameters()}

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        for k, v in sd.items():
            if k in params:
                params[k].data = np.asarray(v, dtype=np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.load_state_dict(pickle.load(f))

    def summary(self, max_rows: int = 40) -> str:
        """Таблица параметров модели — быстрый аудит архитектуры."""
        rows = [(name, tuple(p.shape), p.size) for name, p in self.named_parameters()]
        total = sum(r[2] for r in rows)
        w = max([len(r[0]) for r in rows] + [9])
        head = f"{'параметр'.ljust(w)}  {'форма':<18} {'кол-во':>12}"
        lines = [head, "-" * len(head)]
        for name, shape, size in rows[:max_rows]:
            lines.append(f"{name.ljust(w)}  {str(shape):<18} {size:>12,}")
        if len(rows) > max_rows:
            lines.append(f"... ещё {len(rows) - max_rows} параметров")
        lines.append("-" * len(head))
        lines.append(f"{'ИТОГО'.ljust(w)}  {'':<18} {total:>12,}")
        mb = total * 4 / 1024 ** 2
        lines.append(f"{'память (float32)'.ljust(w)}  {'':<18} {mb:>11.2f}M")
        return "\n".join(lines)

    def forward(self, *a, **kw):  # pragma: no cover - абстрактный
        raise NotImplementedError

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)


# ----------------------------------------------------------------------- слои
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(kaiming(in_features, out_features))
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"Linear: ожидался вход с последней размерностью {self.weight.shape[0]}, "
                f"получен тензор {x.shape}. Проверьте in_features слоя."
            )
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int) -> None:
        super().__init__()
        self.weight = Parameter(get_rng().standard_normal((num_embeddings, dim)).astype(np.float32) * 0.02)

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(int)
        if idx.size:
            lo, hi = int(idx.min()), int(idx.max())
            n = self.weight.shape[0]
            if lo < 0 or hi >= n:
                raise IndexError(
                    f"Embedding: индексы должны быть в [0, {n - 1}], "
                    f"получен диапазон [{lo}, {hi}]. Увеличьте num_embeddings "
                    f"или проверьте словарь токенизатора."
                )
        return self.weight[idx]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=np.float32))
        self.beta = Parameter(np.zeros(dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc ** 2).mean(axis=-1, keepdims=True)
        return xc / (var + self.eps).sqrt() * self.gamma + self.beta


class RMSNorm(Module):
    """RMSNorm (LLaMA/T5): нормировка без вычитания среднего — быстрее LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + self.eps).sqrt()
        return x / rms * self.gamma


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = Parameter(np.ones(dim, dtype=np.float32))
        self.beta = Parameter(np.zeros(dim, dtype=np.float32))
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var = np.ones(dim, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = x.mean(axis=0, keepdims=True)
            xc = x - mu
            var = (xc ** 2).mean(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.data.squeeze(0)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.squeeze(0)
            norm = xc / (var + self.eps).sqrt()
        else:
            norm = (x - Tensor(self.running_mean)) / Tensor(self.running_var + self.eps).sqrt()
        return norm * self.gamma + self.beta


class Dropout(Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x.dropout(self.p, self.training)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)
        for i, l in enumerate(layers):
            self._modules[str(i)] = l

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Residual(Module):
    """x + f(x) — обёртка для остаточных связей."""

    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


# ------------------------------------------------------------------- свёртки
def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    n, c, h, w = x.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, -1), oh, ow


def _col2im(cols: np.ndarray, shape, kh, kw, stride, pad, oh, ow) -> np.ndarray:
    n, c, h, w = shape
    cols = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += cols[:, :, i, j]
    return xp[:, :, pad:pad + h, pad:pad + w]


class Conv2d(Module):
    """Свёртка через im2col, форма входа (N, C, H, W)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.k, self.stride, self.pad = kernel_size, stride, padding
        fan_in = in_ch * kernel_size * kernel_size
        self.weight = Parameter(get_rng().standard_normal((out_ch, fan_in)).astype(np.float32) * math.sqrt(2.0 / fan_in))
        self.bias = Parameter(np.zeros(out_ch, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        cols, oh, ow = _im2col(x.data, self.k, self.k, self.stride, self.pad)
        w = self.weight
        out_data = cols @ w.data.T
        if self.bias is not None:
            out_data = out_data + self.bias.data
        out_data = out_data.reshape(n, oh, ow, self.out_ch).transpose(0, 3, 1, 2)
        parents = [p for p in (x, w, self.bias) if p is not None]
        out = x._make(out_data, parents, "conv2d")
        xshape = x.shape

        def _backward() -> None:
            g = out.grad.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
            w._accum(g.T @ cols)
            if self.bias is not None:
                self.bias._accum(g.sum(0))
            if x.requires_grad:
                dcols = g @ w.data
                x._accum(_col2im(dcols, xshape, self.k, self.k, self.stride, self.pad, oh, ow))

        out._backward = _backward
        return out


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None) -> None:
        super().__init__()
        self.k = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        k, s = self.k, self.stride
        oh, ow = (h - k) // s + 1, (w - k) // s + 1
        strided = np.lib.stride_tricks.as_strided(
            x.data,
            shape=(n, c, oh, ow, k, k),
            strides=x.data.strides[:2] + (x.data.strides[2] * s, x.data.strides[3] * s) + x.data.strides[2:],
        )
        flat = strided.reshape(n, c, oh, ow, k * k)
        arg = flat.argmax(-1)
        out = x._make(flat.max(-1), (x,), "maxpool")

        def _backward() -> None:
            gx = np.zeros_like(x.data)
            ii, jj = np.divmod(arg, k)
            nidx, cidx, hidx, widx = np.indices((n, c, oh, ow))
            np.add.at(gx, (nidx, cidx, hidx * s + ii, widx * s + jj), out.grad)
            x._accum(gx)

        out._backward = _backward
        return out


class AvgPool2d(Module):
    """Усредняющий пулинг (N, C, H, W)."""

    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None) -> None:
        super().__init__()
        self.k = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        k, s = self.k, self.stride
        oh, ow = (h - k) // s + 1, (w - k) // s + 1
        strided = np.lib.stride_tricks.as_strided(
            x.data,
            shape=(n, c, oh, ow, k, k),
            strides=x.data.strides[:2] + (x.data.strides[2] * s, x.data.strides[3] * s) + x.data.strides[2:],
        )
        out = x._make(strided.reshape(n, c, oh, ow, k * k).mean(-1), (x,), "avgpool")

        def _backward() -> None:
            gx = np.zeros_like(x.data)
            g = out.grad / (k * k)
            for i in range(k):
                for j in range(k):
                    gx[:, :, i:i + s * oh:s, j:j + s * ow:s] += g
            x._accum(gx)

        out._backward = _backward
        return out


# ---------------------------------------------------------------- внимание
class RotaryEmbedding(Module):
    """RoPE — вращательное позиционное кодирование (LLaMA, GPT-NeoX).

    Кодирует относительные позиции прямо в q/k, поэтому модель лучше
    обобщается на длины, не встречавшиеся при обучении.
    """

    def __init__(self, head_dim: int, max_seq: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim должен быть чётным для RoPE"
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        self.cos = np.cos(freqs).astype(np.float32)  # (max_seq, head_dim/2)
        self.sin = np.sin(freqs).astype(np.float32)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """x: (B, H, T, head_dim) -> повёрнутый тензор той же формы."""
        t = x.shape[2]
        cos = self.cos[offset:offset + t][None, None]  # (1,1,T,hd/2)
        sin = self.sin[offset:offset + t][None, None]
        x1 = x[:, :, :, 0::2]
        x2 = x[:, :, :, 1::2]
        rot1 = x1 * Tensor(cos) - x2 * Tensor(sin)
        rot2 = x1 * Tensor(sin) + x2 * Tensor(cos)
        # обратное чередование: (..., hd/2, 2) -> (..., hd)
        stacked = cat([rot1.reshape(*rot1.shape, 1), rot2.reshape(*rot2.shape, 1)], axis=-1)
        return stacked.reshape(*x.shape)


class MultiHeadAttention(Module):
    """Многоголовое самовнимание: causal-маска, RoPE и KV-кэш для генерации."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, causal: bool = True,
                 rope: bool = False, max_seq: int = 4096) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim должен делиться на n_heads"
        self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
        self.causal = causal
        self.qkv = Linear(dim, 3 * dim, bias=False)
        self.proj = Linear(dim, dim)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq) if rope else None
        self._cache: Optional[tuple] = None

    def reset_cache(self) -> None:
        self._cache = None

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        b, t, c = x.shape
        # один reshape+transpose вместо трёх срезов: (B,T,3C) -> (3,B,H,T,hd)
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        offset = self._cache[0].shape[2] if (use_cache and self._cache is not None) else 0
        if self.rope is not None:
            q, k = self.rope(q, offset), self.rope(k, offset)

        if use_cache:
            if self._cache is not None:
                pk, pv = self._cache
                k = cat([Tensor(pk), k], axis=2)
                v = cat([Tensor(pv), v], axis=2)
            self._cache = (k.data, v.data)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))
        if self.causal:
            tk = k.shape[2]
            # при генерации с кэшем запрос видит весь префикс, маска сдвигается на offset
            mask = np.triu(np.ones((t, tk), dtype=bool), k=1 + offset)[None, None]
            att = att.masked_fill(np.broadcast_to(mask, att.shape), -1e9)
        att = self.attn_drop(att.softmax(axis=-1))
        y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.resid_drop(self.proj(y))


class SwiGLU(Module):
    """SwiGLU-MLP из LLaMA: gate * silu(up) — сильнее обычного GELU-MLP."""

    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden or int(8 * dim / 3 // 32 * 32) or 4 * dim
        self.w_gate = Linear(dim, hidden, bias=False)
        self.w_up = Linear(dim, hidden, bias=False)
        self.w_down = Linear(hidden, dim, bias=False)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.w_down(self.w_gate(x).silu() * self.w_up(x)))


class TransformerBlock(Module):
    """Pre-LN блок трансформера: attention + MLP с остаточными связями.

    Поддерживает современный режим (RMSNorm + RoPE + SwiGLU) и классический
    GPT-2 (LayerNorm + выученные позиции + GELU-MLP).
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0,
                 causal: bool = True, rope: bool = False, norm: str = "layer",
                 swiglu: bool = False, max_seq: int = 4096) -> None:
        super().__init__()
        make_norm = (lambda: RMSNorm(dim)) if norm == "rms" else (lambda: LayerNorm(dim))
        self.ln1 = make_norm()
        self.attn = MultiHeadAttention(dim, n_heads, dropout, causal, rope=rope, max_seq=max_seq)
        self.ln2 = make_norm()
        if swiglu:
            self.mlp = SwiGLU(dim, dropout=dropout)
        else:
            self.mlp = Sequential(
                Linear(dim, mlp_ratio * dim),
                GELU(),
                Linear(mlp_ratio * dim, dim),
                Dropout(dropout),
            )

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        return x + self.mlp(self.ln2(x))


# -------------------------------------------------------------- рекуррентность
class RNNCell(Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.wx = Linear(in_dim, hidden, bias=False)
        self.wh = Linear(hidden, hidden)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return (self.wx(x) + self.wh(h)).tanh()


class LSTMCell(Module):
    """LSTM-ячейка с входным/забывающим/выходным вентилями."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.x2h = Linear(in_dim, 4 * hidden)
        self.h2h = Linear(hidden, 4 * hidden, bias=False)
        self.hidden = hidden

    def forward(self, x: Tensor, state) -> tuple:
        h, c = state
        g = self.x2h(x) + self.h2h(h)
        n = self.hidden
        i = g[:, 0:n].sigmoid()
        f = g[:, n:2 * n].sigmoid()
        o = g[:, 2 * n:3 * n].sigmoid()
        u = g[:, 3 * n:4 * n].tanh()
        c_new = f * c + i * u
        return o * c_new.tanh(), c_new


class GRUCell(Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.xz, self.hz = Linear(in_dim, hidden), Linear(hidden, hidden, bias=False)
        self.xr, self.hr = Linear(in_dim, hidden), Linear(hidden, hidden, bias=False)
        self.xn, self.hn = Linear(in_dim, hidden), Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = (self.xz(x) + self.hz(h)).sigmoid()
        r = (self.xr(x) + self.hr(h)).sigmoid()
        n = (self.xn(x) + r * self.hn(h)).tanh()
        return (1 - z) * n + z * h


__all__ = [
    "Parameter", "Module", "Linear", "Embedding", "LayerNorm", "RMSNorm", "BatchNorm1d",
    "Dropout", "ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "Flatten", "Sequential", "Residual",
    "Conv2d", "MaxPool2d", "AvgPool2d", "RotaryEmbedding", "MultiHeadAttention", "SwiGLU",
    "TransformerBlock", "RNNCell", "GRUCell", "LSTMCell",
]
