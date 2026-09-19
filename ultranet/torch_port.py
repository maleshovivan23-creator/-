"""Порт GPT на PyTorch: тот же счёт, те же веса, те же сэмплы.

NumPy-движок остаётся референсом. Эта версия нужна только чтобы обучать
на GPU, и она обязана совпадать с референсом численно — иначе обучится
не та модель, которую мы проверяли.

Пять мест, где совпадение легко потерять (все учтены ниже):

1. Linear хранит вес как (in, out), у torch — (out, in). При переносе
   нужен транспонс, иначе матрица перемножится наоборот.
2. RMSNorm в референсе делит на sqrt(mean(x^2) + eps), а torch-версии
   часто пишут sqrt(mean(x^2)) + eps или rsqrt. Это разные числа.
3. RoPE в референсе чередует соседние элементы (0,1), (2,3)...,
   а в HF-моделях половинки (i, i+hd/2). Порядок менять нельзя.
4. Causal-маска заполняется -1e9, а не -inf: -inf даёт NaN в softmax
   на полностью замаскированных строках.
5. GELU — приближение tanh, а не точная erf-версия.

Использование:
    from ultranet.torch_port import TorchGPT, convert_weights
    tm = TorchGPT(cfg)
    convert_weights(numpy_model, tm)      # веса référence -> torch
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as tnn
    import torch.nn.functional as TF
except ImportError as e:  # pragma: no cover
    raise ImportError("нужен torch: pip install torch") from e


# ═════════════════════════════════════════════════════════════ слои
class RMSNorm(tnn.Module):
    """Точно как референс: x / sqrt(mean(x^2) + eps) * gamma."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = tnn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


class LayerNorm(tnn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = tnn.Parameter(torch.ones(dim))
        self.beta = tnn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gamma + self.beta


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Приближение tanh-GELU — как в референсе (не erf!)."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x * x * x)))


class RotaryEmbedding(tnn.Module):
    """RoPE с чередованием соседних элементов — как в референсе."""

    def __init__(self, head_dim: int, max_seq: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv)
        self.register_buffer("cos", torch.tensor(np.cos(freqs), dtype=torch.float32))
        self.register_buffer("sin", torch.tensor(np.sin(freqs), dtype=torch.float32))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        t = x.shape[2]
        cos = self.cos[offset:offset + t][None, None]
        sin = self.sin[offset:offset + t][None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        return torch.stack((r1, r2), dim=-1).flatten(-2)


class MultiHeadAttention(tnn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, causal: bool = True,
                 rope: bool = False, max_seq: int = 4096) -> None:
        super().__init__()
        self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
        self.causal = causal
        self.qkv = tnn.Linear(dim, 3 * dim, bias=False)
        self.proj = tnn.Linear(dim, dim, bias=True)
        self.drop_p = dropout
        self.rope = RotaryEmbedding(self.head_dim, max_seq) if rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)

        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        if self.causal:
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
            # -1e9, а не -inf: -inf даёт NaN на полностью замаскированных строках
            att = att.masked_fill(mask, -1e9)
        att = torch.softmax(att, dim=-1)
        att = TF.dropout(att, self.drop_p, self.training)
        y = (att @ v).transpose(1, 2).reshape(b, t, c)
        return TF.dropout(self.proj(y), self.drop_p, self.training)


class SwiGLU(tnn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden or int(8 * dim / 3 // 32 * 32) or 4 * dim
        self.w_gate = tnn.Linear(dim, hidden, bias=False)
        self.w_up = tnn.Linear(dim, hidden, bias=False)
        self.w_down = tnn.Linear(hidden, dim, bias=False)
        self.drop_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = TF.silu(self.w_gate(x)) * self.w_up(x)
        return TF.dropout(self.w_down(h), self.drop_p, self.training)


class GELUMLP(tnn.Module):
    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = tnn.Linear(dim, ratio * dim, bias=True)
        self.fc2 = tnn.Linear(ratio * dim, dim, bias=True)
        self.drop_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TF.dropout(self.fc2(gelu_tanh(self.fc1(x))), self.drop_p, self.training)


class TransformerBlock(tnn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float, causal: bool,
                 rope: bool, norm: str, swiglu: bool, max_seq: int) -> None:
        super().__init__()
        mk = (lambda: RMSNorm(dim)) if norm == "rms" else (lambda: LayerNorm(dim))
        self.ln1 = mk()
        self.attn = MultiHeadAttention(dim, n_heads, dropout, causal, rope, max_seq)
        self.ln2 = mk()
        self.mlp = SwiGLU(dim, dropout=dropout) if swiglu else GELUMLP(dim, 4, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TorchGPT(tnn.Module):
    """GPT с той же арифметикой, что у NumPy-референса."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = tnn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = None if cfg.rope else tnn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop_p = cfg.dropout
        self.blocks = tnn.ModuleList([
            TransformerBlock(cfg.n_embd, cfg.n_head, cfg.dropout, True,
                             cfg.rope, cfg.norm, cfg.swiglu, cfg.block_size * 4)
            for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd) if cfg.norm == "rms" else LayerNorm(cfg.n_embd)
        self.head = None if cfg.tie_weights else tnn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(t, device=idx.device)
            x = x + self.pos_emb(pos)[None]
        x = TF.dropout(x, self.drop_p, self.training)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T if self.head is None else self.head(x)
        if targets is None:
            return logits
        loss = TF.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ═══════════════════════════════════════════════════ перенос весов
def _T(w: np.ndarray) -> torch.Tensor:
    """(in, out) референса -> (out, in) torch."""
    return torch.tensor(np.ascontiguousarray(w.T), dtype=torch.float32)


def _V(w: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(w), dtype=torch.float32)


@torch.no_grad()
def convert_weights(np_model, t_model: TorchGPT) -> TorchGPT:
    """Перенести веса из NumPy-модели в torch-модель — один в один."""
    t_model.tok_emb.weight.copy_(_V(np_model.tok_emb.weight.data))
    if np_model.pos_emb is not None and t_model.pos_emb is not None:
        t_model.pos_emb.weight.copy_(_V(np_model.pos_emb.weight.data))

    for nb, tb in zip(np_model.blocks, t_model.blocks):
        _copy_norm(nb.ln1, tb.ln1)
        _copy_norm(nb.ln2, tb.ln2)
        tb.attn.qkv.weight.copy_(_T(nb.attn.qkv.weight.data))
        tb.attn.proj.weight.copy_(_T(nb.attn.proj.weight.data))
        if nb.attn.proj.bias is not None:
            tb.attn.proj.bias.copy_(_V(nb.attn.proj.bias.data))
        if isinstance(tb.mlp, SwiGLU):
            tb.mlp.w_gate.weight.copy_(_T(nb.mlp.w_gate.weight.data))
            tb.mlp.w_up.weight.copy_(_T(nb.mlp.w_up.weight.data))
            tb.mlp.w_down.weight.copy_(_T(nb.mlp.w_down.weight.data))
        else:
            lin = [m for m in nb.mlp.layers if hasattr(m, "weight")]
            tb.mlp.fc1.weight.copy_(_T(lin[0].weight.data))
            tb.mlp.fc1.bias.copy_(_V(lin[0].bias.data))
            tb.mlp.fc2.weight.copy_(_T(lin[1].weight.data))
            tb.mlp.fc2.bias.copy_(_V(lin[1].bias.data))

    _copy_norm(np_model.ln_f, t_model.ln_f)
    if np_model.head is not None and t_model.head is not None:
        t_model.head.weight.copy_(_T(np_model.head.weight.data))
    return t_model


@torch.no_grad()
def _copy_norm(n_norm, t_norm) -> None:
    if hasattr(n_norm, "gamma"):
        t_norm.gamma.copy_(_V(n_norm.gamma.data))
    if hasattr(n_norm, "beta") and hasattr(t_norm, "beta"):
        t_norm.beta.copy_(_V(n_norm.beta.data))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())
