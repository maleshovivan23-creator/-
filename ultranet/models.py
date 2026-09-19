"""Готовые архитектуры: MLP, CNN и GPT-подобный трансформер."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import nn
from .tensor import Tensor, no_grad


class MLP(nn.Module):
    """Полносвязная сеть произвольной глубины."""

    def __init__(self, sizes: Sequence[int], activation: str = "relu", dropout: float = 0.0,
                 batchnorm: bool = False) -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(act())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvNet(nn.Module):
    """Небольшая CNN для изображений (N, C, H, W)."""

    def __init__(self, in_ch: int = 1, n_classes: int = 10, width: int = 16, img_size: int = 28) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        s = img_size // 4
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 2 * s * s, 128), nn.ReLU(),
                                  nn.Linear(128, n_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class ResBlock(nn.Module):
    """Остаточный блок: Conv-BN-ReLU x2 + skip (упрощённый ResNet-v1)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.c1(x).relu()
        return (x + self.c2(h)).relu()


class ResNet(nn.Module):
    """Небольшая ResNet для изображений (N, C, H, W)."""

    def __init__(self, in_ch: int = 1, n_classes: int = 10, width: int = 16,
                 n_blocks: int = 2, img_size: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1)
        self.blocks = [ResBlock(width) for _ in range(n_blocks)]
        for i, b in enumerate(self.blocks):
            self._modules[f"res{i}"] = b
        self.pool = nn.AvgPool2d(2)
        s = img_size // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * s * s, n_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x).relu()
        for b in self.blocks:
            x = b(x)
        return self.head(self.pool(x))


class TextClassifier(nn.Module):
    """Трансформер-энкодер (без causal-маски) + среднее по токенам -> классы."""

    def __init__(self, vocab_size: int, n_classes: int, n_embd: int = 64, n_layer: int = 2,
                 n_head: int = 4, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(max_len, n_embd)
        self.blocks = [nn.TransformerBlock(n_embd, n_head, dropout=dropout, causal=False)
                       for _ in range(n_layer)]
        for i, b in enumerate(self.blocks):
            self._modules[f"block{i}"] = b
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_classes)

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(int)
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(np.arange(t))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x).mean(axis=1))


@dataclass
class GPTConfig:
    """Конфигурация GPT. По умолчанию — классический GPT-2 style.

    Для современного LLaMA-style укажите rope=True, norm="rms", swiglu=True.
    """

    vocab_size: int = 128
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    rope: bool = False           # вращательные позиционные эмбеддинги
    norm: str = "layer"          # "layer" | "rms"
    swiglu: bool = False         # SwiGLU-MLP вместо GELU-MLP
    tie_weights: bool = True     # общая матрица эмбеддингов и выходной головы

    @classmethod
    def llama_style(cls, vocab_size: int, **kw) -> "GPTConfig":
        base = dict(vocab_size=vocab_size, rope=True, norm="rms", swiglu=True)
        base.update(kw)
        return cls(**base)


class GPT(nn.Module):
    """Декодер-трансформер: causal-attention, RoPE/RMSNorm/SwiGLU, KV-кэш."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = None if cfg.rope else nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [
            nn.TransformerBlock(cfg.n_embd, cfg.n_head, dropout=cfg.dropout, causal=True,
                                rope=cfg.rope, norm=cfg.norm, swiglu=cfg.swiglu,
                                max_seq=cfg.block_size * 4)
            for _ in range(cfg.n_layer)
        ]
        for i, b in enumerate(self.blocks):
            self._modules[f"block{i}"] = b
        self.ln_f = nn.RMSNorm(cfg.n_embd) if cfg.norm == "rms" else nn.LayerNorm(cfg.n_embd)
        if cfg.tie_weights:
            # weight tying: голова переиспользует матрицу эмбеддингов (экономит vocab*n_embd)
            self.head = None
        else:
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # масштабированная инициализация остаточных проекций (GPT-2 § 2.3)
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            blk.attn.proj.weight.data *= scale

    def _logits(self, x: Tensor) -> Tensor:
        if self.head is not None:
            return self.head(x)
        return x @ self.tok_emb.weight.transpose()

    def forward(self, idx, use_cache: bool = False, pos_offset: int = 0) -> Tensor:
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(int)
        b, t = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            assert pos_offset + t <= self.cfg.block_size, "последовательность длиннее block_size"
            x = x + self.pos_emb(np.arange(pos_offset, pos_offset + t))[None]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        return self._logits(self.ln_f(x))

    def reset_cache(self) -> None:
        for block in self.blocks:
            block.attn.reset_cache()

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float, top_k: Optional[int],
                top_p: Optional[float], repetition_penalty: float,
                seen: Sequence[int], rng) -> int:
        logits = logits.astype(np.float64)
        if repetition_penalty != 1.0 and seen:
            uniq = np.unique(np.asarray(seen, dtype=int))
            pos = logits[uniq] > 0
            logits[uniq] = np.where(pos, logits[uniq] / repetition_penalty,
                                    logits[uniq] * repetition_penalty)
        if temperature <= 1e-6:
            return int(logits.argmax())
        logits = logits / temperature
        if top_k is not None and top_k < len(logits):
            thresh = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < thresh, -np.inf, logits)
        if top_p is not None and 0 < top_p < 1.0:
            order = np.argsort(-logits)
            probs = np.exp(logits[order] - logits[order].max())
            probs /= probs.sum()
            cum = np.cumsum(probs)
            cutoff = int(np.searchsorted(cum, top_p) + 1)
            keep = order[:cutoff]
            masked = np.full_like(logits, -np.inf)
            masked[keep] = logits[keep]
            logits = masked
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def generate(self, prompt: Sequence[int], max_new_tokens: int = 100, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 repetition_penalty: float = 1.0, use_cache: bool = True,
                 stop_tokens: Optional[Sequence[int]] = None,
                 seed: Optional[int] = None) -> List[int]:
        """Авторегрессионная генерация: temperature, top-k, nucleus (top-p), KV-кэш.

        temperature=0 даёт жадный детерминированный декодинг.
        """
        rng = np.random.default_rng(seed)
        was_training = self.training
        self.eval()
        self.reset_cache()
        seq = list(prompt)
        stop = set(stop_tokens or ())

        with no_grad():
            if use_cache:
                ctx = np.array([seq[-self.cfg.block_size:]], dtype=int)
                logits = self(ctx, use_cache=True).data[0, -1]
                pos = len(seq)
                for _ in range(max_new_tokens):
                    nxt = self._sample(logits, temperature, top_k, top_p,
                                       repetition_penalty, seq, rng)
                    seq.append(nxt)
                    if nxt in stop:
                        break
                    if pos >= self.cfg.block_size:  # кэш переполнен — пересобираем
                        self.reset_cache()
                        ctx = np.array([seq[-self.cfg.block_size:]], dtype=int)
                        logits = self(ctx, use_cache=True).data[0, -1]
                        pos = len(seq[-self.cfg.block_size:])
                    else:
                        logits = self(np.array([[nxt]], dtype=int), use_cache=True,
                                      pos_offset=pos).data[0, -1]
                        pos += 1
            else:
                for _ in range(max_new_tokens):
                    ctx = np.array([seq[-self.cfg.block_size:]], dtype=int)
                    logits = self(ctx).data[0, -1]
                    nxt = self._sample(logits, temperature, top_k, top_p,
                                       repetition_penalty, seq, rng)
                    seq.append(nxt)
                    if nxt in stop:
                        break

        self.reset_cache()
        self.train(was_training)
        return seq

    @no_grad()
    def loss_on(self, x: np.ndarray, y: np.ndarray) -> float:
        """Кросс-энтропия на батче без построения графа."""
        from .functional import cross_entropy
        return cross_entropy(self(x), y).item()
