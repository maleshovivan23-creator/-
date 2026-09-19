"""Готовые архитектуры: MLP, CNN и GPT-подобный трансформер."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import nn
from .tensor import Tensor


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


@dataclass
class GPTConfig:
    vocab_size: int = 128
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class GPT(nn.Module):
    """Декодер-трансформер с causal-attention и авторегрессионной генерацией."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [nn.TransformerBlock(cfg.n_embd, cfg.n_head, dropout=cfg.dropout, causal=True)
                       for _ in range(cfg.n_layer)]
        for i, b in enumerate(self.blocks):
            self._modules[f"block{i}"] = b
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx) -> Tensor:
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(int)
        b, t = idx.shape
        assert t <= self.cfg.block_size, "последовательность длиннее block_size"
        x = self.drop(self.tok_emb(idx) + self.pos_emb(np.arange(t))[None])
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def generate(self, prompt: Sequence[int], max_new_tokens: int = 100, temperature: float = 1.0,
                 top_k: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
        """Авторегрессионная генерация с temperature / top-k сэмплированием."""
        rng = np.random.default_rng(seed)
        was_training = self.training
        self.eval()
        seq = list(prompt)
        for _ in range(max_new_tokens):
            ctx = np.array([seq[-self.cfg.block_size:]], dtype=int)
            logits = self(ctx).data[0, -1] / max(temperature, 1e-6)
            if top_k is not None:
                thresh = np.sort(logits)[-top_k]
                logits = np.where(logits < thresh, -np.inf, logits)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            seq.append(int(rng.choice(len(p), p=p)))
        self.train(was_training)
        return seq
