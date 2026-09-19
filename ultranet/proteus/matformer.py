"""MatFormer — вложенная модель: одни веса, много размеров.

Реализует ключевое заявление Протея: «Вложенность. Одна модель — все размеры.
Не нужно хранить разные версии».

Подмодель ширины w использует левый верхний угол каждой матрицы:
W[:in*w, :out*w]. Обучение идёт сразу на нескольких ширинах (matryoshka
training), поэтому любой срез остаётся рабочей моделью.

Дополнительно: ранний выход (early exit) — простые входы не проходят всю сеть.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import nn
from ..nn import Parameter
from ..tensor import Tensor, get_rng, no_grad


@dataclass
class MatConfig:
    """Конфигурация вложенной модели."""

    vocab_size: int = 260          # байты + спецтокены
    block_size: int = 128
    n_layer: int = 6
    n_embd: int = 128
    head_dim: int = 32
    mlp_ratio: int = 4
    dropout: float = 0.0
    widths: Tuple[float, ...] = (0.25, 0.5, 1.0)   # обучаемые срезы
    exit_layers: Tuple[int, ...] = ()              # индексы слоёв с ранним выходом
    n_experts: int = 0            # 0 = обычный MLP; >1 включает MoE
    top_k_experts: int = 2        # сколько экспертов активно на токен

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_dim


def _round_to(d: int, mult: int) -> int:
    return max(mult, int(round(d / mult)) * mult)


def _rmsnorm(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> Tensor:
    rms = ((x ** 2).mean(axis=-1, keepdims=True) + eps).sqrt()
    return x / rms * gamma


class ElasticLinear(nn.Module):
    """Линейный слой, из которого можно взять любой под-блок весов."""

    def __init__(self, in_f: int, out_f: int, bias: bool = False) -> None:
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight = Parameter(
            get_rng().standard_normal((in_f, out_f)).astype(np.float32) * math.sqrt(2.0 / in_f)
        )
        self.bias = Parameter(np.zeros(out_f, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor, in_d: Optional[int] = None, out_d: Optional[int] = None) -> Tensor:
        in_d = in_d or self.in_f
        out_d = out_d or self.out_f
        w = self.weight[:in_d, :out_d]
        out = x @ w
        if self.bias is not None:
            out = out + self.bias[:out_d]
        return out

    def active_params(self, in_d: int, out_d: int) -> int:
        return in_d * out_d + (out_d if self.bias is not None else 0)


class MatBlock(nn.Module):
    """Блок трансформера с эластичной шириной (RMSNorm + RoPE + SwiGLU)."""

    def __init__(self, dim: int, head_dim: int, mlp_ratio: int = 4, dropout: float = 0.0,
                 n_experts: int = 0, top_k: int = 2) -> None:
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.mlp_dim = mlp_ratio * dim
        self.g1 = Parameter(np.ones(dim, dtype=np.float32))
        self.g2 = Parameter(np.ones(dim, dtype=np.float32))
        self.wq = ElasticLinear(dim, dim)
        self.wk = ElasticLinear(dim, dim)
        self.wv = ElasticLinear(dim, dim)
        self.wo = ElasticLinear(dim, dim)
        self.n_experts = n_experts
        if n_experts and n_experts > 1:
            from .experts import MoE
            self.moe = MoE(dim, n_experts, top_k, mlp_ratio)
            self.w_gate = self.w_up = self.w_down = None
        else:
            self.moe = None
            self.w_gate = ElasticLinear(dim, self.mlp_dim)
            self.w_up = ElasticLinear(dim, self.mlp_dim)
            self.w_down = ElasticLinear(self.mlp_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, d_act: int, rope: nn.RotaryEmbedding,
                offset: int = 0, n_active_experts: Optional[int] = None) -> Tensor:
        b, t, _ = x.shape
        hd = self.head_dim
        n_h = d_act // hd

        h = _rmsnorm(x, self.g1[:d_act])
        q = self.wq(h, d_act, d_act).reshape(b, t, n_h, hd).transpose(0, 2, 1, 3)
        k = self.wk(h, d_act, d_act).reshape(b, t, n_h, hd).transpose(0, 2, 1, 3)
        v = self.wv(h, d_act, d_act).reshape(b, t, n_h, hd).transpose(0, 2, 1, 3)
        q, k = rope(q, offset), rope(k, offset)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hd))
        mask = np.triu(np.ones((t, t), dtype=bool), k=1)[None, None]
        att = att.masked_fill(np.broadcast_to(mask, att.shape), -1e9).softmax(axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, d_act)
        x = x + self.drop(self.wo(y, d_act, d_act))

        h2 = _rmsnorm(x, self.g2[:d_act])
        if self.moe is not None:
            return x + self.drop(self.moe(h2, d_act, n_active_experts))
        m_act = _round_to(self.mlp_dim * d_act // self.dim, 8)
        gated = self.w_gate(h2, d_act, m_act).silu() * self.w_up(h2, d_act, m_act)
        return x + self.drop(self.w_down(gated, m_act, d_act))

    def active_params(self, d_act: int, n_active_experts: Optional[int] = None) -> int:
        attn = 4 * d_act * d_act + 2 * d_act
        if self.moe is not None:
            return attn + self.moe.active_params(d_act, n_active_experts)
        m_act = _round_to(self.mlp_dim * d_act // self.dim, 8)
        return attn + 3 * d_act * m_act


class MatFormer(nn.Module):
    """Вложенный байтовый трансформер: одна модель — все размеры.

    >>> m = MatFormer(MatConfig(n_layer=2, n_embd=64))
    >>> m(np.array([[1, 2, 3]]), width=0.5).shape      # четверть весов
    (1, 3, 260)
    """

    def __init__(self, cfg: MatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.rope = nn.RotaryEmbedding(cfg.head_dim, max_seq=cfg.block_size * 8)
        self.blocks = [MatBlock(cfg.n_embd, cfg.head_dim, cfg.mlp_ratio, cfg.dropout,
                                cfg.n_experts, cfg.top_k_experts)
                       for _ in range(cfg.n_layer)]
        for i, b in enumerate(self.blocks):
            self._modules[f"block{i}"] = b
        self.g_out = Parameter(np.ones(cfg.n_embd, dtype=np.float32))
        # нормы для ранних выходов
        self.exit_norms = {i: Parameter(np.ones(cfg.n_embd, dtype=np.float32))
                           for i in cfg.exit_layers}
        for i, p in self.exit_norms.items():
            self._params[f"exit_norm{i}"] = p
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for blk in self.blocks:
            blk.wo.weight.data *= scale
        # активные steering-векторы: слой -> (вектор, сила)
        self._steering: Dict[int, Tuple[np.ndarray, float]] = {}
        self._n_active_experts: Optional[int] = None

    # ------------------------------------------------------------- размерность
    def width_dim(self, width: float) -> int:
        """Активная ширина скрытого состояния для доли width."""
        w = float(np.clip(width, 1e-3, 1.0))
        return min(self.cfg.n_embd, _round_to(int(self.cfg.n_embd * w), self.cfg.head_dim))

    def active_params(self, width: float = 1.0, n_layers: Optional[int] = None) -> int:
        """Сколько параметров реально участвует в вычислении."""
        d = self.width_dim(width)
        n_layers = self.cfg.n_layer if n_layers is None else n_layers
        total = self.cfg.vocab_size * d + d          # эмбеддинги (голова связана) + норма
        for blk in self.blocks[:n_layers]:
            total += blk.active_params(d, self._n_active_experts)
        return total

    def total_params_dense(self) -> int:
        """Все параметры, включая спящих экспертов."""
        return self.num_params()

    # ------------------------------------------------- пошаговый API (для роя)
    def embed(self, idx, d_act: int) -> Tensor:
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx).astype(int)
        if idx.ndim == 1:
            idx = idx[None]
        if idx.ndim != 2:
            raise ValueError(f"ожидались индексы (batch, seq), получено {idx.shape}")
        lo, hi = int(idx.min()), int(idx.max())
        if lo < 0 or hi >= self.cfg.vocab_size:
            raise IndexError(
                f"байтовые id должны быть в [0, {self.cfg.vocab_size - 1}], получено [{lo}, {hi}]"
            )
        return self.tok_emb.weight[:, :d_act][idx]

    def run_block(self, i: int, x: Tensor, d_act: int, offset: int = 0) -> Tensor:
        x = self.blocks[i](x, d_act, self.rope, offset, self._n_active_experts)
        sv = self._steering.get(i)
        if sv is not None:
            vec, alpha = sv
            x = x + Tensor(vec[:d_act].astype(np.float32) * alpha)
        return x

    # ------------------------------------------------------ steering-векторы
    def set_steering(self, layer: int, vector: np.ndarray, strength: float = 1.0) -> None:
        """Сдвинуть активации слоя в заданном направлении (без смены весов)."""
        self._steering[int(layer)] = (np.asarray(vector, dtype=np.float32), float(strength))

    def clear_steering(self) -> None:
        self._steering = {}

    @property
    def steering_active(self) -> bool:
        return bool(self._steering)

    def set_active_experts(self, k: Optional[int]) -> None:
        """Сколько экспертов активировать (мета-контроллер крутит эту ручку)."""
        self._n_active_experts = k

    def norm_for_moe(self, x: Tensor, layer: int, d_act: int) -> Tensor:
        """Нормированный вход MoE слоя — для анализа специализации."""
        return _rmsnorm(x, self.blocks[layer].g2[:d_act])

    def readout(self, x: Tensor, d_act: int, gamma: Optional[Tensor] = None) -> Tensor:
        g = self.g_out[:d_act] if gamma is None else gamma[:d_act]
        h = _rmsnorm(x, g)
        return h @ self.tok_emb.weight[:, :d_act].transpose()   # связанные веса

    # --------------------------------------------------------------- forward
    def forward(self, idx, width: float = 1.0, n_layers: Optional[int] = None,
                return_exits: bool = False):
        d = self.width_dim(width)
        n_layers = self.cfg.n_layer if n_layers is None else n_layers
        x = self.embed(idx, d)
        exits: List[Tuple[int, Tensor]] = []
        for i in range(n_layers):
            x = self.run_block(i, x, d)
            if return_exits and i in self.exit_norms and i < n_layers - 1:
                exits.append((i, self.readout(x, d, self.exit_norms[i])))
        logits = self.readout(x, d)
        return (logits, exits) if return_exits else logits

    # ------------------------------------------------------------ ранний выход
    @staticmethod
    def confidence(logits: Tensor) -> float:
        """Уверенность = max softmax по последней позиции."""
        z = logits.data[0, -1]
        p = np.exp(z - z.max())
        return float((p / p.sum()).max())

    @no_grad()
    def forward_early_exit(self, idx, width: float = 1.0, threshold: float = 0.9):
        """Идти по слоям, пока ответ не станет уверенным.

        Возвращает (logits, использовано_слоёв).
        """
        d = self.width_dim(width)
        x = self.embed(idx, d)
        for i in range(self.cfg.n_layer):
            x = self.run_block(i, x, d)
            if i in self.exit_norms and i < self.cfg.n_layer - 1:
                logits = self.readout(x, d, self.exit_norms[i])
                if self.confidence(logits) >= threshold:
                    return logits, i + 1
        return self.readout(x, d), self.cfg.n_layer

    # -------------------------------------------------------------- генерация
    @no_grad()
    def generate(self, prompt: Sequence[int], max_new_tokens: int = 64, width: float = 1.0,
                 temperature: float = 0.8, top_k: Optional[int] = None,
                 n_layers: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
        rng = np.random.default_rng(seed)
        was = self.training
        self.eval()
        seq = list(prompt)
        for _ in range(max_new_tokens):
            ctx = np.array([seq[-self.cfg.block_size:]], dtype=int)
            z = self(ctx, width=width, n_layers=n_layers).data[0, -1].astype(np.float64)
            if temperature <= 1e-6:
                nxt = int(z.argmax())
            else:
                z = z / temperature
                if top_k:
                    thr = np.partition(z, -top_k)[-top_k]
                    z = np.where(z < thr, -np.inf, z)
                p = np.exp(z - z.max())
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            seq.append(nxt)
        self.train(was)
        return seq
