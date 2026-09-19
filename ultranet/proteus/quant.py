"""Тернарное квантование 1.58 бит: веса в {-1, 0, +1} со скейлом (BitNet b1.58).

Реализует заявление: «Квантование до 1.58-bit. Веса в {-1, 0, 1}.
10-кратная экономия против FP16».

Здесь считается РЕАЛЬНЫЙ размер упакованных весов, а не теоретическая оценка.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..nn import Module, Parameter


def ternary_quantize(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """W -> (Q, scale), где Q ∈ {-1,0,1}. Скейл = среднее |W| (absmean, BitNet)."""
    scale = float(np.abs(w).mean()) + 1e-8
    q = np.clip(np.round(w / scale), -1, 1).astype(np.int8)
    return q, scale


def pack_ternary(q: np.ndarray) -> np.ndarray:
    """Упаковать {-1,0,1} по 2 бита на вес — 4 веса в байт."""
    flat = (q.reshape(-1) + 1).astype(np.uint8)       # -1,0,1 -> 0,1,2
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    g = flat.reshape(-1, 4)
    return (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).astype(np.uint8)


def unpack_ternary(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    b = packed.astype(np.uint8)
    vals = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], axis=1).reshape(-1)
    return (vals[:n].astype(np.int8) - 1).reshape(shape)


@dataclass
class QuantStats:
    """Отчёт о сжатии модели."""

    n_params: int
    fp32_bytes: int
    packed_bytes: int
    bits_per_weight: float
    entropy_bits: float          # реальная энтропия распределения {-1,0,1}
    zero_fraction: float

    @property
    def ratio_vs_fp32(self) -> float:
        return self.fp32_bytes / max(self.packed_bytes, 1)

    @property
    def ratio_vs_fp16(self) -> float:
        return (self.fp32_bytes / 2) / max(self.packed_bytes, 1)

    def __str__(self) -> str:
        return (
            f"параметров {self.n_params:,} | fp32 {self.fp32_bytes / 1024:.1f} КБ -> "
            f"упаковано {self.packed_bytes / 1024:.1f} КБ | {self.bits_per_weight:.2f} бит/вес "
            f"(энтропия {self.entropy_bits:.2f}) | x{self.ratio_vs_fp32:.1f} к fp32, "
            f"x{self.ratio_vs_fp16:.1f} к fp16 | нулей {self.zero_fraction * 100:.0f}%"
        )


class TernaryModel:
    """Квантованная копия модели: хранит упакованные веса и скейлы."""

    def __init__(self, model: Module, skip_1d: bool = True) -> None:
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.packed: Dict[str, np.ndarray] = {}
        self.scales: Dict[str, float] = {}
        self.kept_fp: Dict[str, np.ndarray] = {}
        n_params = fp32_bytes = packed_bytes = 0
        zeros = total_q = 0
        hist = np.zeros(3, dtype=np.int64)

        for name, p in model.named_parameters():
            n_params += p.size
            fp32_bytes += p.size * 4
            # нормы/биасы (1-D) оставляем во fp32 — их доля ничтожна, а точность важна
            if skip_1d and p.data.ndim == 1:
                self.kept_fp[name] = p.data.copy()
                packed_bytes += p.size * 4
                continue
            q, s = ternary_quantize(p.data)
            self.shapes[name] = p.data.shape
            self.packed[name] = pack_ternary(q)
            self.scales[name] = s
            packed_bytes += self.packed[name].nbytes + 4
            zeros += int((q == 0).sum())
            total_q += q.size
            hist += np.bincount((q.reshape(-1) + 1), minlength=3)

        probs = hist / max(hist.sum(), 1)
        entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
        self.stats = QuantStats(
            n_params=n_params,
            fp32_bytes=fp32_bytes,
            packed_bytes=packed_bytes,
            bits_per_weight=packed_bytes * 8 / max(n_params, 1),
            entropy_bits=entropy,
            zero_fraction=zeros / max(total_q, 1),
        )

    def dequantize_into(self, model: Module) -> Module:
        """Записать деквантованные веса обратно в модель (fake-quant инференс)."""
        params = dict(model.named_parameters())
        for name, packed in self.packed.items():
            q = unpack_ternary(packed, self.shapes[name])
            params[name].data = (q.astype(np.float32) * self.scales[name])
        for name, w in self.kept_fp.items():
            params[name].data = w.copy()
        return model


def quantization_error(model: Module) -> Dict[str, float]:
    """Относительная ошибка восстановления весов после тернаризации."""
    errs = {}
    for name, p in model.named_parameters():
        if p.data.ndim == 1:
            continue
        q, s = ternary_quantize(p.data)
        approx = q.astype(np.float32) * s
        errs[name] = float(np.linalg.norm(approx - p.data) / (np.linalg.norm(p.data) + 1e-9))
    return errs
