"""Идеальные веса Протея: семь техник вместо «просто чисел».

Каждая техника убирает свой источник неидеальности и **измеряется**:
относительная ошибка `||w - ŵ|| / ||w||` и реальная цена в битах на вес.

  1. Гиперсферическая нормализация — направление + масштаб.
  2. Лог-полярное квантование — sign × 2^e × (1 + m/4), 7 бит, умножение = сдвиг.
  3. RVQ — остаточные кодбуки, вложенная точность из одного хранилища.
  4. Fisher-важность — важным весам больше бит, остальным меньше.
  5. Фрактальная структура — слои порождаются из базового.
  6. Комплексные веса — амплитуда + фаза.
  7. Гиперсеть — веса порождаются из эмбеддинга задачи.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nn import Module, Parameter
from ..tensor import Tensor, get_rng


def rel_error(w: np.ndarray, hat: np.ndarray) -> float:
    """Относительная ошибка восстановления: главная метрика качества весов."""
    denom = float(np.linalg.norm(w))
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(w - hat) / denom)


def cosine(w: np.ndarray, hat: np.ndarray) -> float:
    a, b = w.ravel(), hat.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ═══════════════════════════════════ 1. Гиперсферическая нормализация
@dataclass
class SphericalWeight:
    """Вес = направление на единичной сфере + масштаб.

    «Все веса имеют длину 1. Их различие — только в направлении».
    Масштаб хранится отдельно в FP16 — по одному числу на строку.
    """

    directions: np.ndarray        # (rows, cols), каждая строка — единичная
    scales: np.ndarray            # (rows,) в fp16

    @classmethod
    def from_weight(cls, w: np.ndarray, per_row: bool = True) -> "SphericalWeight":
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 1:
            w = w[None, :]
        axis = 1 if per_row else None
        norms = np.linalg.norm(w, axis=axis, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        dirs = (w / norms).astype(np.float32)
        return cls(dirs, norms.squeeze(-1).astype(np.float16))

    def to_weight(self) -> np.ndarray:
        return (self.directions * self.scales.astype(np.float32)[:, None]).astype(np.float32)

    def bits_per_weight(self, direction_bits: float) -> float:
        n = self.directions.size
        return direction_bits + (16.0 * self.scales.size / max(n, 1))

    @property
    def is_unit(self) -> bool:
        return bool(np.allclose(np.linalg.norm(self.directions, axis=1), 1.0, atol=1e-5))


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Плавный переход по дуге сферы между двумя направлениями."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    theta = math.acos(dot)
    if theta < 1e-6:
        return a.copy()
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) / s) * a + (math.sin(t * theta) / s) * b


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Угол между весами = их семантическая близость."""
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


# ═══════════════════════════════════════ 2. Лог-полярное квантование
#: экспонента 4 бита (16 значений), мантисса 2 бита, знак 1 бит = 7 бит
LOGPOLAR_BITS = 7
_E_MIN, _E_MAX = -8, 7
_M_LEVELS = 4


@dataclass
class LogPolarWeight:
    """w = sign × 2^e × (1 + m/4).

    Умножение на 2^e — сдвиг. Умножение на (1 + m/4) — сдвиг и сложение.
    Ни одного настоящего умножения.
    """

    sign: np.ndarray              # int8 {-1, +1}
    exponent: np.ndarray          # int8 в [-8, 7] (относительно bias)
    mantissa: np.ndarray          # uint8 в [0, 3]
    zero_mask: np.ndarray         # bool: точные нули
    shape: Tuple[int, ...]
    bias: int = 0                 # общий сдвиг экспоненты на тензор

    BITS = LOGPOLAR_BITS

    @classmethod
    def quantize(cls, w: np.ndarray) -> "LogPolarWeight":
        w = np.asarray(w, dtype=np.float32)
        a = np.abs(w)
        zero = a < 1e-20
        sign = np.where(w < 0, -1, 1).astype(np.int8)

        safe = np.where(zero, 1.0, a)
        # Окно экспоненты всего 16 значений. Без привязки к масштабу тензора
        # веса мельче 2^-8 обнулялись целиком — сдвигаем окно под данные.
        nz = a[~zero]
        bias = 0
        if nz.size:
            top = int(np.floor(math.log2(float(nz.max()))))
            bias = int(top - _E_MAX)
        safe = safe / (2.0 ** bias)
        e = np.floor(np.log2(safe)).astype(np.int32)
        frac = safe / np.exp2(e.astype(np.float32)) - 1.0      # в [0, 1)
        m = np.rint(frac * _M_LEVELS).astype(np.int32)
        # мантисса переполнилась — переносим в экспоненту
        carry = m >= _M_LEVELS
        e = np.where(carry, e + 1, e)
        m = np.where(carry, 0, m)
        # выход за диапазон экспоненты — насыщаем, слишком малое считаем нулём
        zero = zero | (e < _E_MIN)
        e = np.clip(e, _E_MIN, _E_MAX)
        return cls(sign, e.astype(np.int8), m.astype(np.uint8), zero, w.shape, bias)

    def dequantize(self) -> np.ndarray:
        val = (np.exp2(self.exponent.astype(np.float32) + self.bias)
               * (1.0 + self.mantissa.astype(np.float32) / _M_LEVELS)
               * self.sign.astype(np.float32))
        return np.where(self.zero_mask, 0.0, val).astype(np.float32)

    @property
    def n_weights(self) -> int:
        return int(np.prod(self.shape))

    def nbytes(self) -> int:
        return int(math.ceil(self.n_weights * self.BITS / 8))

    def shift_multiply(self, x: np.ndarray) -> np.ndarray:
        """Умножение через сдвиги: x * w = ((x << e) + (x << e) * m/4) * sign.

        Демонстрирует, что настоящего умножения не требуется.
        """
        base = np.ldexp(x, self.exponent.astype(np.int32) + self.bias)   # сдвиг
        acc = base.copy()
        m = self.mantissa.astype(np.int32)
        for bit, frac in ((1, 0.25), (2, 0.5)):                 # m/4 = биты m
            acc = acc + np.where((m & bit) != 0, base * frac, 0.0)
        out = acc * self.sign.astype(np.float32)
        return np.where(self.zero_mask, 0.0, out).astype(np.float32)


# ═══════════════════════════ 3. Остаточное векторное квантование (RVQ)
class RVQ:
    """w ≈ c₁ + c₂ + c₃ + … — каждый кодбук уточняет остаток предыдущего.

    Главное свойство — вложенность: можно взять только c₁ и получить
    грубые, но рабочие веса; добавить c₂ — точнее. Одно хранилище, много
    точностей, без отдельных версий модели.
    """

    def __init__(self, dim: int = 4, size: int = 256, stages: int = 3,
                 iters: int = 12, seed: int = 0) -> None:
        self.dim, self.size, self.stages, self.iters = dim, size, stages, iters
        self.rng = np.random.default_rng(seed)
        self.codebooks: List[np.ndarray] = []
        self.shape: Optional[Tuple[int, ...]] = None
        self.pad = 0

    # ----------------------------------------------------------- обучение
    def _to_vectors(self, w: np.ndarray) -> np.ndarray:
        flat = np.asarray(w, dtype=np.float32).ravel()
        self.pad = (-flat.size) % self.dim
        if self.pad:
            flat = np.concatenate([flat, np.zeros(self.pad, dtype=np.float32)])
        return flat.reshape(-1, self.dim)

    def _kmeans(self, x: np.ndarray) -> np.ndarray:
        k = min(self.size, len(x))
        idx = self.rng.choice(len(x), size=k, replace=False)
        cb = x[idx].copy()
        for _ in range(self.iters):
            d = ((x[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            for j in range(k):
                sel = x[assign == j]
                if len(sel):
                    cb[j] = sel.mean(0)
        if k < self.size:                       # добить кодбук до размера
            cb = np.concatenate([cb, np.zeros((self.size - k, self.dim), np.float32)])
        return cb.astype(np.float32)

    def fit(self, w: np.ndarray, sample: int = 4096) -> "RVQ":
        self.shape = np.asarray(w).shape
        vecs = self._to_vectors(w)
        residual = vecs.copy()
        self.codebooks = []
        for _ in range(self.stages):
            sub = residual
            if len(sub) > sample:
                sub = residual[self.rng.choice(len(residual), sample, replace=False)]
            cb = self._kmeans(sub)
            self.codebooks.append(cb)
            idx = self._assign(residual, cb)
            residual = residual - cb[idx]
        return self

    @staticmethod
    def _assign(x: np.ndarray, cb: np.ndarray, chunk: int = 8192) -> np.ndarray:
        out = np.empty(len(x), dtype=np.int32)
        for s in range(0, len(x), chunk):
            part = x[s:s + chunk]
            d = ((part[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
            out[s:s + chunk] = d.argmin(1)
        return out

    # -------------------------------------------------------------- коды
    def encode(self, w: np.ndarray) -> List[np.ndarray]:
        vecs = self._to_vectors(w)
        residual = vecs.copy()
        codes = []
        for cb in self.codebooks:
            idx = self._assign(residual, cb)
            codes.append(idx)
            residual = residual - cb[idx]
        return codes

    def decode(self, codes: Sequence[np.ndarray], stages: Optional[int] = None
               ) -> np.ndarray:
        n = len(codes) if stages is None else min(stages, len(codes))
        total = np.zeros((len(codes[0]), self.dim), dtype=np.float32)
        for i in range(n):
            total += self.codebooks[i][codes[i]]
        flat = total.ravel()
        if self.pad:
            flat = flat[:-self.pad]
        return flat.reshape(self.shape).astype(np.float32)

    # -------------------------------------------------------------- цена
    def bits_per_weight(self, stages: Optional[int] = None) -> float:
        n = self.stages if stages is None else stages
        return n * math.log2(self.size) / self.dim

    def codebook_bytes(self) -> int:
        return sum(cb.size * 2 for cb in self.codebooks)   # fp16

    def error_by_stage(self, w: np.ndarray) -> List[Tuple[int, float, float]]:
        """(число кодбуков, бит/вес, относительная ошибка) — вложенность в цифрах."""
        codes = self.encode(w)
        out = []
        for s in range(1, len(codes) + 1):
            out.append((s, self.bits_per_weight(s), rel_error(w, self.decode(codes, s))))
        return out


# ═══════════════════════════════════ 4. Важность через Fisher Information
@dataclass
class FisherPlan:
    """Сколько бит дать каждой группе весов."""

    bits: np.ndarray              # бит на элемент
    thresholds: Tuple[float, ...]
    mean_bits: float


class FisherImportance:
    """F_i = E[(∂ log p / ∂ w_i)²] — мера важности веса для задачи.

    Важные веса защищены от квантования, неважные дёшевы.
    """

    def __init__(self, levels: Sequence[int] = (2, 4, 8, 16)) -> None:
        self.levels = tuple(sorted(levels))
        self.fisher: Dict[str, np.ndarray] = {}

    def accumulate(self, model: Module) -> None:
        """Накопить квадраты градиентов после backward()."""
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            self.fisher[name] = self.fisher.get(name, 0.0) + p.grad ** 2

    def plan_for(self, name: str, quantiles: Sequence[float] = (0.5, 0.8, 0.95)
                 ) -> Optional[FisherPlan]:
        """Раздать биты по квантилям важности."""
        f = self.fisher.get(name)
        if f is None:
            return None
        qs = np.quantile(f, quantiles)
        bits = np.full(f.shape, self.levels[0], dtype=np.int32)
        for lvl, thr in zip(self.levels[1:], qs):
            bits = np.where(f > thr, lvl, bits)
        return FisherPlan(bits, tuple(float(q) for q in qs), float(bits.mean()))

    @staticmethod
    def quantize_adaptive(w: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """Квантование с индивидуальной разрядностью на элемент.

        Масштаб считается ОТДЕЛЬНО для каждой битовой группы: общий масштаб
        по всей матрице уничтожал 2-битную группу (шаг оказывался больше
        самих весов) и адаптивность проигрывала равномерной сетке.
        """
        w = np.asarray(w, dtype=np.float32)
        out = np.empty_like(w)
        for b in np.unique(bits):
            mask = bits == b
            part = w[mask]
            if not part.size:
                continue
            scale = float(np.abs(part).max()) or 1.0
            levels = max(2 ** int(b) - 1, 1)
            step = 2.0 * scale / levels
            out[mask] = np.clip(np.rint(part / step) * step, -scale, scale)
        return out

    def uniform_baseline(self, w: np.ndarray, bits: float) -> np.ndarray:
        """Равномерная сетка той же средней стоимости — честный конкурент."""
        return self.quantize_adaptive(w, np.full(w.shape, int(round(bits))))

    @staticmethod
    def weighted_error(w: np.ndarray, hat: np.ndarray, fisher: np.ndarray) -> float:
        """Fisher-взвешенная ошибка ≈ реальный прирост loss (2-й порядок).

        Именно она определяет «качество», а не голая L2: ошибка в неважном
        весе почти бесплатна, ошибка в важном — дорога.
        """
        num = float(np.sum(fisher * (w - hat) ** 2))
        den = float(np.sum(fisher * w ** 2))
        return math.sqrt(num / den) if den > 0 else 0.0


# ═══════════════════════════════════════════ 5. Фрактальная структура
class FractalStack:
    """W_n = f(W_{n-1}): слои порождаются из базового рекурсивно.

    Храним базовый слой и маленькое преобразование — вместо N слоёв.
    Можно дорастить новые слои, применив f ещё раз.
    """

    def __init__(self, base: np.ndarray, n_layers: int, seed: int = 0) -> None:
        self.base = np.asarray(base, dtype=np.float32)
        self.n_layers = n_layers
        r, c = self.base.shape
        rng = np.random.default_rng(seed)
        # f(W) = A @ W @ B + g — компактное обучаемое преобразование
        self.A = np.eye(r, dtype=np.float32) + rng.normal(0, 0.01, (r, r)).astype(np.float32)
        self.B = np.eye(c, dtype=np.float32) + rng.normal(0, 0.01, (c, c)).astype(np.float32)
        self.gain = np.ones(n_layers, dtype=np.float32)

    def generate(self, n: Optional[int] = None) -> List[np.ndarray]:
        n = self.n_layers if n is None else n
        out, w = [], self.base.copy()
        for i in range(n):
            g = self.gain[i] if i < len(self.gain) else 1.0
            out.append((w * g).astype(np.float32))
            w = self.A @ w @ self.B
        return out

    def fit(self, targets: Sequence[np.ndarray], steps: int = 300,
            lr: float = 0.02) -> List[float]:
        """Подогнать A, B, gain под реальный стек слоёв (градиентный спуск)."""
        hist: List[float] = []
        self.gain = np.ones(len(targets), dtype=np.float32)
        for _ in range(steps):
            gen = self.generate(len(targets))
            loss = float(np.mean([np.mean((g - t) ** 2) for g, t in zip(gen, targets)]))
            hist.append(loss)
            # численный градиент по gain (дёшево и устойчиво)
            for i, (g, t) in enumerate(zip(gen, targets)):
                raw = g / max(self.gain[i], 1e-8)
                denom = float(np.sum(raw * raw)) + 1e-8
                self.gain[i] = float(np.sum(raw * t) / denom)
            # шаг по A и B в сторону уменьшения ошибки
            gen = self.generate(len(targets))
            dA = np.zeros_like(self.A)
            dB = np.zeros_like(self.B)
            w = self.base.copy()
            for i in range(1, len(targets)):
                prev = w
                w = self.A @ w @ self.B
                err = (w * self.gain[i] - targets[i]) * self.gain[i]
                dA += err @ (prev @ self.B).T
                dB += (self.A @ prev).T @ err
            n = max(len(targets) - 1, 1)
            self.A -= lr * dA / n
            self.B -= lr * dB / n
        return hist

    # -------------------------------------------------------------- цена
    def stored_params(self) -> int:
        return self.base.size + self.A.size + self.B.size + self.gain.size

    def generated_params(self) -> int:
        return self.base.size * self.n_layers

    def compression(self) -> float:
        return self.generated_params() / max(self.stored_params(), 1)

    def error_vs(self, targets: Sequence[np.ndarray]) -> float:
        gen = self.generate(len(targets))
        return float(np.mean([rel_error(t, g) for t, g in zip(gen, targets)]))


# ═══════════════════════════════════════ 6. Комплексные веса с фазой
@dataclass
class ComplexWeight:
    """w = r·e^{iθ}: амплитуда (важность) + фаза (отношение).

    Фазу квантуем грубо (3 бита = 8 направлений), амплитуду точнее (4 бита).
    Итого 7 бит на комплексное число = 3.5 бита на действительный вес.
    """

    amp_code: np.ndarray          # uint8, amp_bits
    phase_code: np.ndarray        # uint8, phase_bits
    scale: float
    shape: Tuple[int, ...]
    amp_bits: int = 4
    phase_bits: int = 3

    @classmethod
    def from_pairs(cls, w: np.ndarray, amp_bits: int = 4, phase_bits: int = 3
                   ) -> "ComplexWeight":
        """Пары соседних весов -> комплексные числа."""
        flat = np.asarray(w, dtype=np.float32).ravel()
        if flat.size % 2:
            flat = np.concatenate([flat, np.zeros(1, np.float32)])
        z = flat[0::2] + 1j * flat[1::2]
        amp = np.abs(z)
        phase = np.angle(z) % (2 * np.pi)
        scale = float(amp.max()) or 1.0
        a_levels = 2 ** amp_bits - 1
        p_levels = 2 ** phase_bits
        amp_code = np.rint(amp / scale * a_levels).astype(np.uint8)
        phase_code = (np.rint(phase / (2 * np.pi) * p_levels).astype(np.int32)
                      % p_levels).astype(np.uint8)
        return cls(amp_code, phase_code, scale, w.shape, amp_bits, phase_bits)

    def to_weight(self) -> np.ndarray:
        a_levels = 2 ** self.amp_bits - 1
        p_levels = 2 ** self.phase_bits
        amp = self.amp_code.astype(np.float32) / a_levels * self.scale
        phase = self.phase_code.astype(np.float32) / p_levels * 2 * np.pi
        z = amp * np.exp(1j * phase)
        flat = np.empty(z.size * 2, dtype=np.float32)
        flat[0::2] = z.real
        flat[1::2] = z.imag
        n = int(np.prod(self.shape))
        return flat[:n].reshape(self.shape)

    def bits_per_weight(self) -> float:
        return (self.amp_bits + self.phase_bits) / 2.0


def complex_matmul(xr: np.ndarray, xi: np.ndarray, wr: np.ndarray, wi: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Комплексное умножение = поворот + масштаб (богаче простого масштаба)."""
    return xr @ wr - xi @ wi, xr @ wi + xi @ wr


# ═══════════════════════════════════════ 7. Гиперсеть для генерации весов
class HyperNetwork(Module):
    """Маленькая сеть порождает веса большой: W = H(task_embedding).

    Ключ к реальной экономии — генерировать веса ПОСТРОЧНО из координат.
    Наивная схема (hidden × rows × rank) сама весит больше порождаемой
    матрицы; здесь размер гиперсети не зависит от размеров выхода вообще,
    поэтому сжатие растёт вместе с моделью.

    Одна гиперсеть — бесконечно много моделей: поменял эмбеддинг —
    поменял модель, без переобучения. Порождает low-rank факторы,
    поэтому совместима с LoRA-персонализацией.
    """

    POS_DIM = 8       # синусоидальные координаты строк/столбцов: 0 параметров

    def __init__(self, embed_dim: int, out_rows: int, out_cols: int,
                 rank: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.embed_dim, self.rows, self.cols, self.rank = embed_dim, out_rows, out_cols, rank
        rng = get_rng()
        d_in = embed_dim + self.POS_DIM
        self.w1 = Parameter(rng.standard_normal((d_in, hidden)).astype(np.float32)
                            / math.sqrt(d_in))
        self.b1 = Parameter(np.zeros(hidden, dtype=np.float32))
        self.w2 = Parameter(rng.standard_normal((hidden, rank)).astype(np.float32)
                            / math.sqrt(hidden))
        self.b2 = Parameter(np.zeros(rank, dtype=np.float32))
        self.scale = Parameter(np.array([0.1], dtype=np.float32))
        self.tasks: Dict[str, np.ndarray] = {}
        self._pos_cache: Dict[int, np.ndarray] = {}

    # --------------------------------------------------------- координаты
    def _positions(self, n: int) -> np.ndarray:
        """Синусоидальные координаты — не хранятся, а вычисляются."""
        if n in self._pos_cache:
            return self._pos_cache[n]
        idx = np.arange(n, dtype=np.float32)[:, None]
        freqs = np.exp2(np.arange(self.POS_DIM // 2, dtype=np.float32))[None, :]
        ang = idx / max(n, 1) * freqs * np.pi
        pos = np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)
        self._pos_cache[n] = pos
        return pos

    # ------------------------------------------------------------ задачи
    def register_task(self, name: str, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else get_rng()
        e = rng.standard_normal(self.embed_dim).astype(np.float32)
        e /= max(np.linalg.norm(e), 1e-9)
        self.tasks[name] = e
        return e

    def embedding(self, task: str) -> np.ndarray:
        if task not in self.tasks:
            self.register_task(task, seed=abs(hash(task)) % (2 ** 31))
        return self.tasks[task]

    # ----------------------------------------------------------- forward
    def _factor(self, e: np.ndarray, n: int) -> np.ndarray:
        """Сгенерировать n строк ранга rank из эмбеддинга задачи."""
        pos = self._positions(n)
        x = np.concatenate([np.repeat(e[None, :], n, axis=0), pos], axis=1)
        h = np.tanh(x @ self.w1.data + self.b1.data)
        return h @ self.w2.data + self.b2.data

    def generate(self, task_or_vec) -> np.ndarray:
        """Сгенерировать матрицу весов под задачу."""
        e = (self.embedding(task_or_vec) if isinstance(task_or_vec, str)
             else np.asarray(task_or_vec, dtype=np.float32))
        a = self._factor(e, self.rows)                      # (rows, rank)
        b = self._factor(-e, self.cols).T                   # (rank, cols)
        return (a @ b * float(self.scale.data[0])).astype(np.float32)

    def forward(self, task_or_vec) -> Tensor:
        return Tensor(self.generate(task_or_vec))

    # -------------------------------------------------------------- цена
    def n_params(self) -> int:
        return sum(p.size for p in self.parameters())

    def generated_params(self) -> int:
        return self.rows * self.cols

    def compression(self) -> float:
        return self.generated_params() / max(self.n_params(), 1)

    def interpolate(self, task_a: str, task_b: str, t: float) -> np.ndarray:
        """Между задачами можно плавно переходить — модель тоже плавная."""
        return self.generate(slerp(self.embedding(task_a), self.embedding(task_b), t))


# ═══════════════════════════════════════════════ сводный пайплайн
@dataclass
class WeightReport:
    """Результат сжатия одной матрицы весов."""

    method: str
    bits_per_weight: float
    rel_error: float
    cosine: float
    bytes_used: int

    @property
    def vs_fp16(self) -> float:
        return 16.0 / max(self.bits_per_weight, 1e-9)

    def __str__(self) -> str:
        return (f"{self.method:<28} {self.bits_per_weight:>5.2f} бит/вес  "
                f"ошибка {self.rel_error * 100:>6.2f}%  cos {self.cosine:.4f}  "
                f"x{self.vs_fp16:.1f} к FP16")


def compare_methods(w: np.ndarray, rvq_stages: int = 3,
                    rvq_dim: int = 4) -> List[WeightReport]:
    """Честное сравнение всех техник на одной матрице."""
    w = np.asarray(w, dtype=np.float32)
    n = w.size
    out: List[WeightReport] = []

    # базовая линия: ternary
    scale = float(np.abs(w).mean()) or 1.0
    tern = np.clip(np.rint(w / scale), -1, 1) * scale
    out.append(WeightReport("ternary (1.58-bit)", 2.0, rel_error(w, tern),
                            cosine(w, tern), int(n * 2 / 8)))

    # fp16
    h = w.astype(np.float16).astype(np.float32)
    out.append(WeightReport("FP16", 16.0, rel_error(w, h), cosine(w, h), n * 2))

    # лог-полярное
    lp = LogPolarWeight.quantize(w)
    d = lp.dequantize()
    out.append(WeightReport("лог-полярное (7 бит)", 7.0, rel_error(w, d),
                            cosine(w, d), lp.nbytes()))

    # гиперсферическое + лог-полярное
    sph = SphericalWeight.from_weight(w)
    lp2 = LogPolarWeight.quantize(sph.directions)
    rec = (lp2.dequantize() * sph.scales.astype(np.float32)[:, None])
    out.append(WeightReport("сфера + лог-полярное", sph.bits_per_weight(7.0),
                            rel_error(w, rec), cosine(w, rec), lp2.nbytes()))

    # RVQ по стадиям
    rvq = RVQ(dim=rvq_dim, stages=rvq_stages).fit(w)
    codes = rvq.encode(w)
    for s in range(1, rvq_stages + 1):
        rec = rvq.decode(codes, s)
        out.append(WeightReport(f"RVQ, {s} кодбук(а)", rvq.bits_per_weight(s),
                                rel_error(w, rec), cosine(w, rec),
                                int(n * rvq.bits_per_weight(s) / 8) + rvq.codebook_bytes()))

    # комплексные
    cw = ComplexWeight.from_pairs(w)
    rec = cw.to_weight()
    out.append(WeightReport("комплексные (амп+фаза)", cw.bits_per_weight(),
                            rel_error(w, rec), cosine(w, rec),
                            int(n * cw.bits_per_weight() / 8)))
    return out
