"""Данные: загрузчик батчей, синтетические датасеты, символьный токенизатор."""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np


class DataLoader:
    """Итератор по мини-батчам с опциональным перемешиванием."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = False) -> None:
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.bs, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def __len__(self) -> int:
        n = len(self.x)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = np.random.permutation(len(self.x)) if self.shuffle else np.arange(len(self.x))
        for i in range(len(self)):
            b = idx[i * self.bs:(i + 1) * self.bs]
            yield self.x[b], self.y[b]


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return x[tr], y[tr], x[te], y[te]


def make_spirals(n_per_class: int = 300, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """Классическая нелинейная задача «спирали» — проверка мощности модели."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(n_per_class, c))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(int)


def make_moons(n: int = 600, noise: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t = np.linspace(0, np.pi, n1)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    x = np.vstack([a, b]) + rng.normal(0, noise, (2 * n1, 2))
    y = np.r_[np.zeros(n1), np.ones(n1)]
    return x.astype(np.float32), y.astype(int)


def make_digits_like(n: int = 400, size: int = 8, n_classes: int = 4, seed: int = 0):
    """Синтетические «картинки»: полоса, крест, рамка, диагональ + шум."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 1, size, size), dtype=np.float32)
    y = rng.integers(0, n_classes, n)
    for i, c in enumerate(y):
        img = np.zeros((size, size), dtype=np.float32)
        if c == 0:
            img[size // 2, :] = 1.0
        elif c == 1:
            img[size // 2, :] = img[:, size // 2] = 1.0
        elif c == 2:
            img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 1.0
        else:
            np.fill_diagonal(img, 1.0)
        x[i, 0] = img + rng.normal(0, 0.1, (size, size))
    return x, y.astype(int)


class Dataset:
    """Простейшая обёртка над массивами (x, y) с поддержкой len/индексации."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert len(x) == len(y), "x и y должны быть одной длины"
        self.x, self.y = np.asarray(x), np.asarray(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def loader(self, batch_size: int = 32, shuffle: bool = True) -> "DataLoader":
        return DataLoader(self.x, self.y, batch_size, shuffle)


def normalize(x: np.ndarray, mean=None, std=None):
    """Стандартизация признаков; возвращает (x_norm, mean, std) для переиспользования."""
    mean = x.mean(0, keepdims=True) if mean is None else mean
    std = x.std(0, keepdims=True) + 1e-8 if std is None else std
    return ((x - mean) / std).astype(np.float32), mean, std


def make_regression(n: int = 500, n_features: int = 1, noise: float = 0.1, seed: int = 0):
    """Нелинейная регрессия y = sin(3x) + шум."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (n, n_features)).astype(np.float32)
    y = (np.sin(3 * x).sum(1) + rng.normal(0, noise, n)).astype(np.float32)
    return x, y[:, None]


class WordTokenizer:
    """Пословный токенизатор со словарём частотных слов и <unk>."""

    def __init__(self, text: str, max_vocab: int = 5000) -> None:
        from collections import Counter

        words = text.split()
        freq = Counter(words).most_common(max_vocab - 2)
        self.itos = {0: "<pad>", 1: "<unk>"}
        for i, (w, _) in enumerate(freq):
            self.itos[i + 2] = w
        self.stoi = {w: i for i, w in self.itos.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(w, 1) for w in s.split()]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


class CharTokenizer:
    """Простейший символьный токенизатор для языковой модели."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def make_lm_batches(ids: List[int], block_size: int, batch_size: int, seed: Optional[int] = None):
    """Случайные окна (x, y) со сдвигом на один токен."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(ids, dtype=int)
    if len(arr) < block_size + 2:
        raise ValueError(
            f"данных слишком мало: {len(arr)} токенов при block_size={block_size}. "
            f"Нужно минимум {block_size + 2} — уменьшите block_size или возьмите больше текста."
        )
    ix = rng.integers(0, len(arr) - block_size - 1, batch_size)
    x = np.stack([arr[i:i + block_size] for i in ix])
    y = np.stack([arr[i + 1:i + 1 + block_size] for i in ix])
    return x, y
