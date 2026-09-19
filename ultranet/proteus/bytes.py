"""Байтовый «токенизатор» — словаря нет вовсе.

Реализует заявление Протея: «Токенизатор отсутствует. Каждый байт — вход».
Любой UTF-8 текст кодируется без <unk>, все языки равноправны по построению.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

PAD, BOS, EOS, MASK = 256, 257, 258, 259


class ByteTokenizer:
    """Кодирование на уровне байтов: фиксированный «словарь» 256 + 4 спецтокена.

    Словаря в обычном смысле нет — есть биекция байт <-> id, одинаковая
    для всех языков, эмодзи, кода, нот и произвольных бинарных данных.

    >>> tok = ByteTokenizer()
    >>> tok.decode(tok.encode("привет 日本語 🚀")) == "привет 日本語 🚀"
    True
    """

    vocab_size = 260
    pad_id, bos_id, eos_id, mask_id = PAD, BOS, EOS, MASK

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if bos:
            ids = [BOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return raw.decode("utf-8", errors="replace")

    def encode_bytes(self, data: bytes) -> List[int]:
        """Произвольные бинарные данные — тоже последовательность."""
        return list(data)

    # ------------------------------------------------------------------ анализ
    @staticmethod
    def bytes_per_char(text: str) -> float:
        """Сколько байтов приходится на символ — цена языка при байтовом входе."""
        return len(text.encode("utf-8")) / max(len(text), 1)

    @staticmethod
    def fairness_report(samples: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Сравнить стоимость одного и того же смысла на разных языках.

        Возвращает по языку: символы, байты, байт/символ.
        """
        out: Dict[str, Dict[str, float]] = {}
        for lang, text in samples.items():
            n_chars = len(text)
            n_bytes = len(text.encode("utf-8"))
            out[lang] = {
                "chars": float(n_chars),
                "bytes": float(n_bytes),
                "bytes_per_char": n_bytes / max(n_chars, 1),
            }
        return out


def make_byte_batches(ids: Sequence[int], block_size: int, batch_size: int,
                      seed: int | None = None):
    """Случайные окна (x, y) из байтового потока."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(ids, dtype=int)
    if len(arr) < block_size + 2:
        raise ValueError(
            f"данных слишком мало: {len(arr)} байт при block_size={block_size}"
        )
    ix = rng.integers(0, len(arr) - block_size - 1, batch_size)
    x = np.stack([arr[i:i + block_size] for i in ix])
    y = np.stack([arr[i + 1:i + 1 + block_size] for i in ix])
    return x, y
