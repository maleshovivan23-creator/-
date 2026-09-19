"""BPE-токенизатор (byte pair encoding) — как в GPT-2, но компактный.

Работает на уровне байтов, поэтому кодирует любой UTF-8 текст без <unk>.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    """Обучаемый BPE: словарь = 256 байт + выученные слияния.

    >>> tok = BPETokenizer.train(text, vocab_size=512)
    >>> tok.decode(tok.encode("привет")) == "привет"
    True
    """

    def __init__(self, merges: Optional[Dict[Tuple[int, int], int]] = None,
                 special: Optional[Dict[str, int]] = None) -> None:
        self.merges: Dict[Tuple[int, int], int] = merges or {}
        self.special = special or {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        for tokstr, idx in self.special.items():
            self.vocab[idx] = tokstr.encode("utf-8")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ обучение
    @staticmethod
    def _pair_counts(ids: List[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i, n = [], 0, len(ids)
        a, b = pair
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, verbose: bool = False) -> "BPETokenizer":
        """Выучить слияния так, чтобы итоговый словарь был не больше vocab_size."""
        assert vocab_size >= 256, "словарь не может быть меньше 256 байт"
        ids = list(text.encode("utf-8"))
        merges: Dict[Tuple[int, int], int] = {}
        for new_id in range(256, vocab_size):
            counts = cls._pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            ids = cls._merge(ids, pair, new_id)
            merges[pair] = new_id
            if verbose and new_id % 50 == 0:
                print(f"  слияние {new_id}: {pair} x{freq} -> длина {len(ids)}")
        return cls(merges)

    # -------------------------------------------------------------- кодирование
    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        if not self.merges:
            return ids
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            # применяем самое раннее выученное слияние
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        """Во сколько раз короче последовательность по сравнению с байтами."""
        return len(text.encode("utf-8")) / max(len(self.encode(text)), 1)

    # ------------------------------------------------------------- сериализация
    def save(self, path: str) -> None:
        payload = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "special": self.special,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        merges = {tuple(k): v for k, v in payload["merges"]}
        return cls(merges, payload.get("special", {}))
