"""BPE-токенизатор (byte pair encoding) — как в GPT-2, но компактный.

Работает на уровне байтов, поэтому кодирует любой UTF-8 текст без <unk>.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


_CHUNK_RE = re.compile(r"\s*\S+|\s+")
_MAX_CHUNK = 64


def _split_chunks(text: str):
    """Резать по границам слов, длинные куски дробить.

    Слияния BPE почти никогда не пересекают пробел, поэтому счёт по
    словам даёт тот же результат, но кэшируется.
    """
    for m in _CHUNK_RE.finditer(text):
        chunk = m.group()
        if len(chunk) <= _MAX_CHUNK:
            yield chunk
        else:
            for i in range(0, len(chunk), _MAX_CHUNK):
                yield chunk[i:i + _MAX_CHUNK]


class BPETokenizer:
    """Обучаемый BPE: словарь = 256 байт + выученные слияния.

    >>> tok = BPETokenizer.train(text, vocab_size=512)
    >>> tok.decode(tok.encode("привет")) == "привет"
    True
    """

    #: потолок кэша, чтобы он не рос бесконечно на больших корпусах
    _CACHE_MAX = 200_000

    def __init__(self, merges: Optional[Dict[Tuple[int, int], int]] = None,
                 special: Optional[Dict[str, int]] = None) -> None:
        self.merges: Dict[Tuple[int, int], int] = merges or {}
        self.special = special or {}
        #: кэш «кусок текста -> токены»: одни и те же слова встречаются
        #: миллионы раз, пересчитывать их каждый раз незачем
        self._cache: Dict[str, List[int]] = {}
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
        # Обучение идёт по тем же кускам, что и encode. Иначе словарь
        # копит слияния через пробел, которые encode применить не может,
        # и реальное сжатие оказывается хуже измеренного при обучении.
        chunks = [list(c.encode("utf-8")) for c in _split_chunks(text)]
        merges: Dict[Tuple[int, int], int] = {}
        for new_id in range(256, vocab_size):
            counts: Counter = Counter()
            for ids in chunks:
                counts.update(zip(ids, ids[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            chunks = [cls._merge(ids, pair, new_id) if len(ids) >= 2 else ids
                      for ids in chunks]
            merges[pair] = new_id
            if verbose and new_id % 50 == 0:
                total = sum(len(c) for c in chunks)
                print(f"  слияние {new_id}: {pair} x{freq} -> длина {total}")
        return cls(merges)

    # -------------------------------------------------------------- кодирование
    def encode(self, text: str) -> List[int]:
        """Кодирование по выученным слияниям.

        Наивная реализация пересчитывала пары по ВСЕМУ тексту на каждое
        слияние: O(число_слияний × длина). На 8 КБ это 237 проходов и
        ~27k символов/с — токенизация TinyStories заняла бы 19 часов.

        Здесь текст режется на куски по границам пробелов, слияния
        применяются внутри куска, а результат кэшируется: одинаковые
        слова встречаются миллионы раз и считаются один раз.
        """
        if not self.merges:
            return list(text.encode("utf-8"))
        out: List[int] = []
        for chunk in _split_chunks(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk)
                if len(self._cache) < self._CACHE_MAX:
                    self._cache[chunk] = cached
            out.extend(cached)
        return out

    def _encode_chunk(self, chunk: str) -> List[int]:
        """Слияния внутри одного короткого куска, по возрастанию ранга."""
        ids = list(chunk.encode("utf-8"))
        if len(ids) < 2:
            return ids
        merges = self.merges
        while True:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                rank = merges.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_rank is None:
                return ids
            ids[best_pos:best_pos + 2] = [best_rank]

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
