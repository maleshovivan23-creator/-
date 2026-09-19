"""RAG: знание живёт вне весов.

Реализует заявление: «RAG вместо параметров. Крошечная модель знает больше,
чем большая без памяти». Здесь это проверяемо: индекс хранит факты локально,
поиск идёт по символьным n-граммам (не требует эмбеддинг-модели),
всё остаётся на устройстве.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _ngrams(text: str, n: int = 3) -> List[str]:
    """Символьные n-граммы нескольких длин.

    Индексируются длины 2..n: короткие запросы (например, CJK «東京» —
    всего 2 символа) иначе не находились бы в индексе из 3-грамм.
    """
    t = re.sub(r"\s+", " ", text.lower().strip())
    if not t:
        return []
    out: List[str] = []
    for size in range(2, n + 1):
        if len(t) < size:
            continue
        out.extend(t[i:i + size] for i in range(len(t) - size + 1))
    # Целые слова как отдельные термы: иначе запрос «137» матчится
    # на «1372» через общие n-граммы. Слово даёт точное совпадение
    # с высоким весом и разводит такие случаи.
    out.extend(f"\u0000w:{w}" for w in re.findall(r"\w+", t))
    return out or [t]


@dataclass
class Doc:
    id: int
    text: str
    meta: Dict = field(default_factory=dict)


class LocalMemory:
    """Локальный RAG-индекс: BM25 по символьным n-граммам.

    Не покидает устройство, не требует сети и эмбеддинг-модели.

    >>> mem = LocalMemory()
    >>> mem.add("Столица Франции — Париж.")
    >>> mem.search("париж")[0].text
    'Столица Франции — Париж.'
    """

    def __init__(self, n: int = 3, k1: float = 1.5, b: float = 0.75) -> None:
        self.n, self.k1, self.b = n, k1, b
        self.docs: List[Doc] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len: List[int] = []

    def __len__(self) -> int:
        return len(self.docs)

    def add(self, text: str, **meta) -> int:
        doc_id = len(self.docs)
        self.docs.append(Doc(doc_id, text, meta))
        counts = Counter(_ngrams(text, self.n))
        for gram, c in counts.items():
            self.postings[gram].append((doc_id, c))
        self.doc_len.append(sum(counts.values()) or 1)
        return doc_id

    def add_many(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add(t)

    def search(self, query: str, top_k: int = 3) -> List[Doc]:
        return [d for d, _ in self.search_scored(query, top_k)]

    def search_scored(self, query: str, top_k: int = 3) -> List[Tuple[Doc, float]]:
        if not self.docs:
            return []
        avgdl = sum(self.doc_len) / len(self.doc_len)
        N = len(self.docs)
        scores: Dict[int, float] = defaultdict(float)
        for gram in set(_ngrams(query, self.n)):
            posting = self.postings.get(gram)
            if not posting:
                continue
            idf = math.log(1 + (N - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc_id, tf in posting:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.docs[i], s) for i, s in ranked]

    def context_for(self, query: str, top_k: int = 3, max_chars: int = 400) -> str:
        """Собрать контекст для подстановки в промпт."""
        parts, total = [], 0
        for doc in self.search(query, top_k):
            if total + len(doc.text) > max_chars:
                break
            parts.append(doc.text)
            total += len(doc.text)
        return "\n".join(parts)

    # ------------------------------------------------------------ размер/IO
    def size_bytes(self) -> int:
        """Сколько памяти занимает знание — сравнимо с весами модели."""
        return sum(len(d.text.encode("utf-8")) for d in self.docs)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"text": d.text, "meta": d.meta} for d in self.docs],
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, n: int = 3) -> "LocalMemory":
        mem = cls(n=n)
        with open(path, encoding="utf-8") as f:
            for row in json.load(f):
                mem.add(row["text"], **row.get("meta", {}))
        return mem
