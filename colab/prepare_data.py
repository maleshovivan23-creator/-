"""Подготовка TinyStories: обучить BPE, токенизировать, сложить в .bin.

Отличия от наброска — под реальное API проекта:
  * класс называется BPETokenizer, а не BPE;
  * у него нет bos_id/eos_id, поэтому разделитель добавляем сами;
  * BPE учится на выборке, а не на всём корпусе (иначе часы).

Запуск:
    python colab/prepare_data.py --vocab 4096 --limit 500000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ultranet.tokenizer import BPETokenizer

#: обучающий текст BPE — больше почти не улучшает словарь, но сильно дольше
BPE_SAMPLE_CHARS = 2_000_000


def build_tokenizer(texts, vocab_size: int, out: Path) -> BPETokenizer:
    path = out / "tokenizer.json"
    if path.exists():
        print(f"токенизатор уже есть: {path}")
        return BPETokenizer.load(str(path))

    sample, total = [], 0
    for t in texts:
        sample.append(t)
        total += len(t)
        if total >= BPE_SAMPLE_CHARS:
            break
    print(f"учу BPE на {total:,} символах, словарь {vocab_size}...")
    t0 = time.perf_counter()
    tok = BPETokenizer.train("\n".join(sample), vocab_size=vocab_size)
    tok.save(str(path))
    print(f"  готово за {time.perf_counter() - t0:.0f} с, словарь {tok.vocab_size}")
    return tok


def prepare(out_dir: str = "data", vocab_size: int = 4096,
            limit: int | None = None, val_frac: float = 0.005) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.bin").exists():
        print("data/train.bin уже есть — пропускаю")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("нужен datasets: pip install datasets")

    print("качаю TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"историй: {len(ds):,}")

    tok = build_tokenizer((ex["text"] for ex in ds), vocab_size, out)
    if tok.vocab_size >= 65536:
        raise SystemExit("словарь не влезает в uint16 — уменьшите --vocab")

    # разделитель историй: своего eos нет, берём перевод строки
    sep = tok.encode("\n")

    print("токенизирую...")
    t0 = time.perf_counter()
    chunks, total = [], 0
    for i, ex in enumerate(ds):
        ids = tok.encode(ex["text"])
        ids.extend(sep)
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        if i and i % 20000 == 0:
            speed = total / (time.perf_counter() - t0)
            print(f"  {i:>8,}/{len(ds):,}  {total:>12,} токенов  {speed:>10,.0f} ток/с")

    ids = np.concatenate(chunks)
    n_val = max(1024, int(len(ids) * val_frac))
    ids[:-n_val].tofile(out / "train.bin")
    ids[-n_val:].tofile(out / "val.bin")
    dt = time.perf_counter() - t0
    print(f"готово за {dt / 60:.1f} мин")
    print(f"train: {len(ids) - n_val:,} токенов | val: {n_val:,} | словарь {tok.vocab_size}")
    print(f"сжатие: {tok.compression_ratio(ds[0]['text']):.2f} символов на токен")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None,
                    help="сколько историй взять (для пробы)")
    a = ap.parse_args()
    prepare(a.out, a.vocab, a.limit)
