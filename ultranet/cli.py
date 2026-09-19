"""CLI: обучение и генерация из терминала.

    python -m ultranet train-text --file corpus.txt --steps 500
    python -m ultranet generate --checkpoint model.pkl --prompt "привет"
    python -m ultranet demo
    python -m ultranet bench
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

import ultranet as un


def _train_text(args: argparse.Namespace) -> None:
    if args.file and os.path.exists(args.file):
        text = open(args.file, encoding="utf-8").read()
    else:
        text = ("нейронная сеть учится на данных. градиент течёт назад через граф. "
                "трансформер смотрит на контекст через внимание. ") * 60
        print("файл не задан — использую встроенный корпус")

    tok = un.CharTokenizer(text)
    ids = tok.encode(text)
    print(f"символов: {len(ids):,} | словарь: {tok.vocab_size}")

    cfg = (un.GPTConfig.llama_style if args.llama else un.GPTConfig)(
        vocab_size=tok.vocab_size, block_size=args.block, n_layer=args.layers,
        n_head=args.heads, n_embd=args.embd, dropout=args.dropout,
    )
    model = un.GPT(cfg)
    print(f"архитектура: {'LLaMA-style (RoPE+RMSNorm+SwiGLU)' if args.llama else 'GPT-2 style'}")
    print(f"параметров: {model.num_params():,}")

    opt = un.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = un.CosineWarmup(opt, warmup=max(10, args.steps // 10), total=args.steps)
    ema = un.EMA(model.parameters(), decay=0.995)  # с прогревом — безопасно и на коротких ранах

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb, yb = un.make_lm_batches(ids, cfg.block_size, args.batch)
        loss = un.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step()
        lr = sched.step()
        ema.update()
        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"шаг {step:5d}/{args.steps} | loss {loss.item():.4f} "
                  f"| ppl {un.perplexity(loss.item()):8.2f} | lr {lr:.5f} "
                  f"| {time.time() - t0:.1f}s")

    ema.apply()
    if args.out:
        model.save(args.out)
        meta = {"vocab": tok.chars, "cfg": cfg.__dict__}
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"\nсохранено: {args.out} (+ .json с конфигом)")

    print("\n--- сэмпл ---")
    out = model.generate(tok.encode(args.prompt or text[:10]), args.tokens,
                         temperature=args.temp, top_k=args.top_k, top_p=args.top_p, seed=0)
    print(tok.decode(out))


def _generate(args: argparse.Namespace) -> None:
    meta_path = args.checkpoint + ".json"
    if not os.path.exists(meta_path):
        sys.exit(f"не найден {meta_path} — он создаётся командой train-text")
    meta = json.load(open(meta_path, encoding="utf-8"))
    cfg = un.GPTConfig(**meta["cfg"])
    model = un.GPT(cfg)
    model.load(args.checkpoint)

    class _Tok:
        chars = meta["vocab"]
        stoi = {c: i for i, c in enumerate(meta["vocab"])}
        itos = {i: c for i, c in enumerate(meta["vocab"])}

        def encode(self, s):
            return [self.stoi[c] for c in s if c in self.stoi]

        def decode(self, ids):
            return "".join(self.itos[int(i)] for i in ids)

    tok = _Tok()
    prompt = tok.encode(args.prompt) or [0]
    out = model.generate(prompt, args.tokens, temperature=args.temp,
                         top_k=args.top_k, top_p=args.top_p,
                         repetition_penalty=args.rep_penalty, seed=args.seed)
    print(tok.decode(out))


def _demo(args: argparse.Namespace) -> None:
    print("=" * 62)
    print("UltraNet demo — классификация спиралей".center(62))
    print("=" * 62)
    x, y = un.make_spirals(300, 3, seed=0)
    xtr, ytr, xte, yte = un.train_test_split(x, y, 0.2, seed=0)
    model = un.MLP([2, 96, 96, 3], activation="gelu")
    opt = un.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    tr = un.Trainer(model, opt, grad_clip=1.0, verbose=True)
    tr.fit(un.DataLoader(xtr, ytr, 64), un.DataLoader(xte, yte, 128, shuffle=False),
           epochs=25, patience=8)
    loss, acc = tr.evaluate(un.DataLoader(xte, yte, 128, shuffle=False))
    print(f"\nтест: loss={loss:.4f} accuracy={acc * 100:.2f}%")
    print("\nкривые обучения:")
    print(tr.plot_history())


def _bench(args: argparse.Namespace) -> None:
    from .benchmark import run_all
    run_all()


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser("ultranet", description="UltraNet — нейросети на чистом NumPy")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train-text", help="обучить символьную языковую модель")
    t.add_argument("--file", type=str, default=None, help="текстовый корпус utf-8")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--block", type=int, default=48)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--embd", type=int, default=96)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--llama", action="store_true", help="RoPE + RMSNorm + SwiGLU")
    t.add_argument("--out", type=str, default=None, help="куда сохранить веса")
    t.add_argument("--prompt", type=str, default=None)
    t.add_argument("--tokens", type=int, default=160)
    t.add_argument("--temp", type=float, default=0.8)
    t.add_argument("--top-k", dest="top_k", type=int, default=8)
    t.add_argument("--top-p", dest="top_p", type=float, default=None)
    t.set_defaults(func=_train_text)

    g = sub.add_parser("generate", help="сгенерировать текст из чекпоинта")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--prompt", type=str, default="")
    g.add_argument("--tokens", type=int, default=200)
    g.add_argument("--temp", type=float, default=0.8)
    g.add_argument("--top-k", dest="top_k", type=int, default=None)
    g.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    g.add_argument("--rep-penalty", dest="rep_penalty", type=float, default=1.1)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=_generate)

    d = sub.add_parser("demo", help="быстрая демонстрация обучения")
    d.set_defaults(func=_demo)

    b = sub.add_parser("bench", help="бенчмарк скорости операций")
    b.set_defaults(func=_bench)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
