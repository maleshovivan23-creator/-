"""Обучение TorchGPT на TinyStories. Рассчитано на Colab T4.

Под реальное API проекта:
  * модель строится из GPTConfig, а не из россыпи аргументов;
  * TorchGPT(idx, targets) сам считает loss;
  * чекпоинт совместим с NumPy-референсом (convert_back).

Запуск:
    python colab/train.py --steps 200          # проба
    python colab/train.py --steps 60000        # полный прогон
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultranet.models import GPTConfig
from ultranet.tokenizer import BPETokenizer
from ultranet.torch_port import TorchGPT


def load_bin(path: str | Path) -> np.memmap:
    return np.memmap(str(path), dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, batch: int, seq: int, device: str, rng):
    ix = rng.integers(0, len(data) - seq - 1, size=batch)
    x = np.stack([data[i:i + seq] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + seq + 1] for i in ix]).astype(np.int64)
    xt = torch.from_numpy(x).to(device, non_blocking=True)
    yt = torch.from_numpy(y).to(device, non_blocking=True)
    return xt, yt


def cosine_warmup(opt, warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def estimate_loss(model, data, batch, seq, device, rng, iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch, seq, device, rng)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(model, tok: BPETokenizer, prompt: str, max_new: int,
           device: str, block: int, temperature: float = 0.8,
           top_k: int = 40, seed: int = 0) -> str:
    g = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    ids = tok.encode(prompt) or [10]
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits = model(x[:, -block:])[:, -1, :].float()
        if temperature > 0:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)
        else:
            nxt = logits.argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    model.train()
    return tok.decode(x[0].tolist())


def train(cfg: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 0))

    tok = BPETokenizer.load(cfg["tokenizer"])
    train_data = load_bin(cfg["train_bin"])
    val_data = load_bin(cfg["val_bin"])
    print(f"данные: train {len(train_data):,} | val {len(val_data):,} токенов")

    gcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=cfg["seq"],
                     n_layer=cfg["layers"], n_head=cfg["heads"],
                     n_embd=cfg["dim"], dropout=cfg.get("dropout", 0.0),
                     rope=True, norm="rms", swiglu=True, tie_weights=True)
    model = TorchGPT(gcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"модель: {n_params / 1e6:.2f}M параметров, {device}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=(0.9, 0.95))
    sched = cosine_warmup(opt, max(100, cfg["steps"] // 20), cfg["steps"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    step0 = 0
    ckpt = out / "last.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        step0 = st["step"] + 1
        print(f"продолжаю с шага {step0}")

    log_path = out / "log.jsonl"
    model.train()
    t0 = time.time()
    tokens_done = 0

    for step in range(step0, cfg["steps"]):
        x, y = get_batch(train_data, cfg["batch"], cfg["seq"], device, rng)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        tokens_done += cfg["batch"] * cfg["seq"]

        if step % cfg["log_every"] == 0:
            tps = tokens_done / max(time.time() - t0, 1e-9)
            line = {"step": step, "loss": round(loss.item(), 4),
                    "lr": sched.get_last_lr()[0], "tok_s": round(tps)}
            print(f"шаг {step:>6} | loss {loss.item():.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {tps:,.0f} ток/с")
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")

        if step and step % cfg["eval_every"] == 0:
            vl = estimate_loss(model, val_data, cfg["batch"], cfg["seq"], device, rng)
            print(f"  val loss {vl:.4f}  (ppl {math.exp(min(vl, 20)):.1f})")

        if step and step % cfg["sample_every"] == 0:
            text = sample(model, tok, cfg["prompt"], cfg["sample_len"],
                          device, cfg["seq"])
            print(f"  --- сэмпл ---\n{text}\n  -------------")
            with open(out / "samples.txt", "a") as f:
                f.write(f"--- шаг {step} ---\n{text}\n\n")

        if step and step % cfg["ckpt_every"] == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "cfg": cfg, "gcfg": vars(gcfg)}, ckpt)
            print(f"  сохранено: {ckpt}")

    torch.save({"model": model.state_dict(), "step": cfg["steps"],
                "cfg": cfg, "gcfg": vars(gcfg)}, out / "final.pt")
    print("готово")


def default_cfg(**kw) -> dict:
    cfg = dict(
        tokenizer="data/tokenizer.json",
        train_bin="data/train.bin", val_bin="data/val.bin",
        dim=256, layers=6, heads=8, seq=256, batch=32,
        steps=60000, lr=6e-4, dropout=0.0,
        log_every=100, eval_every=1000, sample_every=1000,
        ckpt_every=500, sample_len=120,
        prompt="Once upon a time",
        out="checkpoints/tinystories-16m", seed=0)
    cfg.update(kw)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--out", default="checkpoints/tinystories-16m")
    a = ap.parse_args()
    train(default_cfg(**vars(a)))
