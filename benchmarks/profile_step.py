"""Профиль одного шага обучения: forward, backward, optimizer — по отдельности.

Нельзя ускорять то, что не измерено. Скрипт отвечает на один вопрос:
куда уходит время на шаге обучения GPT.

Запуск:
    PYTHONPATH=. .venv/bin/python benchmarks/profile_step.py
    PYTHONPATH=. .venv/bin/python benchmarks/profile_step.py --profile
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import resource
import time
from pathlib import Path

import numpy as np

import ultranet as un
from ultranet.functional import cross_entropy
from ultranet.models import GPT, GPTConfig
from ultranet.optim import Adam
from ultranet.tensor import Tensor

BATCH, SEQ = 8, 64
CFG = GPTConfig(vocab_size=256, n_layer=4, n_head=4, n_embd=128, block_size=64)


def build():
    un.manual_seed(0)
    model = GPT(CFG)
    opt = Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (BATCH, SEQ))
    y = rng.integers(0, 256, (BATCH, SEQ))
    return model, opt, x, y


def timed(fn, n: int, warmup: int = 3) -> float:
    """Медиана по n прогонам, в миллисекундах."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples))


def measure(n: int = 20) -> dict:
    model, opt, x, y = build()

    # ---- forward отдельно
    def fwd():
        return model(Tensor(x))

    forward_ms = timed(fwd, n)

    # ---- forward + loss
    def fwd_loss():
        return cross_entropy(model(Tensor(x)), y)

    forward_loss_ms = timed(fwd_loss, n)

    # ---- backward отдельно: граф строим вне замера
    def bwd():
        loss = cross_entropy(model(Tensor(x)), y)
        t0 = time.perf_counter()
        opt.zero_grad()
        loss.backward()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(3):
        bwd()
    backward_ms = float(np.median([bwd() for _ in range(n)]))

    # ---- optimizer отдельно: градиенты уже готовы
    loss = cross_entropy(model(Tensor(x)), y)
    opt.zero_grad()
    loss.backward()
    opt_ms = timed(opt.step, n)

    # ---- полный шаг
    def full():
        opt.zero_grad()
        cross_entropy(model(Tensor(x)), y).backward()
        opt.step()

    step_ms = timed(full, n)

    n_params = sum(p.data.size for p in model.parameters())
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return {
        "model": "gpt-2.7m",
        "n_params": int(n_params),
        "batch": BATCH,
        "seq": SEQ,
        "n_layer": CFG.n_layer,
        "n_head": CFG.n_head,
        "n_embd": CFG.n_embd,
        "dtype": "float32",
        "forward_ms": round(forward_ms, 2),
        "forward_loss_ms": round(forward_loss_ms, 2),
        "backward_ms": round(backward_ms, 2),
        "optimizer_ms": round(opt_ms, 2),
        "step_ms": round(step_ms, 2),
        "steps_per_sec": round(1000.0 / step_ms, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "tokens_per_sec": round(BATCH * SEQ * 1000.0 / step_ms, 1),
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "blas": _blas_name(),
    }


def _blas_name() -> str:
    try:
        cfg = np.__config__.show(mode="dicts")
        build = cfg.get("Build Dependencies", {})
        return str(build.get("blas", {}).get("name", "?"))
    except Exception:
        return "?"


def share(res: dict) -> None:
    step = res["step_ms"]
    fwd = res["forward_loss_ms"]
    bwd = res["backward_ms"]
    opt = res["optimizer_ms"]
    other = step - fwd - bwd - opt
    print("\n  доля шага:")
    for name, ms in (("forward+loss", fwd), ("backward", bwd),
                     ("optimizer", opt), ("прочее", other)):
        bar = "█" * max(0, int(round(40 * ms / step)))
        print(f"    {name:<14} {ms:7.2f} мс  {100 * ms / step:5.1f}%  {bar}")


def run_cprofile(n: int = 10) -> str:
    model, opt, x, y = build()

    def step():
        opt.zero_grad()
        cross_entropy(model(Tensor(x)), y).backward()
        opt.step()

    for _ in range(3):
        step()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        step()
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
    return s.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true", help="показать cProfile")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--save", default="benchmarks/baseline.json")
    args = ap.parse_args()

    res = measure(args.repeats)

    print(f"  GPT {res['n_params']:,} параметров, batch={res['batch']} seq={res['seq']}, "
          f"{res['dtype']}, BLAS={res['blas']}")
    print(f"  forward        {res['forward_ms']:8.2f} мс")
    print(f"  forward+loss   {res['forward_loss_ms']:8.2f} мс")
    print(f"  backward       {res['backward_ms']:8.2f} мс")
    print(f"  optimizer      {res['optimizer_ms']:8.2f} мс")
    print(f"  шаг целиком    {res['step_ms']:8.2f} мс   "
          f"({res['steps_per_sec']} шагов/с, {res['tokens_per_sec']:,.0f} токенов/с)")
    print(f"  пик RSS        {res['peak_rss_mb']:8.1f} МБ")
    share(res)

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  записано: {out}")

    if args.profile:
        print("\n" + "=" * 70)
        print(run_cprofile())


if __name__ == "__main__":
    main()
