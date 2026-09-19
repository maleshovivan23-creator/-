"""Бенчмарк: скорость ключевых операций и выигрыш от оптимизаций."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

import ultranet as un
from ultranet import nn
from ultranet.tensor import Tensor, no_grad


def timeit(fn: Callable, n: int = 20, warmup: int = 3) -> float:
    """Среднее время вызова в миллисекундах."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def _row(name: str, ms: float, extra: str = "") -> None:
    print(f"  {name:<38} {ms:8.2f} ms  {extra}")


def bench_ops() -> None:
    print("\n▸ Базовые операции (batch=128, dim=256)")
    x = Tensor(np.random.randn(128, 256).astype(np.float32), requires_grad=True)
    w = Tensor(np.random.randn(256, 256).astype(np.float32), requires_grad=True)

    def fwd_bwd():
        x.grad = w.grad = None
        ((x @ w).gelu() ** 2).sum().backward()

    _row("matmul+gelu forward+backward", timeit(fwd_bwd))
    _row("softmax (fused)", timeit(lambda: x.softmax(-1)))

    logits = Tensor(np.random.randn(256, 1000).astype(np.float32), requires_grad=True)
    y = np.random.randint(0, 1000, 256)

    def ce():
        logits.grad = None
        un.cross_entropy(logits, y).backward()

    _row("cross_entropy fwd+bwd (vocab=1000)", timeit(ce))


def bench_grad_mode() -> None:
    print("\n▸ Выигрыш no_grad() на инференсе")
    m = un.MLP([256, 512, 512, 10])
    x = Tensor(np.random.randn(128, 256).astype(np.float32))
    with_graph = timeit(lambda: m(x), n=20)

    def infer():
        with no_grad():
            m(x)

    without = timeit(infer, n=20)
    _row("forward со сборкой графа", with_graph)
    _row("forward под no_grad()", without, f"ускорение x{with_graph / max(without, 1e-9):.2f}")


def bench_layers() -> None:
    print("\n▸ Слои")
    conv = nn.Conv2d(16, 32, 3, padding=1)
    img = Tensor(np.random.randn(16, 16, 32, 32).astype(np.float32), requires_grad=True)

    def conv_fb():
        img.grad = None
        conv(img).sum().backward()

    _row("Conv2d 16->32, 16x32x32 fwd+bwd", timeit(conv_fb, n=5))

    attn = nn.MultiHeadAttention(256, 8)
    seq = Tensor(np.random.randn(8, 128, 256).astype(np.float32), requires_grad=True)

    def attn_fb():
        seq.grad = None
        attn(seq).sum().backward()

    _row("Attention B8 T128 D256 fwd+bwd", timeit(attn_fb, n=5))

    ln, rn = nn.LayerNorm(512), nn.RMSNorm(512)
    z = Tensor(np.random.randn(256, 512).astype(np.float32))
    lm, rm = timeit(lambda: ln(z)), timeit(lambda: rn(z))
    _row("LayerNorm", lm)
    _row("RMSNorm", rm, f"x{lm / max(rm, 1e-9):.2f} быстрее LayerNorm")


def bench_generation() -> None:
    print("\n▸ Генерация GPT: KV-кэш против полного пересчёта")
    cfg = un.GPTConfig(vocab_size=96, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = un.GPT(cfg).eval()
    print(f"  модель: {model.num_params():,} параметров, block_size={cfg.block_size}")
    prompt = list(np.random.randint(0, 96, 64))
    n_tok = 32

    t0 = time.perf_counter()
    model.generate(prompt, n_tok, temperature=0.8, use_cache=False, seed=0)
    slow = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    model.generate(prompt, n_tok, temperature=0.8, use_cache=True, seed=0)
    fast = (time.perf_counter() - t0) * 1000

    _row(f"без кэша ({n_tok} токенов)", slow, f"{slow / n_tok:.1f} ms/токен")
    _row(f"с KV-кэшем ({n_tok} токенов)", fast,
         f"{fast / n_tok:.1f} ms/токен — x{slow / max(fast, 1e-9):.2f}")


def bench_training() -> None:
    print("\n▸ Пропускная способность обучения")
    cfg = un.GPTConfig(vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=128)
    model = un.GPT(cfg)
    opt = un.AdamW(model.parameters(), lr=1e-3)
    ids = list(np.random.randint(0, 64, 5000))
    batch = 8

    def step():
        xb, yb = un.make_lm_batches(ids, cfg.block_size, batch)
        loss = un.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ms = timeit(step, n=5, warmup=2)
    tps = batch * cfg.block_size / (ms / 1000)
    _row(f"GPT {model.num_params():,} парам., шаг обучения", ms, f"{tps:,.0f} токенов/с")


def run_all() -> None:
    np.random.seed(0)
    print("=" * 72)
    print("UltraNet — бенчмарк (чистый NumPy, CPU)".center(72))
    print("=" * 72)
    bench_ops()
    bench_grad_mode()
    bench_layers()
    bench_generation()
    bench_training()
    print("\n" + "=" * 72)


if __name__ == "__main__":
    run_all()
