"""Пример 5: сравнение GPT-2 style и LLaMA style (RoPE + RMSNorm + SwiGLU)."""
import time

import numpy as np

import ultranet as un

CORPUS = (
    "нейронная сеть учится на данных. градиент течёт назад через граф вычислений. "
    "трансформер смотрит на контекст через механизм внимания. "
    "модель предсказывает следующий символ последовательности. "
) * 50


def train(cfg: un.GPTConfig, ids, steps: int = 250, label: str = "") -> tuple:
    np.random.seed(0)
    model = un.GPT(cfg)
    opt = un.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = un.CosineWarmup(opt, warmup=steps // 10, total=steps)
    t0 = time.time()
    losses = []
    for step in range(steps):
        xb, yb = un.make_lm_batches(ids, cfg.block_size, 16)
        loss = un.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    dt = time.time() - t0
    final = float(np.mean(losses[-20:]))
    print(f"{label:<14} параметров {model.num_params():>8,} | "
          f"loss {final:.4f} | ppl {un.perplexity(final):6.2f} | {dt:.1f}s")
    return model, final


def main() -> None:
    tok = un.CharTokenizer(CORPUS)
    ids = tok.encode(CORPUS)
    print(f"корпус: {len(ids):,} символов, словарь {tok.vocab_size}\n")

    common = dict(vocab_size=tok.vocab_size, block_size=48, n_layer=3, n_head=4,
                  n_embd=96, dropout=0.05)

    gpt2, _ = train(un.GPTConfig(**common), ids, label="GPT-2 style")
    llama, _ = train(un.GPTConfig(rope=True, norm="rms", swiglu=True, **common),
                     ids, label="LLaMA style")

    print("\n--- сэмплы (top-p 0.9, repetition_penalty 1.1) ---")
    for name, model in (("GPT-2 ", gpt2), ("LLaMA ", llama)):
        out = model.generate(tok.encode("нейронная "), 120, temperature=0.7,
                             top_p=0.9, repetition_penalty=1.1, seed=0)
        print(f"\n[{name}] {tok.decode(out)}")

    # RoPE умеет экстраполировать за block_size
    long_ctx = np.array([ids[:96]])
    print(f"\nLLaMA принимает контекст длиной {long_ctx.shape[1]} "
          f"при block_size={common['block_size']}: "
          f"{llama(long_ctx).shape} (RoPE экстраполирует)")

    # скорость генерации с кэшем и без
    t0 = time.time(); llama.generate(ids[:32], 24, use_cache=False, seed=0); slow = time.time() - t0
    t0 = time.time(); llama.generate(ids[:32], 24, use_cache=True, seed=0); fast = time.time() - t0
    print(f"генерация 24 токенов: без кэша {slow:.2f}s | с KV-кэшем {fast:.2f}s "
          f"(x{slow / max(fast, 1e-9):.1f})")


if __name__ == "__main__":
    main()
