"""Пример 3: GPT-трансформер учится генерировать текст посимвольно."""
import numpy as np

import ultranet as un

CORPUS = (
    "нейронная сеть учится на данных. "
    "градиент течёт назад через граф вычислений. "
    "трансформер смотрит на контекст через внимание. "
    "модель предсказывает следующий символ. "
) * 40


def main() -> None:
    np.random.seed(1337)
    tok = un.CharTokenizer(CORPUS)
    ids = tok.encode(CORPUS)
    print(f"символов: {len(ids)}, размер словаря: {tok.vocab_size}")

    cfg = un.GPTConfig(vocab_size=tok.vocab_size, block_size=48, n_layer=3, n_head=4,
                       n_embd=96, dropout=0.05)
    model = un.GPT(cfg)
    print(f"параметров: {model.num_params():,}")

    steps, batch = 400, 16
    opt = un.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = un.CosineWarmup(opt, warmup=40, total=steps)

    for step in range(1, steps + 1):
        xb, yb = un.make_lm_batches(ids, cfg.block_size, batch)
        loss = un.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step()
        lr = sched.step()
        if step % 50 == 0 or step == 1:
            ppl = float(np.exp(min(loss.item(), 20)))
            print(f"шаг {step:4d}/{steps} | loss {loss.item():.4f} | ppl {ppl:7.2f} | lr {lr:.5f}")

    for temp in (0.4, 0.8):
        out = model.generate(tok.encode("нейронная "), max_new_tokens=160,
                             temperature=temp, top_k=8, seed=0)
        print(f"\n--- сэмпл (temperature={temp}) ---\n{tok.decode(out)}")


if __name__ == "__main__":
    main()
