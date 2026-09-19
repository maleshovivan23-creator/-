"""Пример 6: BPE-токенизация, чекпоинты и возобновление обучения.

Показывает рабочий цикл «как в больших фреймворках»:
  обучение -> сохранение полного состояния -> падение -> возобновление с того же места.
"""
import os
import tempfile

import numpy as np

import ultranet as un

CORPUS = (
    "нейронная сеть учится на данных. градиент течёт назад через граф вычислений. "
    "трансформер смотрит на контекст через механизм внимания. "
    "модель предсказывает следующий токен последовательности. "
) * 60


def main() -> None:
    un.manual_seed(1234)          # полная воспроизводимость

    # 1. BPE против посимвольной токенизации
    char = un.CharTokenizer(CORPUS)
    bpe = un.BPETokenizer.train(CORPUS, vocab_size=400)
    print("Токенизация одного предложения:")
    sample = "нейронная сеть учится на данных."
    print(f"  посимвольно : {len(char.encode(sample)):3d} токенов, словарь {char.vocab_size}")
    print(f"  BPE         : {len(bpe.encode(sample)):3d} токенов, словарь {bpe.vocab_size} "
          f"(сжатие x{bpe.compression_ratio(sample):.2f})")
    print(f"  roundtrip корректен: {bpe.decode(bpe.encode(sample)) == sample}")

    ids = bpe.encode(CORPUS)
    print(f"\nкорпус: {len(CORPUS)} символов -> {len(ids)} BPE-токенов")

    # 2. Модель и её паспорт
    cfg = un.GPTConfig.llama_style(bpe.vocab_size, block_size=32, n_layer=2,
                                   n_head=4, n_embd=64, dropout=0.05)
    model = un.GPT(cfg)
    print("\n" + model.summary(max_rows=6))

    opt = un.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    ckpt = os.path.join(tempfile.gettempdir(), "ultranet_demo.pkl")

    # 3. Первый этап обучения
    print("\n--- этап 1: 60 шагов ---")
    for step in range(1, 61):
        xb, yb = un.make_lm_batches(ids, cfg.block_size, 8)
        loss = un.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step()
        if step % 20 == 0:
            print(f"  шаг {step:3d} | loss {loss.item():.4f} | ppl {un.perplexity(loss.item()):.2f}")

    un.save_checkpoint(ckpt, model, opt, step=opt.t, config=cfg)
    loss_before = loss.item()
    moment_before = opt.m[0].copy()
    print(f"чекпоинт сохранён: шаг {opt.t}, loss {loss_before:.4f}")

    # 4. «Перезапуск процесса»: новые объекты, состояние из файла
    print("\n--- перезапуск: восстанавливаем модель И оптимизатор ---")
    model2 = un.GPT(cfg)
    opt2 = un.AdamW(model2.parameters(), lr=999.0)   # заведомо неверный lr
    ck = un.load_checkpoint(ckpt, model2, opt2)
    print(f"  восстановлено: шаг {opt2.t}, lr {opt2.lr} (был перезаписан из файла)")
    print(f"  моменты Adam идентичны: {np.allclose(moment_before, opt2.m[0])}")

    model.eval(); model2.eval()      # без dropout — иначе сравнение зашумлено
    with un.no_grad():
        xb, yb = un.make_lm_batches(ids, cfg.block_size, 8, seed=0)
        l1 = un.cross_entropy(model(xb), yb).item()
        l2 = un.cross_entropy(model2(xb), yb).item()
    print(f"  loss до/после загрузки: {l1:.6f} / {l2:.6f} -> совпадает: {abs(l1 - l2) < 1e-6}")

    # 5. Продолжаем обучение как ни в чём не бывало
    model2.train()
    print("\n--- этап 2: ещё 60 шагов из чекпоинта ---")
    for step in range(61, 121):
        xb, yb = un.make_lm_batches(ids, cfg.block_size, 8)
        loss = un.cross_entropy(model2(xb), yb)
        opt2.zero_grad()
        loss.backward()
        opt2.clip_grad_norm(1.0)
        opt2.step()
        if step % 20 == 0:
            print(f"  шаг {step:3d} | loss {loss.item():.4f} | ppl {un.perplexity(loss.item()):.2f}")

    print(f"\nloss улучшился: {loss_before:.4f} -> {loss.item():.4f}")

    out = model2.generate(bpe.encode("нейронная сеть"), 40, temperature=0.6,
                          top_p=0.9, repetition_penalty=1.1, seed=0)
    print(f"\nсэмпл: {bpe.decode(out)}")
    os.remove(ckpt)


if __name__ == "__main__":
    main()
