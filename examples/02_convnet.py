"""Пример 2: свёрточная сеть на синтетических изображениях 8x8."""
import numpy as np

import ultranet as un


def main() -> None:
    np.random.seed(0)
    x, y = un.data.make_digits_like(n=1200, size=8, n_classes=4, seed=0)
    xtr, ytr, xte, yte = un.train_test_split(x, y, 0.2, seed=0)

    model = un.ConvNet(in_ch=1, n_classes=4, width=12, img_size=8)
    print(f"параметров: {model.num_params():,}")

    opt = un.Adam(model.parameters(), lr=2e-3)
    sched = un.CosineWarmup(opt, warmup=20, total=12 * (len(xtr) // 64))
    trainer = un.Trainer(model, opt, scheduler=sched, grad_clip=2.0)
    trainer.fit(un.DataLoader(xtr, ytr, 64), un.DataLoader(xte, yte, 128, shuffle=False), epochs=12)

    loss, acc = trainer.evaluate(un.DataLoader(xte, yte, 128, shuffle=False))
    print(f"\nТест: loss={loss:.4f}  accuracy={acc * 100:.2f}%")

    names = ["полоса", "крест", "рамка", "диагональ"]
    probs = trainer.predict(xte[:4])
    for i in range(4):
        print(f"\nистина={names[yte[i]]:10s} прогноз={names[int(probs[i].argmax())]}")
        for row in xte[i, 0]:
            print("".join("#" if v > 0.5 else "." for v in row))


if __name__ == "__main__":
    main()
