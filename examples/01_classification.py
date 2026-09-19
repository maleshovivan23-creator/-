"""Пример 1: классификация «спиралей» полносвязной сетью + ASCII-визуализация."""
import numpy as np

import ultranet as un


def main() -> None:
    np.random.seed(42)
    x, y = un.make_spirals(n_per_class=400, n_classes=3, noise=0.18, seed=42)
    xtr, ytr, xte, yte = un.train_test_split(x, y, test_size=0.2, seed=42)

    model = un.MLP([2, 96, 96, 64, 3], activation="gelu", dropout=0.05)
    print(f"параметров: {model.num_params():,}")

    opt = un.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    trainer = un.Trainer(model, opt, grad_clip=1.0)
    trainer.fit(
        un.DataLoader(xtr, ytr, batch_size=64),
        un.DataLoader(xte, yte, batch_size=128, shuffle=False),
        epochs=40,
        patience=10,
    )

    loss, acc = trainer.evaluate(un.DataLoader(xte, yte, 128, shuffle=False))
    print(f"\nТест: loss={loss:.4f}  accuracy={acc * 100:.2f}%")

    # карта решений в ASCII
    print("\nГраницы решений (символ = предсказанный класс):")
    gy, gx = np.mgrid[-1.2:1.2:25j, -1.2:1.2:50j]
    grid = np.c_[gx.ravel(), gy.ravel()].astype(np.float32)
    pred = trainer.predict(grid).argmax(1).reshape(gx.shape)
    chars = ".oO"
    for row in pred[::-1]:
        print("".join(chars[c] for c in row))


if __name__ == "__main__":
    main()
