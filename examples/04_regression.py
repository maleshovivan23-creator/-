"""Пример 4: нелинейная регрессия + сравнение функций потерь и ASCII-график."""
import numpy as np

import ultranet as un
from ultranet import Tensor


def ascii_plot(x, y_true, y_pred, width=64, height=16) -> str:
    order = np.argsort(x.ravel())
    xs, yt, yp = x.ravel()[order], y_true.ravel()[order], y_pred.ravel()[order]
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    rng = max(hi - lo, 1e-9)
    grid = [[" "] * width for _ in range(height)]
    for series, ch in ((yt, "."), (yp, "#")):
        for i in range(len(xs)):
            col = int((xs[i] - xs.min()) / max(float(np.ptp(xs)), 1e-9) * (width - 1))
            row = height - 1 - int((series[i] - lo) / rng * (height - 1))
            grid[row][col] = ch
    return "\n".join("".join(r) for r in grid) + "\n  (. истина, # предсказание)"


def main() -> None:
    np.random.seed(0)
    x, y = un.make_regression(n=400, n_features=1, noise=0.12, seed=0)
    xtr, ytr, xte, yte = un.train_test_split(x, y, 0.2, seed=0)

    results = {}
    for name, loss_fn in [("MSE", un.mse_loss), ("MAE", un.mae_loss), ("Huber", un.huber_loss)]:
        np.random.seed(1)
        model = un.MLP([1, 64, 64, 1], activation="tanh")
        opt = un.Adam(model.parameters(), lr=5e-3)
        sched = un.OneCycleLR(opt, max_lr=5e-3, total=400)
        for _ in range(400):
            loss = loss_fn(model(Tensor(xtr)), ytr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with un.no_grad():
            pred = model(Tensor(xte)).data
        rmse = float(np.sqrt(((pred - yte) ** 2).mean()))
        results[name] = (rmse, model)
        print(f"{name:>6}: test RMSE = {rmse:.4f}")

    best = min(results, key=lambda k: results[k][0])
    print(f"\nлучшая функция потерь: {best}")
    model = results[best][1]
    with un.no_grad():
        pred = model(Tensor(x)).data
    print("\nаппроксимация y = sin(3x):")
    print(ascii_plot(x, y, pred))


if __name__ == "__main__":
    main()
