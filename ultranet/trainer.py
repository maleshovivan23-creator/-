"""Тренер: цикл обучения, валидация, ранняя остановка, EMA, чекпоинты."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .data import DataLoader
from .functional import accuracy, cross_entropy
from .nn import Module
from .optim import EMA, Optimizer
from .checkpoint import load_checkpoint, save_checkpoint
from .tensor import Tensor, no_grad


def _fmt_time(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else f"{int(sec // 60)}m{int(sec % 60):02d}s"


def _bar(frac: float, width: int = 20) -> str:
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


class Trainer:
    """Универсальный цикл обучения.

    Возможности: валидация, ранняя остановка с откатом к лучшим весам,
    накопление градиентов, EMA-веса, чекпоинты, коллбэки, прогресс-бар.
    """

    def __init__(self, model: Module, optimizer: Optimizer,
                 loss_fn: Callable = cross_entropy,
                 scheduler=None, grad_clip: Optional[float] = None,
                 metric: Optional[Callable] = accuracy, verbose: bool = True,
                 accum_steps: int = 1, ema_decay: Optional[float] = None,
                 checkpoint_path: Optional[str] = None,
                 callbacks: Optional[List[Callable]] = None) -> None:
        self.model, self.opt = model, optimizer
        self.loss_fn, self.scheduler = loss_fn, scheduler
        self.grad_clip, self.metric, self.verbose = grad_clip, metric, verbose
        self.accum_steps = max(1, accum_steps)
        self.checkpoint_path = checkpoint_path
        self.callbacks = callbacks or []
        self.ema = EMA(model.parameters(), ema_decay) if ema_decay else None
        self.history: Dict[str, List[float]] = {
            "loss": [], "val_loss": [], "metric": [], "val_metric": [], "lr": [], "grad_norm": []
        }
        self.best_score = np.inf
        self._stop = False

    # ------------------------------------------------------------------ шаги
    def _forward(self, xb: np.ndarray, yb: np.ndarray):
        inp = xb if np.issubdtype(np.asarray(xb).dtype, np.integer) else Tensor(xb)
        out = self.model(inp)
        return out, self.loss_fn(out, yb)

    def _train_epoch(self, loader: DataLoader, epoch: int, epochs: int):
        self.model.train()
        losses, metrics, gnorms = [], [], []
        n_batches = len(loader)
        t0 = time.time()
        self.opt.zero_grad()

        for i, (xb, yb) in enumerate(loader, 1):
            out, loss = self._forward(xb, yb)
            (loss * (1.0 / self.accum_steps)).backward()

            if i % self.accum_steps == 0 or i == n_batches:
                gn = self.opt.clip_grad_norm(self.grad_clip) if self.grad_clip else 0.0
                gnorms.append(gn)
                self.opt.step()
                self.opt.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update()

            losses.append(loss.item())
            if self.metric:
                metrics.append(self.metric(out, yb))

            if self.verbose and (i % max(1, n_batches // 10) == 0 or i == n_batches):
                print(f"\r  эпоха {epoch}/{epochs} {_bar(i / n_batches)} "
                      f"{i}/{n_batches} loss={np.mean(losses[-50:]):.4f}", end="", flush=True)

        if self.verbose:
            print("\r" + " " * 80, end="\r")
        return float(np.mean(losses)), float(np.mean(metrics)) if metrics else 0.0, \
            float(np.mean(gnorms)) if gnorms else 0.0, time.time() - t0

    # ------------------------------------------------------------------- цикл
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 10, patience: Optional[int] = None,
            monitor: str = "val_loss") -> Dict[str, List[float]]:
        bad, best_state = 0, None
        self.best_score = np.inf

        for ep in range(1, epochs + 1):
            loss, metric, gnorm, dt = self._train_epoch(train_loader, ep, epochs)
            self.history["loss"].append(loss)
            self.history["metric"].append(metric)
            self.history["lr"].append(self.opt.lr)
            self.history["grad_norm"].append(gnorm)

            msg = f"эпоха {ep:3d}/{epochs} | loss {loss:.4f} | metric {metric:.4f}"
            if val_loader is not None:
                vl, vm = self.evaluate(val_loader, use_ema=self.ema is not None)
                self.history["val_loss"].append(vl)
                self.history["val_metric"].append(vm)
                msg += f" | val_loss {vl:.4f} | val_metric {vm:.4f}"
                score = vl if monitor == "val_loss" else -vm
            else:
                score = loss
            msg += f" | lr {self.opt.lr:.2e} | {_fmt_time(dt)}"
            if self.verbose:
                print(msg)

            improved = score < self.best_score - 1e-6
            if improved:
                self.best_score, bad = score, 0
                best_state = self.model.state_dict()
                if self.checkpoint_path:
                    save_checkpoint(self.checkpoint_path, self.model, self.opt,
                                    epoch=ep, step=self.opt.t, history=self.history,
                                    best_score=self.best_score)
            else:
                bad += 1

            for cb in self.callbacks:
                cb(self, ep)
            if self._stop:
                if self.verbose:
                    print("остановлено коллбэком")
                break
            if patience is not None and bad >= patience:
                if self.verbose:
                    print(f"ранняя остановка на эпохе {ep} (лучший {monitor}={self.best_score:.4f})")
                if best_state:
                    self.model.load_state_dict(best_state)
                break

        return self.history

    def resume(self, path: str) -> int:
        """Продолжить обучение из чекпоинта; возвращает номер эпохи."""
        ck = load_checkpoint(path, self.model, self.opt)
        if ck.get("history"):
            self.history = ck["history"]
        self.best_score = ck.get("extra", {}).get("best_score", np.inf)
        return int(ck.get("epoch", 0))

    def stop(self) -> None:
        """Прервать обучение из коллбэка."""
        self._stop = True

    # -------------------------------------------------------------- инференс
    @no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = False):
        if use_ema and self.ema is not None:
            self.ema.apply()
        self.model.eval()
        losses, metrics = [], []
        for xb, yb in loader:
            out, loss = self._forward(xb, yb)
            losses.append(loss.item())
            if self.metric:
                metrics.append(self.metric(out, yb))
        self.model.train()
        if use_ema and self.ema is not None:
            self.ema.restore()
        return float(np.mean(losses)), float(np.mean(metrics)) if metrics else 0.0

    @no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        inp = x if np.issubdtype(np.asarray(x).dtype, np.integer) else Tensor(x)
        out = self.model(inp).data
        self.model.train()
        return out

    def plot_history(self, width: int = 56, height: int = 12) -> str:
        """ASCII-график кривых обучения (работает без matplotlib)."""
        tr, va = self.history["loss"], self.history["val_loss"]
        if not tr:
            return "(история пуста)"
        series = [("train", tr, "*"), ("val", va, "o")] if va else [("train", tr, "*")]
        lo = min(min(s) for _, s, _ in series)
        hi = max(max(s) for _, s, _ in series)
        rng = max(hi - lo, 1e-9)
        grid = [[" "] * width for _ in range(height)]
        for _, s, ch in series:
            for i, v in enumerate(s):
                col = int(i / max(len(s) - 1, 1) * (width - 1))
                row = height - 1 - int((v - lo) / rng * (height - 1))
                grid[row][col] = ch
        lines = [f"{hi:7.4f} ┤" + "".join(grid[0])]
        lines += ["        │" + "".join(r) for r in grid[1:-1]]
        lines.append(f"{lo:7.4f} ┤" + "".join(grid[-1]))
        lines.append("        └" + "─" * width)
        legend = "  ".join(f"{ch} {name}" for name, _, ch in series)
        lines.append(f"         эпохи 1..{len(tr)}    {legend}")
        return "\n".join(lines)
