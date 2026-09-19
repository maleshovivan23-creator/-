"""Тренер: цикл обучения, валидация, ранняя остановка, история метрик."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .data import DataLoader
from .functional import accuracy, cross_entropy
from .nn import Module
from .optim import Optimizer
from .tensor import Tensor


class Trainer:
    def __init__(self, model: Module, optimizer: Optimizer,
                 loss_fn: Callable = cross_entropy,
                 scheduler=None, grad_clip: Optional[float] = None,
                 metric: Optional[Callable] = accuracy, verbose: bool = True) -> None:
        self.model, self.opt = model, optimizer
        self.loss_fn, self.scheduler = loss_fn, scheduler
        self.grad_clip, self.metric, self.verbose = grad_clip, metric, verbose
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}

    def _run_batch(self, xb: np.ndarray, yb: np.ndarray, train: bool):
        out = self.model(Tensor(xb) if not np.issubdtype(xb.dtype, np.integer) else xb)
        loss = self.loss_fn(out, yb)
        if train:
            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip:
                self.opt.clip_grad_norm(self.grad_clip)
            self.opt.step()
            if self.scheduler:
                self.scheduler.step()
        m = self.metric(out, yb) if self.metric else 0.0
        return loss.item(), m

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 10, patience: Optional[int] = None) -> Dict[str, List[float]]:
        best, bad, best_state = np.inf, 0, None
        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            losses, metrics = [], []
            for xb, yb in train_loader:
                l, m = self._run_batch(xb, yb, True)
                losses.append(l)
                metrics.append(m)
            self.history["loss"].append(float(np.mean(losses)))
            self.history["metric"].append(float(np.mean(metrics)))

            msg = f"epoch {ep:3d}/{epochs} | loss {self.history['loss'][-1]:.4f} | metric {self.history['metric'][-1]:.4f}"
            if val_loader is not None:
                vl, vm = self.evaluate(val_loader)
                self.history["val_loss"].append(vl)
                self.history["val_metric"].append(vm)
                msg += f" | val_loss {vl:.4f} | val_metric {vm:.4f}"
                score = vl
            else:
                score = self.history["loss"][-1]
            msg += f" | {time.time() - t0:.2f}s"
            if self.verbose:
                print(msg)

            if patience is not None:
                if score < best - 1e-5:
                    best, bad, best_state = score, 0, self.model.state_dict()
                else:
                    bad += 1
                    if bad >= patience:
                        if self.verbose:
                            print(f"ранняя остановка на эпохе {ep} (лучший score={best:.4f})")
                        if best_state:
                            self.model.load_state_dict(best_state)
                        break
        return self.history

    def evaluate(self, loader: DataLoader):
        self.model.eval()
        losses, metrics = [], []
        for xb, yb in loader:
            l, m = self._run_batch(xb, yb, False)
            losses.append(l)
            metrics.append(m)
        self.model.train()
        return float(np.mean(losses)), float(np.mean(metrics))

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = self.model(Tensor(x) if not np.issubdtype(np.asarray(x).dtype, np.integer) else x)
        self.model.train()
        return out.data
