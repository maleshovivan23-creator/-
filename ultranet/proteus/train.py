"""Matryoshka-обучение: один проход учит сразу все вложенные размеры.

Без этого срез весов — просто обрезанная матрица, которая ничего не умеет.
Смысл MatFormer именно в том, что каждый размер обучается явно.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..functional import cross_entropy, perplexity
from ..optim import AdamW, CosineWarmup, Optimizer
from .bytes import make_byte_batches
from .matformer import MatFormer


def train_matryoshka(model: MatFormer, ids: Sequence[int], steps: int = 300,
                     batch_size: int = 8, lr: float = 3e-3,
                     widths: Optional[Sequence[float]] = None,
                     exit_loss_weight: float = 0.3,
                     grad_clip: float = 1.0, log_every: int = 50,
                     verbose: bool = True) -> Dict[str, List[float]]:
    """Обучить все вложенные размеры одновременно.

    На каждом шаге считается loss для каждой ширины и суммируется, поэтому
    градиент получают и малые срезы (они входят во все большие), и полная модель.
    Ранние выходы обучаются вспомогательным loss — иначе early exit бесполезен.
    """
    widths = list(widths or model.cfg.widths)
    block = model.cfg.block_size
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = CosineWarmup(opt, warmup=max(10, steps // 10), total=steps)
    history: Dict[str, List[float]] = {f"w{w}": [] for w in widths}
    history["total"] = []

    t0 = time.time()
    for step in range(1, steps + 1):
        xb, yb = make_byte_batches(ids, block, batch_size)
        opt.zero_grad()
        total = None
        per_width = {}
        for w in widths:
            use_exits = exit_loss_weight > 0 and bool(model.cfg.exit_layers)
            out = model(xb, width=w, return_exits=use_exits)
            logits, exits = out if use_exits else (out, [])
            loss = cross_entropy(logits, yb)
            per_width[w] = loss.item()
            term = loss
            for _, ex_logits in exits:
                term = term + cross_entropy(ex_logits, yb) * exit_loss_weight
            total = term if total is None else total + term
        total.backward()
        opt.clip_grad_norm(grad_clip)
        opt.step()
        sched.step()

        for w in widths:
            history[f"w{w}"].append(per_width[w])
        history["total"].append(float(total.item()))

        if verbose and (step % log_every == 0 or step == 1):
            parts = " | ".join(f"w{w}: {per_width[w]:.3f}" for w in widths)
            print(f"  шаг {step:4d}/{steps} | {parts} | {time.time() - t0:.1f}s")

    return history


def evaluate_widths(model: MatFormer, ids: Sequence[int], widths: Sequence[float],
                    n_batches: int = 8, batch_size: int = 8,
                    seed: int = 0) -> Dict[float, Dict[str, float]]:
    """Замерить loss/перплексию каждого вложенного размера на одних данных."""
    from ..tensor import no_grad

    block = model.cfg.block_size
    was = model.training
    model.eval()
    out: Dict[float, Dict[str, float]] = {}
    with no_grad():
        for w in widths:
            losses = []
            for i in range(n_batches):
                xb, yb = make_byte_batches(ids, block, batch_size, seed=seed + i)
                losses.append(cross_entropy(model(xb, width=w), yb).item())
            mean = float(np.mean(losses))
            out[w] = {
                "loss": mean,
                "ppl": perplexity(mean),
                "bpb": mean / np.log(2),          # бит на байт — метрика байтовых LM
                "active_params": float(model.active_params(w)),
            }
    model.train(was)
    return out
