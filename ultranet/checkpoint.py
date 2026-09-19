"""Чекпоинты: сохранение модели вместе с состоянием оптимизатора и метаданными.

В отличие от `model.save()`, позволяет продолжить обучение ровно с того места,
где оно прервалось (моменты Adam, счётчик шагов, расписание lr).
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np

from .nn import Module
from .optim import Optimizer


def _optim_state(opt: Optimizer) -> Dict[str, Any]:
    state: Dict[str, Any] = {"cls": type(opt).__name__, "lr": opt.lr, "t": opt.t}
    for attr in ("m", "v", "buf", "sq", "slow"):
        if hasattr(opt, attr):
            state[attr] = [np.asarray(a).copy() for a in getattr(opt, attr)]
    return state


def _load_optim_state(opt: Optimizer, state: Dict[str, Any]) -> None:
    if state.get("cls") != type(opt).__name__:
        raise ValueError(
            f"оптимизатор не совпадает: в чекпоинте {state.get('cls')}, передан {type(opt).__name__}"
        )
    opt.lr, opt.t = state["lr"], state["t"]
    for attr in ("m", "v", "buf", "sq", "slow"):
        if attr in state and hasattr(opt, attr):
            setattr(opt, attr, [np.asarray(a) for a in state[attr]])


def save_checkpoint(path: str, model: Module, optimizer: Optional[Optimizer] = None,
                    epoch: int = 0, step: int = 0, config: Any = None,
                    history: Optional[Dict] = None, **extra) -> None:
    """Сохранить полное состояние обучения."""
    payload = {
        "format": 1,
        "model": model.state_dict(),
        "optimizer": _optim_state(optimizer) if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
        "history": history,
        "extra": extra,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str, model: Optional[Module] = None,
                    optimizer: Optional[Optimizer] = None,
                    strict: bool = True) -> Dict[str, Any]:
    """Загрузить чекпоинт; при переданных model/optimizer — восстановить их состояние."""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if model is not None:
        sd = payload["model"]
        if strict:
            have = set(dict(model.named_parameters()))
            want = set(sd)
            if have != want:
                missing, unexpected = want - have, have - want
                raise ValueError(f"несовпадение параметров: нет {sorted(missing)[:3]}, "
                                 f"лишние {sorted(unexpected)[:3]}")
            for k, v in sd.items():
                p = dict(model.named_parameters())[k]
                if p.data.shape != np.asarray(v).shape:
                    raise ValueError(f"форма {k}: ожидалась {p.data.shape}, в файле {np.asarray(v).shape}")
        model.load_state_dict(sd)

    if optimizer is not None and payload.get("optimizer"):
        _load_optim_state(optimizer, payload["optimizer"])

    return payload
