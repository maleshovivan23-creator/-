"""Proteus — единый фасад: один объект, живущий на любом железе.

Собирает вместе байтовый вход, вложенную модель, мета-контроллер,
локальную память и рой. Это реализация цикла из Части 2 спецификации:
приём -> профилирование -> сборка -> мышление -> память -> ранний выход -> ответ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..tensor import no_grad
from .bytes import ByteTokenizer
from .controller import MetaController, Plan
from .devices import Device, DeviceState, get_device
from .matformer import MatConfig, MatFormer
from .memory import LocalMemory
from .quant import TernaryModel
from .swarm import Swarm


@dataclass
class Response:
    """Результат одного запроса — со всей телеметрией «как думал»."""

    text: str
    plan: Plan
    layers_used: int
    context_used: str = ""
    swarm_hops: int = 0
    stayed_local: bool = True

    def __str__(self) -> str:
        return self.text

    def trace(self) -> str:
        lines = [
            f"  план      : {self.plan}",
            f"  слоёв     : {self.layers_used}",
            f"  локально  : {'да' if self.stayed_local else 'НЕТ'}",
        ]
        if self.context_used:
            lines.append(f"  из памяти : {self.context_used[:60]}...")
        if self.swarm_hops:
            lines.append(f"  прыжков   : {self.swarm_hops}")
        return "\n".join(lines)


class Proteus:
    """Один Протей: байтовый вход, вложенная модель, адаптация под железо.

    >>> p = Proteus.tiny()
    >>> p.attach("phone")
    >>> r = p.respond("привет", max_new_tokens=8)
    >>> r.stayed_local
    True
    """

    def __init__(self, model: Optional[MatFormer] = None,
                 memory: Optional[LocalMemory] = None,
                 device: str | Device = "phone") -> None:
        self.tok = ByteTokenizer()
        self.model = model or MatFormer(MatConfig())
        self.memory = memory or LocalMemory()
        self.controller = MetaController(self.model)
        self.state = DeviceState(get_device(device) if isinstance(device, str) else device)
        self.swarm: Optional[Swarm] = None
        self.cloud_enabled = False        # по умолчанию — никогда

    # ------------------------------------------------------------ конструкторы
    @classmethod
    def tiny(cls, **kw) -> "Proteus":
        """Маленький Протей для тестов и микроконтроллеров."""
        cfg = MatConfig(n_layer=4, n_embd=96, head_dim=32, block_size=48,
                        widths=(0.33, 0.67, 1.0), exit_layers=(1,))
        return cls(MatFormer(cfg), **kw)

    @classmethod
    def standard(cls, **kw) -> "Proteus":
        cfg = MatConfig(n_layer=8, n_embd=256, head_dim=32, block_size=128,
                        widths=(0.25, 0.5, 1.0), exit_layers=(2, 5))
        return cls(MatFormer(cfg), **kw)

    # ------------------------------------------------------------- устройства
    def attach(self, device: str | Device, **state_kw) -> "Proteus":
        """Переехать на устройство (или описать его текущее состояние)."""
        dev = get_device(device) if isinstance(device, str) else device
        self.state = DeviceState(dev, **state_kw)
        return self

    def form_swarm(self, devices: Sequence[str], width: float = 1.0) -> Swarm:
        """Собрать рой из нескольких устройств."""
        states = [DeviceState(get_device(d)) for d in devices]
        self.swarm = Swarm(self.model, states, width=width)
        return self.swarm

    def dissolve_swarm(self) -> None:
        self.swarm = None

    # ------------------------------------------------------------------ знание
    def remember(self, *facts: str) -> None:
        """Положить знание в локальную память (вне весов)."""
        for f in facts:
            self.memory.add(f)

    # ---------------------------------------------------------------- ответ
    @no_grad()
    def respond(self, prompt: str, max_new_tokens: int = 48,
                task_complexity: Optional[float] = None,
                temperature: float = 0.7, top_k: int = 40,
                use_memory: bool = True, seed: Optional[int] = None) -> Response:
        """Полный цикл: профилирование -> план -> (память) -> генерация."""
        complexity = self.estimate_complexity(prompt) if task_complexity is None else task_complexity
        plan = self.controller.plan(self.state, complexity)

        context = ""
        if use_memory and len(self.memory):
            context = self.memory.context_for(prompt, top_k=2, max_chars=200)

        full = f"{context}\n{prompt}" if context else prompt
        ids = self.tok.encode(full)[-self.model.cfg.block_size:]
        if not ids:
            ids = [self.tok.bos_id]

        if self.swarm is not None:
            logits = self.swarm.forward(np.array([ids]))
            out_ids = self._sample_loop(ids, plan, max_new_tokens, temperature, top_k, seed)
            hops = self.swarm.hops
        else:
            out_ids = self._sample_loop(ids, plan, max_new_tokens, temperature, top_k, seed)
            hops = 0

        text = self.tok.decode(out_ids[len(ids):])
        return Response(text=text, plan=plan, layers_used=plan.n_layers,
                        context_used=context, swarm_hops=hops,
                        stayed_local=not self.cloud_enabled)

    def _sample_loop(self, ids: List[int], plan: Plan, n: int,
                     temperature: float, top_k: int, seed: Optional[int]) -> List[int]:
        return self.model.generate(ids, max_new_tokens=n, width=plan.width,
                                   temperature=temperature, top_k=top_k,
                                   n_layers=plan.n_layers, seed=seed)

    @staticmethod
    def estimate_complexity(prompt: str) -> float:
        """Грубая оценка сложности запроса: команда или размышление."""
        p = prompt.lower().strip()
        simple_markers = ("включи", "выключи", "открой", "закрой", "стоп", "да", "нет",
                          "on", "off", "open", "close", "yes", "no")
        if any(p.startswith(m) for m in simple_markers) or len(p) < 12:
            return 0.15
        hard_markers = ("почему", "объясни", "проанализируй", "докажи", "напиши код",
                        "why", "explain", "analyze", "prove")
        if any(m in p for m in hard_markers) or len(p) > 120:
            return 0.95
        return 0.5

    # --------------------------------------------------------------- отчёты
    def footprint(self) -> str:
        """Сколько Протей весит на текущем устройстве."""
        q = TernaryModel(self.model)
        plan = self.controller.plan(self.state, 1.0)
        return (f"{self.state}\n"
                f"  полная модель : {q.stats}\n"
                f"  активная часть: {plan.active_params:,} параметров "
                f"({plan.weight_bytes / 1024:.0f} КБ)\n"
                f"  память знаний : {self.memory.size_bytes() / 1024:.1f} КБ "
                f"({len(self.memory)} фактов, вне весов)")

    def device_matrix(self, devices: Optional[Sequence[str]] = None,
                      task_complexity: float = 0.8) -> str:
        """Таблица: как Протей выглядит на разных устройствах."""
        keys = list(devices or ["smartcard", "esp32", "ring", "earbuds", "smartwatch",
                                "phone", "laptop", "workstation", "server"])
        rows = [f"{'устройство':<22} {'RAM':>9} {'активно':>12} {'вес':>10} {'слоёв':>6}"]
        rows.append("-" * 64)
        for k in keys:
            dev = get_device(k)
            plan = self.controller.plan(DeviceState(dev), task_complexity)
            ram = (f"{dev.ram_bytes / 1024**3:.0f}ГБ" if dev.ram_bytes >= 1024**3
                   else f"{dev.ram_bytes / 1024**2:.0f}МБ" if dev.ram_bytes >= 1024**2
                   else f"{dev.ram_bytes / 1024:.0f}КБ")
            w = (f"{plan.weight_bytes / 1024**2:.1f}МБ" if plan.weight_bytes >= 1024**2
                 else f"{plan.weight_bytes / 1024:.0f}КБ")
            mark = "" if plan.fits else "  (не влезает!)"
            rows.append(f"{dev.name:<22} {ram:>9} {plan.active_params:>12,} {w:>10} "
                        f"{plan.n_layers:>6}{mark}")
        return "\n".join(rows)
