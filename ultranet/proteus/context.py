"""Адаптивность к контексту и четыре петли адаптации по скоростям.

Из спецификации:
  миллисекунды — steering-векторы (без обучения);
  секунды      — мета-контроллер (ширина, слои, эксперты);
  часы         — LoRA-адаптеры (персонализация);
  недели       — база знаний / RAG (новые факты).

Контекст: время суток, место, активность, устройства рядом, сеть.
«За рулём -> короткие ответы голосом. Ночью -> тише и короче».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class Activity(str, Enum):
    IDLE = "покой"
    DRIVING = "за рулём"
    WALKING = "ходьба"
    WORKING = "работа"
    SLEEPING = "сон"
    MEETING = "встреча"


class Place(str, Enum):
    HOME = "дом"
    WORK = "работа"
    TRANSIT = "транспорт"
    OUTDOORS = "улица"
    UNKNOWN = "неизвестно"


@dataclass
class Context:
    """Ситуация вокруг пользователя прямо сейчас."""

    hour: int = 12                       # 0..23
    place: Place = Place.UNKNOWN
    activity: Activity = Activity.IDLE
    nearby_devices: List[str] = field(default_factory=list)
    network: str = "wifi"                # wifi | cellular | none
    headphones: bool = False

    @property
    def is_night(self) -> bool:
        return self.hour >= 23 or self.hour < 6

    @property
    def offline(self) -> bool:
        return self.network == "none"

    @property
    def hands_busy(self) -> bool:
        return self.activity in (Activity.DRIVING, Activity.WALKING)

    def __str__(self) -> str:
        return (f"{self.hour:02d}:00, {self.place.value}, {self.activity.value}, "
                f"сеть={self.network}, рядом={len(self.nearby_devices)}")


@dataclass
class Style:
    """Как отвечать в данном контексте."""

    max_tokens: int = 64
    voice: bool = False
    verbosity: float = 1.0        # 0 = телеграфно, 1 = обычно, 2 = подробно
    steering: Optional[str] = None
    reason: str = ""

    def __str__(self) -> str:
        v = "голосом" if self.voice else "текстом"
        return (f"{v}, до {self.max_tokens} токенов, подробность {self.verbosity:.1f}"
                + (f", вектор «{self.steering}»" if self.steering else "")
                + f" [{self.reason}]")


def style_for(ctx: Context) -> Style:
    """Правила из спецификации: контекст -> стиль ответа."""
    if ctx.activity is Activity.DRIVING:
        return Style(max_tokens=24, voice=True, verbosity=0.3,
                     steering="простота", reason="за рулём: коротко и голосом")
    if ctx.activity is Activity.SLEEPING or ctx.is_night:
        return Style(max_tokens=24, voice=ctx.headphones, verbosity=0.4,
                     steering="простота", reason="ночь: тихо и коротко")
    if ctx.activity is Activity.MEETING:
        return Style(max_tokens=32, voice=False, verbosity=0.5,
                     reason="встреча: только текст, коротко")
    if ctx.activity is Activity.WORKING and ctx.place is Place.WORK:
        return Style(max_tokens=128, voice=False, verbosity=1.5,
                     steering="детальность", reason="работа: подробно и технически")
    if ctx.activity is Activity.WALKING:
        return Style(max_tokens=32, voice=ctx.headphones, verbosity=0.5,
                     reason="в движении: кратко")
    return Style(reason="обычный режим")


@dataclass
class LoopEvent:
    loop: str
    what: str
    at_step: int


class AdaptationLoops:
    """Четыре петли, работающие на разных скоростях одновременно.

    >>> loops = AdaptationLoops(proteus)
    >>> loops.tick(ctx)        # каждый запрос
    >>> loops.log              # что сработало
    """

    # период срабатывания в «тиках» (условных запросах)
    PERIOD = {"мс": 1, "сек": 1, "часы": 20, "недели": 50}

    def __init__(self, proteus, consolidate_steps: int = 10) -> None:
        self.p = proteus
        self.steps = 0
        self.consolidate_steps = consolidate_steps
        self.log: List[LoopEvent] = []
        self.counts: Dict[str, int] = {k: 0 for k in self.PERIOD}

    def _fire(self, loop: str, what: str) -> None:
        self.counts[loop] += 1
        self.log.append(LoopEvent(loop, what, self.steps))

    def tick(self, ctx: Context, prompt: str = "") -> Style:
        """Один запрос: срабатывают те петли, чьё время пришло."""
        self.steps += 1
        style = style_for(ctx)

        # ---- петля 1: миллисекунды — steering, без обучения
        if style.steering and style.steering in self.p.steering:
            self.p.steering.apply(style.steering)
            self._fire("мс", f"вектор «{style.steering}»")
        # ---- петля 2: секунды — мета-контроллер меняет конфигурацию
        plan = self.p.controller.plan(self.p.state,
                                      self.p.estimate_complexity(prompt) if prompt else 0.5)
        self._fire("сек", f"план {plan.n_layers} слоёв, w={plan.width:.2f}")
        # ---- петля 3: часы — консолидация LoRA
        if self.steps % self.PERIOD["часы"] == 0 and self.p.adapter is not None:
            self.p.consolidate(steps=self.consolidate_steps)
            self._fire("часы", "консолидация LoRA")
        # ---- петля 4: недели — перестройка базы знаний
        if self.steps % self.PERIOD["недели"] == 0:
            n = len(self.p.memory) if self.p.memory is not None else 0
            self._fire("недели", f"переиндексация памяти ({n} фактов)")
        return style

    def report(self) -> str:
        names = {"мс": "миллисекунды (steering)", "сек": "секунды (мета-контроллер)",
                 "часы": "часы (LoRA)", "недели": "недели (знания)"}
        lines = [f"тиков: {self.steps}"]
        for k, v in self.counts.items():
            lines.append(f"  {names[k]:<30} сработала {v:>3} раз")
        return "\n".join(lines)
