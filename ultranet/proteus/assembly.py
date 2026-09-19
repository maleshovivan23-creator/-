"""Сборка Протея из 48 капсул под конкретный запрос и конкретное железо.

«Что не нужно — не активируется». Движок делает три вещи:
  1. Понимает, какие капсулы нужны запросу (триггеры + зависимости).
  2. Отсекает те, что не влезают в железо.
  3. Выстраивает их в конвейер: вход -> понимание -> память -> мышление
     -> экспертиза -> генерация -> проверка, плюс фоновые.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .capsules import Capsule, CapsuleRegistry, Level
from .catalog import (BY_KEY, CATALOG, GROUP_COUNTS, CapsuleSpec, Group, Stage,
                      resolve_deps, tier_rank)
from .devices import Device, DeviceState, get_device

#: слова запроса -> триггеры капсул
_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "код": ("код", "функци", "程序", "def ", "class ", "багу", "баг", "рефактор",
            "скрипт", "программ", "python", "тест"),
    "математика": ("посчитай", "сколько будет", "уравнен", "реши", "интеграл",
                   "производн", "формул", "процент"),
    "творчество": ("стих", "сочини", "придумай", "поэм", "рассказ", "истори",
                   "песн", "идея", "идеи"),
    "рассуждение": ("почему", "объясни", "проанализируй", "разбер", "сравни",
                    "докажи", "как работает"),
    "план": ("план", "распиши шаги", "организуй", "спланируй", "расписание"),
    "перевод": ("переведи", "перевод", "как будет по-", "translate", "на японск",
                "на английск", "на русск"),
    "медицина": ("симптом", "болит", "боль", "лекарств", "диагноз", "анализ крови",
                 "давлени", "температура тела", "врач"),
    "право": ("договор", "закон", "юрид", "иск", "штраф", "суд"),
    "финансы": ("расход", "инвест", "налог", "бюджет", "зарплат", "кредит"),
    "наука": ("стать", "гипотез", "исследован", "эксперимент", "публикац",
              "научн"),
    "изображение": ("нарисуй", "картинк", "изображен", "фото", "видео", "сцен"),
    "голос": ("скажи вслух", "озвучь", "голосом", "произнеси"),
    "аудио": ("послушай", "запись", "звук", "шум"),
    "действие": ("открой", "закрой", "отправь", "включи", "выключи", "купи",
                 "позвони", "напомни"),
    "жест": ("жест", "покажи руками", "язык жестов"),
    "сенсор": ("пульс", "шаги", "сон", "температур", "давлени"),
    "формула": ("формула", "ноты", "днк", "химическ", "шахмат"),
}


def detect_triggers(prompt: str) -> Set[str]:
    """Какие триггеры зажигает запрос."""
    p = prompt.lower()
    found = {t for t, words in _KEYWORDS.items() if any(w in p for w in words)}
    found.add("текст")                      # текст на входе есть всегда
    return found


@dataclass
class Assembly:
    """Собранный Протей: какие капсулы проснулись и во что это обошлось."""

    prompt: str
    device: Device
    active: List[CapsuleSpec]
    skipped: List[CapsuleSpec]
    unavailable: List[CapsuleSpec]
    triggers: Set[str]
    params: int
    bytes_used: int
    budget: int

    # --------------------------------------------------------------- доли
    @property
    def n_active(self) -> int:
        return len(self.active)

    @property
    def capsule_fraction(self) -> float:
        return self.n_active / len(CATALOG)

    @property
    def foreground(self) -> List[CapsuleSpec]:
        """Капсулы конвейера — без фоновых служб, которые работают всегда."""
        return [c for c in self.active if c.stage is not Stage.BACKGROUND]

    @property
    def foreground_fraction(self) -> float:
        total = len([c for c in CATALOG if c.stage is not Stage.BACKGROUND])
        return len(self.foreground) / max(total, 1)

    def heavy_active(self, threshold: int = 50_000_000) -> List[CapsuleSpec]:
        """Тяжёлые капсулы (>50M) — именно они определяют реальную цену."""
        return [c for c in self.active
                if c.params_for(self.device.tier) >= threshold]

    @property
    def fits(self) -> bool:
        return self.bytes_used <= self.budget

    @property
    def degraded(self) -> List[CapsuleSpec]:
        """Капсулы, которые железо потянуло бы, но не хватило бюджета памяти."""
        active = {c.key for c in self.active}
        unavail = {c.key for c in self.unavailable}
        return [c for c in self.skipped
                if c.key not in active and c.key not in unavail
                and (c.always_on or c.triggers & self.triggers)]

    def energy_fraction(self) -> float:
        """Грубая доля энергии: активные параметры от полного тела."""
        total = sum(c.params_for(self.device.tier) for c in CATALOG
                    if c.fits_on(self.device))
        return self.params / max(total, 1)

    # ------------------------------------------------------------ конвейер
    def pipeline(self) -> List[Tuple[Stage, List[CapsuleSpec]]]:
        """Активные капсулы, выстроенные по стадиям обработки."""
        out: List[Tuple[Stage, List[CapsuleSpec]]] = []
        for st in Stage:
            caps = sorted([c for c in self.active if c.stage is st], key=lambda c: c.key)
            if caps:
                out.append((st, caps))
        return out

    def groups_used(self) -> Dict[Group, int]:
        d: Dict[Group, int] = {}
        for c in self.active:
            d[c.group] = d.get(c.group, 0) + 1
        return d

    def names(self) -> List[str]:
        return [c.name for c in self.active]

    # -------------------------------------------------------------- отчёты
    def explain(self) -> str:
        stage_names = {
            Stage.INPUT: "вход", Stage.PARSE: "понимание", Stage.RECALL: "память",
            Stage.REASON: "мышление", Stage.EXPERT: "экспертиза",
            Stage.PRODUCE: "генерация", Stage.VERIFY: "проверка",
            Stage.BACKGROUND: "фон",
        }
        lines = [f"запрос: «{self.prompt}»",
                 f"железо: {self.device}",
                 f"триггеры: {', '.join(sorted(self.triggers))}",
                 ""]
        n = 0
        for st, caps in self.pipeline():
            lines.append(f"  [{stage_names[st]}]")
            for c in caps:
                n += 1
                p = c.params_for(self.device.tier)
                lines.append(f"    {n:>2}. {c.name:<26} {p / 1e6:>8.1f}M  {c.does[:44]}")
        lines.append("")
        n_heavy = len(self.heavy_active())
        total_heavy = len([c for c in CATALOG
                           if c.params_for(self.device.tier) >= 50_000_000])
        lines.append(f"активно {self.n_active} из {len(CATALOG)} капсул "
                     f"({self.capsule_fraction * 100:.0f}%), из них в конвейере "
                     f"{len(self.foreground)} ({self.foreground_fraction * 100:.0f}%)")
        lines.append(f"тяжёлых (>50M) задействовано {n_heavy} из {total_heavy}; "
                     f"{self.params / 1e6:.0f}M параметров, "
                     f"{self.bytes_used / 1024 / 1024:.1f} МБ "
                     f"из бюджета {self.budget / 1024 / 1024:.1f} МБ")
        if self.skipped:
            names = ", ".join(c.name for c in self.skipped[:6])
            lines.append(f"не активировалось ({len(self.skipped)}): {names}…")
        if self.unavailable:
            lines.append(f"не помещается на это железо: {len(self.unavailable)} капсул")
        if self.degraded:
            names = ", ".join(c.name for c in self.degraded[:5])
            lines.append(f"урезано по памяти ({len(self.degraded)}): {names}…")
        return "\n".join(lines)

    def __str__(self) -> str:
        return (f"Assembly({self.n_active}/{len(CATALOG)} капсул, "
                f"{self.params / 1e6:.0f}M, {'влезает' if self.fits else 'НЕ влезает'})")


class ProteusBody:
    """Полное тело Протея: все 48 капсул и правила их сборки.

    >>> body = ProteusBody("phone")
    >>> a = body.assemble("напиши код сортировки")
    >>> "Кодовая экспертная" in a.names()
    True
    """

    def __init__(self, device: str | Device = "phone",
                 state: Optional[DeviceState] = None) -> None:
        self.device = get_device(device) if isinstance(device, str) else device
        self.state = state or DeviceState(self.device)

    # ------------------------------------------------------------ выбор
    def available(self) -> List[CapsuleSpec]:
        """Капсулы, которые физически могут жить на этом железе."""
        return [c for c in CATALOG if c.fits_on(self.device)]

    def _needed_keys(self, triggers: Set[str], want: Sequence[str] = ()) -> Set[str]:
        keys: Set[str] = set(want)
        for c in CATALOG:
            if c.always_on:
                keys.add(c.key)
            if c.triggers & triggers:
                keys.add(c.key)
        # текстовый ответ нужен почти всегда
        keys.add("gen_text")
        return resolve_deps(keys)

    def assemble(self, prompt: str, want: Sequence[str] = (),
                 respect_budget: bool = True) -> Assembly:
        """Собрать тело под запрос: триггеры -> зависимости -> бюджет."""
        triggers = detect_triggers(prompt)
        needed = self._needed_keys(triggers, want)

        avail = {c.key for c in self.available()}
        unavailable = [BY_KEY[k] for k in sorted(needed - avail)]
        chosen = [BY_KEY[k] for k in sorted(needed & avail)]

        tier = self.device.tier
        budget = self.state.available_bytes
        # приоритет: всегда включённые и ранние стадии выживают первыми
        chosen.sort(key=lambda c: (not c.always_on, int(c.stage), -c.params_for(tier)))

        # Бюджет обязателен для всех, включая always_on: на смарт-карте с 16 КБ
        # физически не может жить 26 МБ фоновых служб. Дешёвые капсулы идут
        # первыми, чтобы при жёстком лимите выжило максимум функций.
        chosen.sort(key=lambda c: (not c.always_on, int(c.stage), c.params_for(tier)))
        active: List[CapsuleSpec] = []
        used = 0
        for c in chosen:
            b = c.bytes_for(tier)
            if respect_budget and used + b > budget:
                continue
            active.append(c)
            used += b

        active_keys = {c.key for c in active}
        skipped = [c for c in CATALOG
                   if c.key not in active_keys and c.key not in {x.key for x in unavailable}]
        params = sum(c.params_for(tier) for c in active)
        return Assembly(prompt, self.device, active, skipped, unavailable,
                        triggers, params, used, budget)

    # --------------------------------------------------------- как дерево
    def to_registry(self, assembly: Optional[Assembly] = None) -> CapsuleRegistry:
        """Представить тело деревом капсул (орган = группа, клетка = капсула)."""
        tier = self.device.tier
        active_keys = {c.key for c in assembly.active} if assembly else set()
        root = Capsule("Протей", Level.BODY, 0, {"ядро"})
        for g in Group:
            organ = Capsule(g.value, Level.ORGAN, 0, {"ядро"})
            for spec in [c for c in CATALOG if c.group is g]:
                cell = Capsule(spec.name, Level.CELL, spec.params_for(tier),
                               set(spec.skills))
                if spec.key in active_keys:
                    cell.wake()
                organ.children.append(cell)
            root.children.append(organ)
        return CapsuleRegistry(root)

    # -------------------------------------------------------------- отчёты
    def inventory(self) -> str:
        """Сводная таблица каталога — как в спецификации."""
        tier = self.device.tier
        lines = [f"Каталог капсул Протея на «{self.device.name}» [{tier}]",
                 f"{'группа':<14}{'капсул':>7}{'влезает':>9}{'параметров':>14}"]
        total = fit_total = 0
        for g in Group:
            caps = [c for c in CATALOG if c.group is g]
            fit = [c for c in caps if c.fits_on(self.device)]
            p = sum(c.params_for(tier) for c in fit)
            total += len(caps)
            fit_total += len(fit)
            lines.append(f"{g.value:<14}{len(caps):>7}{len(fit):>9}{p / 1e6:>12.0f}M")
        allp = sum(c.params_for(tier) for c in CATALOG if c.fits_on(self.device))
        lines.append(f"{'ВСЕГО':<14}{total:>7}{fit_total:>9}{allp / 1e6:>12.0f}M")
        return "\n".join(lines)

    def device_capabilities(self) -> str:
        """Какие группы капсул доступны на разных классах железа."""
        rows = ["устройство          micro→ сенс пони памя мышл эксп гене упра адап сист  всего"]
        for key in ("smartcard", "esp32", "earbuds", "smartwatch", "phone",
                    "laptop", "workstation", "server"):
            dev = get_device(key)
            cells = []
            for g in Group:
                n = sum(1 for c in CATALOG if c.group is g and c.fits_on(dev))
                cells.append(f"{n:>4}")
            tot = sum(1 for c in CATALOG if c.fits_on(dev))
            rows.append(f"{dev.name:<20}{dev.tier:<7}{''.join(cells)}{tot:>7}")
        return "\n".join(rows)
