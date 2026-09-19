"""Наночастицы (капсулы): минимальные единицы Протея.

«Капсула — функциональный блок, который знает про себя всё: сколько ест,
что умеет, как сжимается, как взаимодействует с другими».

Уровни вложенности из спецификации: атом -> молекула -> клетка -> орган -> тело.
Неактивные капсулы «не существуют» — они не занимают ни памяти, ни энергии,
пока не собраны (lazy materialization).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np


class Level(IntEnum):
    """Уровни вложенности наночастиц."""

    ATOM = 0        # один нейрон / строка матрицы
    MOLECULE = 1    # группа нейронов
    CELL = 2        # слой
    ORGAN = 3       # группа слоёв (эксперт, блок внимания)
    BODY = 4        # весь Протей

    @property
    def label(self) -> str:
        return {0: "атом", 1: "молекула", 2: "клетка", 3: "орган", 4: "тело"}[int(self)]


@dataclass
class Capsule:
    """Наночастица: веса + метаданные + интерфейс + состояние."""

    name: str
    level: Level
    n_params: int
    skills: Set[str] = field(default_factory=set)
    children: List["Capsule"] = field(default_factory=list)
    materialize: Optional[Callable[[], object]] = None
    energy_per_call: float = 1.0
    latency_ms: float = 1.0

    # состояние
    _awake: bool = field(default=False, init=False)
    _payload: object = field(default=None, init=False)
    calls: int = field(default=0, init=False)

    # -------------------------------------------------------------- жизнь
    @property
    def awake(self) -> bool:
        return self._awake

    def wake(self, recursive: bool = False) -> object:
        """Пробуждение: капсула собирается на месте из своих компонентов.

        recursive=False — просыпается только сама капсула; её дети решают
        сами (иначе пробуждение органа тянуло бы всех спящих экспертов).
        """
        if not self._awake:
            self._payload = self.materialize() if self.materialize else None
            self._awake = True
        if recursive:
            for c in self.children:
                c.wake(recursive=True)
        self.calls += 1
        return self._payload

    def sleep(self) -> None:
        """Сон: капсула перестаёт существовать в памяти."""
        self._payload = None
        self._awake = False
        for c in self.children:
            c.sleep()

    # -------------------------------------------------------------- размеры
    def bytes_at(self, bits: float = 2.0) -> int:
        """Сколько весит капсула при заданной битности (по умолчанию 1.58-bit)."""
        return int(self.total_params() * bits / 8)

    def total_params(self) -> int:
        return self.n_params + sum(c.total_params() for c in self.children)

    def active_params(self) -> int:
        """Только то, что реально проснулось (по всему поддереву)."""
        own = self.n_params if self._awake else 0
        return own + sum(c.active_params() for c in self.children)

    def all_skills(self) -> Set[str]:
        s = set(self.skills)
        for c in self.children:
            s |= c.all_skills()
        return s

    def matches(self, need: Sequence[str]) -> float:
        """Насколько капсула подходит под требуемые навыки (0..1)."""
        if not need:
            return 0.0
        have = self.all_skills()
        return len(have & set(need)) / len(set(need))

    def tree(self, indent: int = 0) -> str:
        mark = "●" if self._awake else "○"
        pad = "  " * indent
        line = (f"{pad}{mark} {self.name:<22} {self.level.label:<9} "
                f"{self.total_params():>9,} пар. {self.bytes_at() / 1024:>8.1f} КБ")
        if self.skills:
            line += f"  [{', '.join(sorted(self.skills))}]"
        return "\n".join([line] + [c.tree(indent + 1) for c in self.children])

    def __repr__(self) -> str:
        return f"Capsule({self.name}, {self.level.label}, {self.total_params():,})"


class CapsuleRegistry:
    """Реестр капсул: сборка тела под задачу.

    «Что не нужно — не активируется. Атомы "поэзия" спят».

    >>> reg = CapsuleRegistry.from_model(model)
    >>> reg.assemble(["код"])            # проснутся только кодовые капсулы
    >>> reg.active_fraction() < 1.0
    True
    """

    def __init__(self, root: Capsule) -> None:
        self.root = root
        self.index: Dict[str, Capsule] = {}
        self._index(root)

    def _index(self, c: Capsule) -> None:
        self.index[c.name] = c
        for ch in c.children:
            self._index(ch)

    # ------------------------------------------------------------- сборка
    def assemble(self, need: Sequence[str], threshold: float = 0.01) -> List[Capsule]:
        """Разбудить капсулы, подходящие под задачу; остальные спят."""
        self.sleep_all()
        woken: List[Capsule] = []
        need_set = set(need)
        for cap in self.index.values():
            if cap.level is Level.BODY:
                continue
            # капсула нужна, если её СОБСТВЕННЫЕ навыки пересекаются с задачей
            # либо это инфраструктура ядра (внимание, эмбеддинги)
            own_match = bool(cap.skills & need_set)
            is_core = "ядро" in cap.skills and not (cap.skills - {"ядро", "байты", "контекст"})
            if own_match or is_core:
                cap.wake()
                woken.append(cap)

        # фолбэк: если ни одна специализированная капсула не подошла,
        # будим все — незнакомая задача должна решаться полным телом,
        # а не голым ядром.
        specialized = [c for c in woken if not ("ядро" in c.skills)]
        if not specialized:
            for cap in self.index.values():
                if cap.level is not Level.BODY and not cap.awake:
                    cap.wake()
                    woken.append(cap)
        return woken

    def sleep_all(self) -> None:
        self.root.sleep()

    def wake_all(self) -> None:
        self.root.wake(recursive=True)

    # -------------------------------------------------------------- отчёты
    def active_params(self) -> int:
        return self.root.active_params()

    def total_params(self) -> int:
        return self.root.total_params()

    def active_fraction(self) -> float:
        return self.active_params() / max(self.total_params(), 1)

    def by_level(self, level: Level) -> List[Capsule]:
        return [c for c in self.index.values() if c.level == level]

    def report(self) -> str:
        lines = [f"капсул всего: {len(self.index)}, "
                 f"активно: {sum(1 for c in self.index.values() if c.awake)}"]
        for lvl in Level:
            caps = self.by_level(lvl)
            if caps:
                awake = sum(1 for c in caps if c.awake)
                lines.append(f"  {lvl.label:<9} {len(caps):>3} шт, активно {awake:>3}")
        lines.append(f"активная доля параметров: {self.active_fraction() * 100:.1f}%")
        return "\n".join(lines)

    # --------------------------------------------------------- из модели
    @classmethod
    def from_model(cls, model, skills_by_expert: Optional[Dict[int, Set[str]]] = None
                   ) -> "CapsuleRegistry":
        """Построить дерево капсул по реальной структуре MatFormer."""
        cfg = model.cfg
        d = cfg.n_embd
        body = Capsule("Протей", Level.BODY, 0, {"ядро"})

        emb = Capsule("эмбеддинги", Level.ORGAN, cfg.vocab_size * d, {"ядро", "байты"})
        body.children.append(emb)

        for li, blk in enumerate(model.blocks):
            organ = Capsule(f"слой{li}", Level.ORGAN, 0, {"ядро"})
            attn = Capsule(f"слой{li}.внимание", Level.CELL, 4 * d * d,
                           {"ядро", "контекст"})
            organ.children.append(attn)
            if getattr(blk, "moe", None) is not None:
                for ei, ex in enumerate(blk.moe.experts):
                    sk = (skills_by_expert or {}).get(ei, {f"эксперт{ei}"})
                    cell = Capsule(f"слой{li}.эксперт{ei}", Level.CELL,
                                   ex.active_params(d), set(sk))
                    # молекулы внутри эксперта
                    for gi in range(2):
                        cell.children.append(
                            Capsule(f"слой{li}.эксперт{ei}.мол{gi}", Level.MOLECULE,
                                    ex.active_params(d) // 8, set(sk))
                        )
                    organ.children.append(cell)
            else:
                organ.children.append(
                    Capsule(f"слой{li}.mlp", Level.CELL, 3 * d * blk.mlp_dim, {"ядро"})
                )
            body.children.append(organ)
        return cls(body)
