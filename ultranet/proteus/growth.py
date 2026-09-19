"""Путь Протея от 0% до 100%: автоматический рост под железо.

«Протей не обновляется. Он растёт» — по мере использования, а не по расписанию.

Девять стадий из спецификации: 0 (ничего) → 1 (зонд) → 5 (базовая сборка)
→ 10 (знакомство) → 25 (локальный) → 50 (центр) → 75 (полный рой)
→ 90 (персонализация) → 100 (ультимативная форма).

Рост считается не «по расписанию», а по пяти реальным осям:
  железо (что зондировали) · использование (сколько запросов)
  · знания (RAG) · персонализация (LoRA) · рой (сколько узлов).
"""
from __future__ import annotations

import math
import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .catalog import BY_KEY, CATALOG, CapsuleSpec, Group, Stage
from .devices import Device, DeviceState, get_device

GB = 1024 ** 3
MB = 1024 ** 2


# ═══════════════════════════════════════════════════ 1%: зонд железа
@dataclass
class HardwareProfile:
    """Карта возможного: что железо реально потянет.

    Строится капсулой профилирования на 1% — первой из всех.
    """

    name: str
    cpu_count: int
    ram_bytes: int
    ram_available: int
    disk_bytes: int
    has_gpu: bool = False
    battery: bool = True
    tier: str = "medium"
    machine: str = ""
    system: str = ""

    # ---------------------------------------------------------- вычисления
    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / GB

    def weight_budget(self, frac: float = 0.35) -> int:
        """Сколько байт можно отдать весам модели."""
        return int(min(self.ram_available, self.ram_bytes * frac))

    def max_params(self, bits: float = 2.0) -> int:
        """Сколько параметров влезет при 1.58-битном квантовании."""
        return int(self.weight_budget() * 8 / bits)

    def tokens_per_sec(self) -> float:
        """Грубая оценка скорости: ядра × частота-эквивалент."""
        return max(1.0, self.cpu_count * 8.0 * (2.0 if self.has_gpu else 1.0))

    def context_window(self) -> int:
        """Какой контекст потянет память."""
        for ctx in (128_000, 32_000, 8_000, 2_000, 512):
            if self.ram_available > ctx * 2048:
                return ctx
        return 256

    def as_device(self) -> Device:
        """Превратить зонд в Device — дальше работает обычный конвейер."""
        return Device(self.name, self.tier, self.ram_bytes, 0.35,
                      self.cpu_count * (4.0 if self.has_gpu else 0.5),
                      self.battery, False)

    def report(self) -> str:
        return (f"железо     : {self.name} [{self.tier}]\n"
                f"CPU        : {self.cpu_count} ядер"
                + (", GPU есть" if self.has_gpu else ", GPU нет") + "\n"
                f"RAM        : {self.ram_gb:.1f} ГБ "
                f"(свободно {self.ram_available / GB:.1f} ГБ)\n"
                f"диск       : {self.disk_bytes / GB:.0f} ГБ\n"
                f"бюджет     : {self.weight_budget() / MB:.0f} МБ весов "
                f"= до {self.max_params() / 1e6:.0f}M параметров @1.58-bit\n"
                f"скорость   : ~{self.tokens_per_sec():.0f} ток/с, "
                f"контекст {self.context_window():,}")


def probe_hardware(simulate: Optional[str] = None) -> HardwareProfile:
    """Зонд железа. Реальный — или симуляция известного устройства.

    Это первое, что делает Протей на 1%: «Анализирую устройство».
    """
    if simulate is not None:
        d = get_device(simulate)
        return HardwareProfile(
            name=d.name, cpu_count=max(1, int(d.gflops) or 1),
            ram_bytes=d.ram_bytes, ram_available=d.weight_budget,
            disk_bytes=d.ram_bytes * 8, has_gpu=d.gflops >= 8.0,
            battery=d.battery, tier=d.tier, machine="simulated",
            system="simulated")

    cpu = os.cpu_count() or 1
    ram = _total_ram()
    avail = _available_ram(ram)
    try:
        disk = shutil.disk_usage("/").total
    except OSError:
        disk = ram * 4
    tier = _tier_for(ram)
    return HardwareProfile(
        name=platform.node() or "это устройство", cpu_count=cpu,
        ram_bytes=ram, ram_available=avail, disk_bytes=disk,
        has_gpu=_detect_gpu(), battery=_detect_battery(), tier=tier,
        machine=platform.machine(), system=platform.system())


def _total_ram() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 2 * GB


def _available_ram(total: int) -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(total * 0.5)


def _detect_gpu() -> bool:
    if shutil.which("nvidia-smi"):
        return True
    return any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/dri/card0"))


def _detect_battery() -> bool:
    return os.path.isdir("/sys/class/power_supply/BAT0")


def _device_by_name(name: str) -> Optional[Device]:
    """Найти Device по человеческому имени (узлы роя хранятся именами)."""
    from .devices import DEVICES
    for d in DEVICES.values():
        if d.name == name:
            return d
    return None


def _tier_for(ram: int) -> str:
    if ram < 1 * MB:
        return "micro"
    if ram < 2 * GB:
        return "small"
    if ram < 64 * GB:
        return "medium"
    if ram < 256 * GB:
        return "large"
    return "datacenter"


# ═══════════════════════════════════════════════════ шкала стадий
@dataclass(frozen=True)
class Stage_:
    """Одна ступень пути 0→100%."""

    pct: int
    title: str
    capsules: int
    params: int
    memory: str
    can_do: str
    user_says: str
    hardware: str


LADDER: Tuple[Stage_, ...] = (
    Stage_(0, "Абсолютный ноль", 0, 0, "0",
           "ничего: нет модели, нет весов, нет капсул", "«Что это?»",
           "голое железо"),
    Stage_(1, "Первый байт", 1, 100_000, "1 МБ",
           "зондирует железо, строит карту возможного", "«Что это?»",
           "одно устройство, минимум ресурсов"),
    Stage_(5, "Базовая сборка", 7, 100_000_000, "50 МБ",
           "простые вопросы, короткие ответы, один язык", "«Полезно»",
           "одно устройство, минимум ресурсов"),
    Stage_(10, "Первое знакомство", 12, 150_000_000, "80 МБ",
           "простые задачи, слабая персонализация, 1–2 языка", "«Полезно»",
           "одно устройство, средние ресурсы"),
    Stage_(25, "Локальный Протей", 25, 1_000_000_000, "600 МБ",
           "сложные задачи, персонализация, память, 5–10 языков", "«Я привык»",
           "одно устройство, средние ресурсы"),
    Stage_(50, "Центр цифровой жизни", 35, 3_000_000_000, "2 ГБ",
           "рой из 2 узлов, эпизодическая память, 20+ языков",
           "«Я не могу без него»", "два устройства, рой"),
    Stage_(75, "Полный рой", 45, 7_000_000_000, "5 ГБ",
           "6 узлов, все экспертизы, рефлексия, 50+ языков", "«Он знает меня»",
           "4–6 устройств, полный рой"),
    Stage_(90, "Ультимативная персонализация", 48, 14_000_000_000, "10 ГБ",
           "предсказывает вопросы, 100+ языков, все модальности",
           "«Он — часть меня»", "все устройства, включая IoT"),
    Stage_(100, "Ультимативная форма", 48, 0, "всё железо",
           "предел железа: слияние с пользователем", "«Он — часть меня»",
           "все устройства + внешние GPU + кооперация"),
)


def stage_at(pct: float) -> Stage_:
    """Ближайшая достигнутая ступень."""
    reached = [s for s in LADDER if s.pct <= pct]
    return reached[-1] if reached else LADDER[0]


def next_stage(pct: float) -> Optional[Stage_]:
    ahead = [s for s in LADDER if s.pct > pct]
    return ahead[0] if ahead else None


# ═══════════════════════════════════════════════════ счётчик роста
@dataclass
class GrowthState:
    """Что Протей накопил. Из этого считается процент."""

    probed: bool = False
    installed: bool = False
    requests: int = 0
    documents: int = 0
    adapters: int = 0
    vectors: int = 0
    nodes: int = 1
    reflections: int = 0
    feedback: int = 0

    #: цели «100%» по каждой оси — из спецификации
    TARGET_REQUESTS = 2000
    TARGET_DOCS = 1_000_000
    TARGET_ADAPTERS = 200
    TARGET_VECTORS = 500
    TARGET_NODES = 6

    @staticmethod
    def _log_frac(value: float, target: float) -> float:
        """Логарифмический прогресс: первые шаги дают больше, чем сотые."""
        if value <= 0:
            return 0.0
        return float(np.clip(math.log1p(value) / math.log1p(target), 0.0, 1.0))

    # ------------------------------------------------------------- оси
    def axes(self) -> Dict[str, float]:
        return {
            "железо": 1.0 if self.probed else 0.0,
            "использование": self._log_frac(self.requests, self.TARGET_REQUESTS),
            "знания": self._log_frac(self.documents, self.TARGET_DOCS),
            "персонализация": max(
                self._log_frac(self.adapters, self.TARGET_ADAPTERS),
                self._log_frac(self.vectors, self.TARGET_VECTORS)),
            "рой": float(np.clip((self.nodes - 1) / (self.TARGET_NODES - 1), 0.0, 1.0)),
        }

    def percent(self) -> float:
        """Итоговый процент роста 0..100."""
        if not self.probed:
            return 0.0
        if not self.installed:
            return 1.0
        a = self.axes()
        # железо — предусловие, а не ось роста: зонд уже учтён в базовых 5%.
        # Растят Протея использование, знания, персонализация и рой.
        w = {"использование": 0.32, "знания": 0.26,
             "персонализация": 0.26, "рой": 0.16}
        score = sum(a[k] * w[k] for k in w)
        return float(np.clip(5.0 + score * 95.0, 5.0, 100.0))


# ═══════════════════════════════════════════════════ живой организм
class GrowingProteus:
    """Протей, который растёт сам: зондирует железо и разворачивается под него.

    >>> p = GrowingProteus.install(simulate="phone")
    >>> p.percent > 0
    True
    >>> p.use("напиши код", documents=10)
    >>> p.add_node("laptop")
    """

    def __init__(self, profile: Optional[HardwareProfile] = None) -> None:
        self.profile = profile
        self.state = GrowthState(probed=profile is not None)
        self.nodes: List[str] = []
        self.history: List[Tuple[float, str]] = []
        self._log(0.0, "Протея нет: голое железо")

    # ------------------------------------------------------------- 0 → 1%
    @classmethod
    def bare(cls) -> "GrowingProteus":
        """0%: ничего нет."""
        return cls(None)

    def probe(self, simulate: Optional[str] = None) -> HardwareProfile:
        """1%: капсула профилирования просыпается первой."""
        self.profile = probe_hardware(simulate)
        self.state.probed = True
        self._log(self.percent, f"зонд: {self.profile.name}, "
                                f"{self.profile.ram_gb:.1f} ГБ, "
                                f"до {self.profile.max_params() / 1e6:.0f}M параметров")
        return self.profile

    # ------------------------------------------------------------- 1 → 5%
    def install(self) -> "GrowingProteus":
        """5%: мета-контроллер собирает минимальную конфигурацию."""
        if self.profile is None:
            self.probe()
        self.state.installed = True
        self.nodes = [self.profile.name]
        self.state.nodes = 1
        self._log(self.percent, f"базовая сборка: {len(self.active_capsules())} капсул")
        return self

    @classmethod
    def install_on(cls, simulate: Optional[str] = None) -> "GrowingProteus":
        p = cls.bare()
        p.probe(simulate)
        return p.install()

    # --------------------------------------------------------------- рост
    def use(self, prompt: str = "", documents: int = 0, liked: bool = False,
            requests: int = 1) -> None:
        """«Каждый запрос → адаптеры, каждый документ → RAG»."""
        if not self.state.installed:
            self.install()
        before = self.percent
        self.state.requests += max(0, requests)
        self.state.documents += max(0, documents)
        if liked:
            self.state.feedback += 1
        # адаптеры и векторы копятся по мере использования
        self.state.adapters = min(GrowthState.TARGET_ADAPTERS,
                                  1 + self.state.requests // 10)
        self.state.vectors = min(GrowthState.TARGET_VECTORS,
                                 self.state.requests // 4)
        self.state.reflections = self.state.requests // 25
        after = self.percent
        if int(after) > int(before):
            self._log(after, f"вырос до {after:.0f}% — {self.stage.title}")

    def add_node(self, device_key: str) -> None:
        """«Каждое новое устройство → расширение роя»."""
        if not self.state.installed:
            self.install()
        name = get_device(device_key).name
        if name in self.nodes:
            return
        self.nodes.append(name)
        self.state.nodes = len(self.nodes)
        self._log(self.percent, f"в рой пришёл {name} ({self.state.nodes} узлов)")

    def remove_node(self, name: str) -> None:
        if name in self.nodes and len(self.nodes) > 1:
            self.nodes.remove(name)
            self.state.nodes = len(self.nodes)
            self._log(self.percent, f"узел {name} ушёл ({self.state.nodes})")

    # -------------------------------------------------------------- статус
    @property
    def percent(self) -> float:
        """Процент роста, честно ограниченный возможностями железа.

        ESP32 с четырьмя капсулами не может называться «полным роем»:
        потолок задаётся тем, сколько капсул железо реально тянет.
        """
        return min(self.state.percent(), self.ceiling())

    def ceiling(self) -> float:
        """Максимум, достижимый на этом железе (и текущем рое)."""
        if self.profile is None:
            return 0.0
        dev = self.host
        fits = [c for c in CATALOG if c.fits_on(dev)]
        if not fits:
            return 1.0
        # сколько капсул влезает в бюджет памяти роя при полной зрелости
        budget = self.total_budget()
        order = sorted(fits, key=lambda c: (not c.always_on, c.params_for(dev)))
        used = n = 0
        for c in order:
            b = c.bytes_for(dev)
            if used + b > budget:
                break
            used += b
            n += 1
        if n >= len(CATALOG):
            return 100.0
        pts = [(0, 0), (1, 1), (5, 7), (10, 12), (25, 25),
               (50, 35), (75, 45), (90, 48)]
        caps = [c for _, c in pts]
        pcts = [p for p, _ in pts]
        return float(np.clip(np.interp(n, caps, pcts), 1.0, 100.0))

    @property
    def stage(self) -> Stage_:
        return stage_at(self.percent)

    def active_capsules(self) -> List[CapsuleSpec]:
        """Какие капсулы развёрнуты на текущем проценте роста.

        Порядок пробуждения повторяет спецификацию: сначала зонд, затем
        базовая семёрка, дальше — по мере роста, от дешёвых к дорогим.
        """
        if not self.state.probed:
            return []
        if not self.state.installed:
            return [BY_KEY["profiler"]]

        pct = self.percent
        target = self._target_count(pct)
        dev = self.host
        fits = [c for c in CATALOG if c.fits_on(dev)]

        # приоритет пробуждения
        maturity = self.maturity()
        base = ["profiler", "meta", "text_in", "meaning", "intent",
                "working_mem", "gen_text", "critic", "safety"]
        order: List[str] = [k for k in base if BY_KEY[k] in fits]
        rest = sorted([c for c in fits if c.key not in order],
                      key=lambda c: (not c.always_on, c.params_for(dev)))
        order += [c.key for c in rest]
        wanted = [BY_KEY[k] for k in order[:target]]

        # Жёсткая гарантия: сумма капсул не превышает бюджет памяти роя.
        # Протей растёт под железо, а не поверх него.
        budget = self.total_budget()
        out: List[CapsuleSpec] = []
        used = 0
        for c in wanted:
            b = self.capsule_bytes(c)
            if used + b > budget:
                continue
            out.append(c)
            used += b
        return out

    def maturity(self) -> float:
        """Зрелость 0..1: молодой Протей берёт нижнюю границу размеров.

        «Базовый размер — 100M» на 5% и «максимум, который тянет железо»
        на 100%. Растёт не только число капсул, но и каждая капсула.
        """
        return float(np.clip((self.percent - 5.0) / 95.0, 0.0, 1.0))

    def capsule_params(self, spec: CapsuleSpec) -> int:
        """Размер капсулы с учётом и железа, и текущей зрелости Протея."""
        ceiling = spec.params_for(self.host)
        if ceiling <= 0:
            return 0
        lo = float(spec.min_params)
        hi = float(max(ceiling, spec.min_params))
        return int(round(lo * (hi / lo) ** self.maturity()))

    def capsule_bytes(self, spec: CapsuleSpec, bits: float = 2.0) -> int:
        return int(self.capsule_params(spec) * bits / 8)

    def total_budget(self) -> int:
        """Бюджет памяти всего роя: каждый узел приносит свою долю."""
        if self.profile is None:
            return 0
        own = self.profile.weight_budget()
        extra = 0
        for name in self.nodes[1:]:
            dev = _device_by_name(name)
            if dev is not None:
                extra += dev.weight_budget
        return own + extra

    def _target_count(self, pct: float) -> int:
        """Сколько капсул должно быть живо на этом проценте (шкала спеки)."""
        pts = [(0, 0), (1, 1), (5, 7), (10, 12), (25, 25),
               (50, 35), (75, 45), (90, 48), (100, 48)]
        xs = [p for p, _ in pts]
        ys = [c for _, c in pts]
        n = int(round(float(np.interp(pct, xs, ys))))
        return min(n, len([c for c in CATALOG if c.fits_on(self.host)]))

    @property
    def device(self) -> Device:
        """Своё железо."""
        if self.profile is None:
            return get_device("phone")
        return self.profile.as_device()

    @property
    def host(self) -> Device:
        """Самый мощный узел роя — он определяет, что Протею вообще доступно.

        Часы в паре с ноутбуком умеют то же, что ноутбук: тяжёлые капсулы
        просто живут на сильном узле. Это и есть «рой как одно тело».
        """
        best = self.device
        for name in self.nodes[1:]:
            d = _device_by_name(name)
            if d is not None and d.ram_bytes > best.ram_bytes:
                best = d
        return best

    def active_params(self) -> int:
        return sum(self.capsule_params(c) for c in self.active_capsules())

    def memory_bytes(self, bits: float = 2.0) -> int:
        return int(self.active_params() * bits / 8)

    def fits_hardware(self) -> bool:
        """Главная гарантия: Протей никогда не перерастает своё железо."""
        if self.profile is None:
            return True
        return self.memory_bytes() <= self.total_budget()

    # -------------------------------------------------------------- отчёты
    def _log(self, pct: float, what: str) -> None:
        self.history.append((pct, what))

    def status(self) -> str:
        s = self.stage
        nxt = next_stage(self.percent)
        lines = [
            f"Протей {self.percent:.0f}% — {s.title}",
            f"  капсул    : {len(self.active_capsules())} из 48",
            f"  параметров: {self.active_params() / 1e6:.0f}M",
            f"  память    : {self.memory_bytes() / MB:.0f} МБ"
            + (f" из {self.total_budget() / MB:.0f} МБ бюджета роя"
               if self.profile else ""),
            f"  узлов роя : {self.state.nodes} ({', '.join(self.nodes) or '—'})",
            f"  умеет     : {s.can_do}",
            f"  железо    : {s.hardware}",
            f"  влезает   : {'да' if self.fits_hardware() else 'НЕТ'}",
        ]
        if nxt:
            lines.append(f"  следующая : {nxt.pct}% — {nxt.title}")
        return "\n".join(lines)

    def axes_bar(self, width: int = 22) -> str:
        out = []
        for name, val in self.state.axes().items():
            filled = int(round(val * width))
            out.append(f"  {name:<16}[{'█' * filled}{'·' * (width - filled)}] "
                       f"{val * 100:>3.0f}%")
        return "\n".join(out)

    def timeline(self) -> str:
        return "\n".join(f"  {p:>5.0f}%  {w}" for p, w in self.history)


def ladder_table() -> str:
    """Сводная таблица пути 0→100% — как в спецификации."""
    rows = [f"{'%':>4}  {'капсул':>7}{'параметров':>13}{'память':>11}  что умеет"]
    for s in LADDER:
        p = "максимум" if s.pct == 100 else (
            f"{s.params / 1e9:.0f}B" if s.params >= 1e9 else
            (f"{s.params / 1e6:.0f}M" if s.params >= 1e6 else
             (f"{s.params / 1e3:.0f}K" if s.params else "0")))
        rows.append(f"{s.pct:>3}%  {s.capsules:>7}{p:>13}{s.memory:>11}  {s.can_do[:44]}")
    return "\n".join(rows)
