"""Каталог всех 48 капсул Протея.

Девять групп: сенсорные, понимания, памяти, мышления, экспертные,
генерации, управления, адаптации, системные.

Каждая капсула знает про себя всё: сколько ест (min/max параметров),
что умеет (навыки), где может жить (минимальный класс железа), от кого
зависит (граф) и когда просыпается (триггеры).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .capsules import Capsule, Level
from .devices import Device, DeviceState, get_device

K, M, B = 1_000, 1_000_000, 1_000_000_000

#: порядок классов железа от слабого к сильному
TIERS: Tuple[str, ...] = ("micro", "small", "medium", "large", "datacenter")


def tier_rank(tier: str) -> int:
    return TIERS.index(tier) if tier in TIERS else 0


class Group(str, Enum):
    """Девять групп капсул."""

    SENSORY = "сенсорные"
    UNDERSTANDING = "понимания"
    MEMORY = "памяти"
    THINKING = "мышления"
    EXPERT = "экспертные"
    GENERATION = "генерации"
    CONTROL = "управления"
    ADAPTATION = "адаптации"
    SYSTEM = "системные"


class Stage(int, Enum):
    """Позиция в конвейере обработки запроса."""

    INPUT = 0        # приём сигнала
    PARSE = 1        # понимание
    RECALL = 2       # память
    REASON = 3       # мышление
    EXPERT = 4       # предметная экспертиза
    PRODUCE = 5      # генерация
    VERIFY = 6       # проверка
    BACKGROUND = 7   # фон: управление, адаптация, система


@dataclass(frozen=True)
class CapsuleSpec:
    """Паспорт капсулы из каталога."""

    key: str
    name: str
    group: Group
    does: str
    min_params: int
    max_params: int
    min_tier: str = "medium"
    min_ram: int = 0                # 0 = ограничения по RAM нет
    skills: frozenset = frozenset()
    triggers: frozenset = frozenset()
    depends_on: frozenset = frozenset()
    stage: Stage = Stage.REASON
    always_on: bool = False
    private: bool = False
    notes: str = ""

    # ------------------------------------------------------------- размеры
    def params_for(self, tier: str) -> int:
        """Сколько параметров капсула возьмёт на данном классе железа.

        Слабое железо получает нижнюю границу диапазона, мощное — верхнюю.
        """
        if tier_rank(tier) < tier_rank(self.min_tier):
            return 0
        span = tier_rank("datacenter") - tier_rank(self.min_tier)
        pos = tier_rank(tier) - tier_rank(self.min_tier)
        frac = 0.0 if span <= 0 else pos / span
        lo, hi = float(self.min_params), float(self.max_params)
        # геометрическая интерполяция: размеры растут на порядки, не линейно
        return int(round(lo * (hi / lo) ** frac))

    def bytes_for(self, tier: str, bits: float = 2.0) -> int:
        return int(self.params_for(tier) * bits / 8)

    def fits_on(self, device: Device) -> bool:
        """Влезает ли капсула на устройство: и по классу, и по объёму RAM."""
        return (tier_rank(device.tier) >= tier_rank(self.min_tier)
                and device.ram_bytes >= self.min_ram)

    def size_range(self) -> str:
        def h(n: int) -> str:
            if n >= B:
                return f"{n / B:g}B"
            if n >= M:
                return f"{n / M:g}M"
            if n >= K:
                return f"{n / K:g}K"
            return str(n)
        return f"{h(self.min_params)}–{h(self.max_params)}"

    def __str__(self) -> str:
        return f"{self.name} [{self.group.value}, {self.size_range()}, от {self.min_tier}]"


def _c(key, name, group, does, lo, hi, tier, skills=(), triggers=(),
       depends=(), stage=Stage.REASON, always_on=False, private=False, notes="",
       min_ram=0):
    return CapsuleSpec(key, name, group, does, lo, hi, tier, min_ram,
                       frozenset(skills), frozenset(triggers), frozenset(depends),
                       stage, always_on, private, notes)


# ═══════════════════════════════════════════ ГРУППА 1: СЕНСОРНЫЕ (7)
_SENSORY = [
    _c("text_in", "Текстовая", Group.SENSORY,
       "принимает байты текста, превращает в векторы смыслов",
       1 * M, 500 * M, "small", ("текст", "байты", "языки"), ("текст",),
       (), Stage.INPUT, notes="байтовый уровень: любой язык, опечатки, транслит"),
    _c("voice_in", "Голосовая", Group.SENSORY,
       "звук в байты: речь, интонация, эмоция, темп, диаризация",
       5 * M, 300 * M, "small", ("звук", "речь", "эмоция"), ("аудио", "голос"),
       (), Stage.INPUT, notes="офлайн; шум, акцент, шёпот"),
    _c("vision_in", "Зрительная", Group.SENSORY,
       "изображения и видео: объекты, лица, текст, сцены, движение",
       10 * M, 1 * B, "medium", ("зрение", "объекты", "лица"), ("изображение", "видео"),
       (), Stage.INPUT, notes="поток, а не кадры; «это твоя кошка»"),
    _c("sensor_in", "Сенсорная", Group.SENSORY,
       "датчики: температура, пульс, движение, свет, GPS, магнитометр",
       1 * K, 50 * M, "micro", ("датчики", "тело", "движение"), ("сенсор", "пульс"),
       (), Stage.INPUT, True, notes="реальное время; «пульс + движение = бег»"),
    _c("code_in", "Кодовая (вход)", Group.SENSORY,
       "код на любом языке: синтаксис, семантика, контекст проекта",
       10 * M, 1 * B, "medium", ("код", "синтаксис"), ("код",),
       (), Stage.INPUT, notes="структура проекта и зависимости"),
    _c("gesture_in", "Жестовая", Group.SENSORY,
       "движение рук, тела, мимика в смыслы",
       5 * M, 200 * M, "medium", ("жесты", "мимика"), ("жест", "видео"),
       (), Stage.INPUT, notes="ASL, РЖЯ, международный"),
    _c("formal_in", "Формальная", Group.SENSORY,
       "математика, логика, химия, ноты, ДНК, шахматы",
       1 * M, 500 * M, "medium", ("формальное", "математика", "ноты", "днк"),
       ("формула", "математика", "ноты"), (), Stage.INPUT,
       notes="каждый формальный язык — подкапсула"),
]

# ═══════════════════════════════════════ ГРУППА 2: ПОНИМАНИЯ (4)
_UNDERSTANDING = [
    _c("meaning", "Смысла", Group.UNDERSTANDING,
       "векторы в смыслы: не слова, а понятия",
       10 * M, 1 * B, "medium", ("смысл", "понятия"), (),
       ("text_in",), Stage.PARSE, notes="безъязыковая: собака = dog = 犬"),
    _c("intent", "Интента", Group.UNDERSTANDING,
       "что пользователь имел в виду, а не что сказал",
       5 * M, 200 * M, "small", ("интент", "намерение"), (),
       ("meaning",), Stage.PARSE, notes="вопрос / команда / просьба / шутка"),
    _c("context", "Контекста", Group.UNDERSTANDING,
       "время, место, активность, устройства, сеть, настроение",
       1 * M, 100 * M, "small", ("контекст", "время", "место"), (),
       (), Stage.PARSE, True, notes="в фоне, обновляется каждую секунду"),
    _c("task", "Задачи", Group.UNDERSTANDING,
       "классифицирует: факт, рассуждение, код, творчество, анализ, план",
       1 * M, 50 * M, "micro", ("классификация", "задача"), (),
       ("intent",), Stage.PARSE, True,
       notes="решает за миллисекунды, включает остальные капсулы"),
]

# ══════════════════════════════════════════ ГРУППА 3: ПАМЯТИ (6)
_MEMORY = [
    _c("working_mem", "Рабочей памяти", Group.MEMORY,
       "текущий контекст: последние сообщения, документы, факты",
       10 * M, 500 * M, "micro", ("память", "контекст"), (),
       (), Stage.RECALL, True, notes="аналог RAM: 2k / 8k / 128k токенов"),
    _c("episodic_mem", "Эпизодической памяти", Group.MEMORY,
       "события: «вчера ты спрашивал про проект»",
       50 * M, 1 * B, "medium", ("память", "события", "время"), (),
       (), Stage.RECALL, private=True, notes="временные метки, связь событий"),
    _c("semantic_mem", "Семантической памяти", Group.MEMORY,
       "факты о тебе: «твоя жена — Анна», «аллергия на пыль»",
       100 * M, 2 * B, "medium", ("память", "факты"), (),
       ("rag",), Stage.RECALL, True, private=True, notes="в RAG, только локально"),
    _c("procedural_mem", "Процедурной памяти", Group.MEMORY,
       "навыки: как ты пишешь письма, как решаешь задачи",
       10 * M, 500 * M, "medium", ("память", "навыки", "стиль"), (),
       ("lora",), Stage.RECALL, private=True, notes="хранится в LoRA-адаптерах"),
    _c("rag", "RAG-поиска", Group.MEMORY,
       "ищет в локальной базе: документы, заметки, сообщения",
       5 * M, 200 * M, "small", ("поиск", "документы"), (),
       (), Stage.RECALL, True, notes="HNSW, ~1 мс на 100 документах, офлайн"),
    _c("forgetting", "Забывания", Group.MEMORY,
       "удаляет устаревшее, неважное, дублирующееся",
       1 * M, 50 * M, "micro", ("забывание", "приватность"), (),
       (), Stage.BACKGROUND, True, private=True, notes="в фоне, оценивает важность"),
]

# ════════════════════════════════════════ ГРУППА 4: МЫШЛЕНИЯ (6)
_THINKING = [
    _c("logic", "Логики", Group.THINKING,
       "логические цепочки, проверка противоречий, выводы",
       50 * M, 2 * B, "medium", ("логика", "рассуждение"),
       ("рассуждение", "анализ", "код", "математика", "план", "наука",
        "право", "финансы", "медицина"),
       ("meaning",), Stage.REASON, notes="пошагово, с откатом"),
    _c("math", "Математики", Group.THINKING,
       "считает, решает уравнения, доказывает теоремы",
       20 * M, 1 * B, "medium", ("математика", "вычисления"), ("математика",),
       ("formal_in",), Stage.REASON, notes="символьные + численные методы"),
    _c("planning", "Планирования", Group.THINKING,
       "строит планы, разбивает задачи, распределяет шаги",
       30 * M, 1 * B, "medium", ("план", "декомпозиция"), ("план",),
       ("logic",), Stage.REASON, notes="на несколько шагов вперёд"),
    _c("creativity", "Творчества", Group.THINKING,
       "генерирует новое: идеи, стихи, истории, музыку",
       50 * M, 2 * B, "medium", ("творчество", "идеи"), ("творчество",),
       (), Stage.REASON, notes="высокая temperature + steering «творчество»"),
    _c("critic", "Критики", Group.THINKING,
       "проверяет ответ: логику, факты, стиль, безопасность",
       10 * M, 500 * M, "micro", ("критика", "проверка"), (),
       (), Stage.VERIFY, True, notes="может вернуть на доработку"),
    _c("reflection", "Рефлексии", Group.THINKING,
       "анализирует своё мышление: «я ошибся здесь»",
       20 * M, 1 * B, "medium", ("рефлексия", "самоанализ"), (),
       ("critic",), Stage.BACKGROUND, notes="в фоне, обновляет адаптеры"),
]

# ═══════════════════════════════════════ ГРУППА 5: ЭКСПЕРТНЫЕ (6)
_EXPERT = [
    _c("expert_code", "Кодовая экспертная", Group.EXPERT,
       "пишет, читает, рефакторит код на любом языке",
       100 * M, 7 * B, "medium", ("код", "рефакторинг"), ("код",),
       ("code_in",), Stage.EXPERT, notes="Python, C, Rust, JS; фреймворки"),
    _c("expert_med", "Медицинская экспертная", Group.EXPERT,
       "анализирует симптомы, лекарства, анализы",
       200 * M, 7 * B, "medium", ("медицина", "здоровье"), ("медицина",),
       (), Stage.EXPERT, private=True, notes="не заменяет врача; только локально"),
    _c("expert_law", "Юридическая экспертная", Group.EXPERT,
       "анализирует договоры, законы, права",
       200 * M, 7 * B, "medium", ("право", "договоры"), ("право",),
       (), Stage.EXPERT, private=True, notes="местное законодательство"),
    _c("expert_sci", "Научная экспертная", Group.EXPERT,
       "анализирует статьи, данные, гипотезы",
       500 * M, 14 * B, "medium", ("наука", "данные"), ("наука",),
       ("math",), Stage.EXPERT, notes="формулы и графики", min_ram=16 * 1024 ** 3),
    _c("expert_fin", "Финансовая экспертная", Group.EXPERT,
       "анализирует расходы, инвестиции, налоги",
       100 * M, 3 * B, "medium", ("финансы", "прогноз"), ("финансы",),
       (), Stage.EXPERT, private=True, notes="твои данные, прогнозы"),
    _c("expert_lang", "Языковая экспертная", Group.EXPERT,
       "переводит, объясняет грамматику, учит языку",
       50 * M, 2 * B, "medium", ("перевод", "языки", "грамматика"), ("перевод",),
       ("meaning",), Stage.EXPERT, notes="7000+ языков"),
]

# ══════════════════════════════════════ ГРУППА 6: ГЕНЕРАЦИИ (5)
_GENERATION = [
    _c("gen_text", "Текстовая генерация", Group.GENERATION,
       "пишет текст на любом языке, в любом стиле",
       50 * M, 7 * B, "micro", ("генерация", "текст"), (),
       ("steering",), Stage.PRODUCE, True, notes="учитывает стиль пользователя"),
    _c("gen_voice", "Голосовая генерация", Group.GENERATION,
       "говорит голосом, похожим на твой или любой другой",
       10 * M, 500 * M, "small", ("генерация", "голос"), ("голос",),
       (), Stage.PRODUCE, notes="клонирование, эмоции, офлайн"),
    _c("gen_image", "Визуальная генерация", Group.GENERATION,
       "рисует, редактирует, анимирует изображения",
       100 * M, 7 * B, "medium", ("генерация", "изображение"), ("изображение",),
       (), Stage.PRODUCE, notes="диффузия + трансформер", min_ram=8 * 1024 ** 3),
    _c("gen_code", "Кодовая генерация", Group.GENERATION,
       "пишет код, тесты, документацию",
       100 * M, 7 * B, "medium", ("генерация", "код"), ("код",),
       ("expert_code",), Stage.PRODUCE, notes="пошагово, с проверкой синтаксиса"),
    _c("action", "Действие", Group.GENERATION,
       "выполняет действия: открыть, закрыть, отправить, купить",
       1 * M, 100 * M, "micro", ("действие", "api"), ("действие",),
       ("intent", "safety"), Stage.PRODUCE, notes="API + проверка безопасности"),
]

# ═════════════════════════════════════ ГРУППА 7: УПРАВЛЕНИЯ (5)
_CONTROL = [
    _c("meta", "Мета-контроллер", Group.CONTROL,
       "какой размер, какие эксперты, где выход",
       10 * M, 100 * M, "micro", ("управление", "бюджет"), (),
       ("profiler",), Stage.BACKGROUND, True,
       notes="RL-агент; награда = качество − задержка − энергия"),
    _c("router", "Роутер", Group.CONTROL,
       "выбирает, какие эксперты активировать",
       5 * M, 50 * M, "micro", ("управление", "роутинг"), (),
       ("task",), Stage.BACKGROUND, True, notes="на каждый запрос, быстрый"),
    _c("early_exit", "Раннего выхода", Group.CONTROL,
       "решает, когда остановиться",
       1 * M, 20 * M, "micro", ("управление", "выход"), (),
       (), Stage.BACKGROUND, True, notes="проверяет качество на каждом слое"),
    _c("distribution", "Распределения", Group.CONTROL,
       "какие слои на каком узле",
       5 * M, 50 * M, "micro", ("управление", "рой"), (),
       ("network",), Stage.BACKGROUND, True, notes="пересобирается каждую секунду"),
    _c("safety", "Безопасности", Group.CONTROL,
       "проверяет запросы и ответы на безопасность",
       10 * M, 200 * M, "micro", ("безопасность", "приватность"), (),
       (), Stage.VERIFY, True, private=True, notes="локально, без облака"),
]

# ═════════════════════════════════════ ГРУППА 8: АДАПТАЦИИ (4)
_ADAPTATION = [
    _c("lora", "LoRA", Group.ADAPTATION,
       "хранит твои персональные адаптеры",
       1 * M, 100 * M, "medium", ("адаптация", "персонализация"), (),
       (), Stage.BACKGROUND, private=True,
       notes="на устройстве; защита от forgetting; несколько адаптеров"),
    _c("steering", "Steering vectors", Group.ADAPTATION,
       "направляет мышление в момент ответа",
       100 * K, 10 * M, "micro", ("адаптация", "направление"), (),
       (), Stage.BACKGROUND, True, notes="не меняет веса, работает мгновенно"),
    _c("feedback", "Фидбека", Group.ADAPTATION,
       "собирает реакции: лайки, правки, игноры",
       100 * K, 5 * M, "micro", ("адаптация", "обратная связь"), (),
       ("lora",), Stage.BACKGROUND, True, private=True, notes="в фоне"),
    _c("profiler", "Профилирования", Group.ADAPTATION,
       "зондирует железо: RAM, GPU, батарея, температура",
       100 * K, 5 * M, "micro", ("железо", "профиль"), (),
       (), Stage.BACKGROUND, True, notes="каждые несколько секунд"),
]

# ══════════════════════════════════════ ГРУППА 9: СИСТЕМНЫЕ (5)
_SYSTEM = [
    _c("energy", "Энергии", Group.SYSTEM,
       "управляет энергопотреблением",
       1 * M, 20 * M, "micro", ("энергия", "батарея"), (),
       ("profiler",), Stage.BACKGROUND, True, notes="сжимает Протея при низком заряде"),
    _c("thermal", "Тепла", Group.SYSTEM,
       "управляет троттлингом",
       1 * M, 10 * M, "small", ("тепло", "троттлинг"), (),
       ("profiler",), Stage.BACKGROUND, True, notes="перекладывает слои"),
    _c("network", "Сети", Group.SYSTEM,
       "управляет связью между узлами",
       1 * M, 20 * M, "micro", ("сеть", "p2p"), (),
       (), Stage.BACKGROUND, True, notes="Wi-Fi Direct, Bluetooth, Thread; шифрование"),
    _c("storage", "Хранения", Group.SYSTEM,
       "что хранить, что удалить, что сжать",
       1 * M, 50 * M, "micro", ("хранение", "сжатие"), (),
       (), Stage.BACKGROUND, True, notes="LRU-выгрузка, сжатие холодных слоёв"),
    _c("update", "Обновления", Group.SYSTEM,
       "новые знания, новые капсулы",
       1 * M, 30 * M, "medium", ("обновление",), (),
       ("network",), Stage.BACKGROUND, notes="в фоне, с откатом при ошибке"),
]

#: полный каталог: 48 капсул
CATALOG: Tuple[CapsuleSpec, ...] = tuple(
    _SENSORY + _UNDERSTANDING + _MEMORY + _THINKING + _EXPERT
    + _GENERATION + _CONTROL + _ADAPTATION + _SYSTEM
)

BY_KEY: Dict[str, CapsuleSpec] = {c.key: c for c in CATALOG}

#: сколько капсул в каждой группе (из сводной таблицы спецификации)
GROUP_COUNTS: Dict[Group, int] = {
    Group.SENSORY: 7, Group.UNDERSTANDING: 4, Group.MEMORY: 6,
    Group.THINKING: 6, Group.EXPERT: 6, Group.GENERATION: 5,
    Group.CONTROL: 5, Group.ADAPTATION: 4, Group.SYSTEM: 5,
}


def by_group(group: Group) -> List[CapsuleSpec]:
    return [c for c in CATALOG if c.group is group]


def resolve_deps(keys: Iterable[str]) -> Set[str]:
    """Дотянуть все зависимости: капсула не работает без тех, на ком стоит."""
    out: Set[str] = set()
    stack = list(keys)
    while stack:
        k = stack.pop()
        if k in out or k not in BY_KEY:
            continue
        out.add(k)
        stack.extend(BY_KEY[k].depends_on)
    return out
