"""Все 48 капсул Протея: каталог, сборка под задачу, жизнь на разном железе.

Запуск:  PYTHONPATH=. python examples/09_capsule_catalog.py
"""
from __future__ import annotations

from ultranet.proteus import CATALOG, Group, ProteusBody, Stage, by_group
from ultranet.proteus.catalog import GROUP_COUNTS
from ultranet.proteus.devices import get_device


def title(n: int, s: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {s}\n{'=' * 78}")


# ───────────────────────────────────────────────── 1. полный каталог
title(1, "ПОЛНЫЙ КАТАЛОГ: 48 КАПСУЛ В ДЕВЯТИ ГРУППАХ")

for g in Group:
    caps = by_group(g)
    print(f"\n  ── {g.value.upper()} ({len(caps)}) " + "─" * (52 - len(g.value)))
    for c in caps:
        flags = ("⚡" if c.always_on else " ") + ("🔒" if c.private else " ")
        print(f"   {flags} {c.name:<26} {c.size_range():>10}  от {c.min_tier:<7} {c.does[:34]}")

print(f"\n  ⚡ = работает всегда   🔒 = приватная, только локально")
print(f"  ВСЕГО: {len(CATALOG)} капсул, "
      f"{sum(GROUP_COUNTS.values())} по сводной таблице спецификации")

# ───────────────────────────────────────────── 2. сценарий спеки
title(2, "СЦЕНАРИЙ ИЗ СПЕЦИФИКАЦИИ: «НАПИШИ КОД СОРТИРОВКИ»")

body = ProteusBody("phone")
asm = body.assemble("напиши код сортировки")
print(asm.explain())

print("\n  проверка по шагам спецификации:")
steps = ["Текстовая", "Интента", "Задачи", "Контекста", "RAG-поиска",
         "Кодовая экспертная", "Логики", "Кодовая генерация", "Критики",
         "Безопасности", "Текстовая генерация"]
names = asm.names()
for i, s in enumerate(steps, 1):
    print(f"    {i:>2}. {'✓' if s in names else '✗'} {s}")
print("\n  не активировалось (как и обещано спецификацией):")
for s in ("Медицинская экспертная", "Юридическая экспертная",
          "Финансовая экспертная", "Творчества", "Голосовая"):
    print(f"    {'✓ спит' if s not in names else '✗ АКТИВНА'}  {s}")

# ───────────────────────────────────────── 3. разные задачи
title(3, "РАЗНЫЕ ЗАДАЧИ — РАЗНЫЕ ОРГАНЫ")

queries = [
    "напиши код сортировки", "сочини стих про море", "сколько будет 17 умножить на 3",
    "у меня болит голова, что делать", "проверь договор аренды",
    "посчитай мои налоги за год", "переведи на японский", "нарисуй закат",
    "открой почту", "объясни почему небо голубое",
]
print(f"  {'запрос':<36}{'капсул':>7}{'конвейер':>10}{'параметров':>12}  эксперты / мышление")
for q in queries:
    a = body.assemble(q)
    special = [c.name for c in a.active
               if c.group in (Group.EXPERT,) or
               (c.group is Group.THINKING and c.name != "Критики")]
    print(f"  {q[:34]:<36}{a.n_active:>5}/48{a.foreground_fraction * 100:>9.0f}%"
          f"{a.params / 1e6:>10.0f}M  {', '.join(special)[:38] or '—'}")

# ─────────────────────────────────────── 4. железо
title(4, "ГДЕ КАКИЕ КАПСУЛЫ ПОМЕЩАЮТСЯ")

print(body.device_capabilities())
print("\n  реальная сборка одного и того же запроса «включи свет»:")
print(f"  {'устройство':<20}{'бюджет':>12}{'капсул':>8}{'занято':>12}  что выжило")
for key in ("smartcard", "esp32", "earbuds", "smartwatch", "phone", "laptop"):
    b = ProteusBody(key)
    a = b.assemble("включи свет")
    kb = a.budget / 1024
    bs = f"{kb:.0f} КБ" if kb < 1024 else f"{kb / 1024:.0f} МБ"
    us = (f"{a.bytes_used / 1024:.0f} КБ" if a.bytes_used < 1024 ** 2
          else f"{a.bytes_used / 1024 / 1024:.0f} МБ")
    print(f"  {b.device.name:<20}{bs:>12}{a.n_active:>6}/48{us:>12}  "
          f"{', '.join(a.names()[:3])[:34]}")

# ─────────────────────────────────────── 5. инвентарь
title(5, "ИНВЕНТАРЬ НА СМАРТФОНЕ")

print(body.inventory())

# ─────────────────────────────────────── 6. зависимости
title(6, "ЗАВИСИМОСТИ: КАПСУЛА НЕ РАБОТАЕТ В ОДИНОЧКУ")

for key in ("gen_code", "semantic_mem", "action", "expert_sci", "meta"):
    from ultranet.proteus import BY_KEY
    from ultranet.proteus.catalog import resolve_deps
    spec = BY_KEY[key]
    deps = sorted(resolve_deps([key]) - {key})
    print(f"  {spec.name:<26} тянет: {', '.join(BY_KEY[d].name for d in deps) or '—'}")

# ─────────────────────────────────────── 7. приватность
title(7, "ПРИВАТНЫЕ КАПСУЛЫ: ДАННЫЕ НЕ ПОКИДАЮТ УСТРОЙСТВО")

private = [c for c in CATALOG if c.private]
print(f"  {len(private)} капсул помечены приватными:")
for c in private:
    print(f"    🔒 {c.name:<28} {c.group.value:<14} {c.notes[:40]}")

print(f"\n{'=' * 78}")
print(f"ИТОГ: {len(CATALOG)} капсул — органы Протея. Каждая знает свой размер,")
print("своё железо и своих соседей. Под запрос собирается только нужное.")
print("=" * 78)
