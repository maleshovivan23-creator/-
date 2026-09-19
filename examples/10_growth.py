"""Протей от 0% до 100%: автоматический рост под железо.

Запуск:  PYTHONPATH=. python examples/10_growth.py
"""
from __future__ import annotations

from ultranet.proteus import (GrowingProteus, ladder_table, probe_hardware,
                              stage_at)
from ultranet.proteus.catalog import CATALOG

GB = 1024 ** 3
MB = 1024 ** 2


def title(n: int, s: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {s}\n{'=' * 78}")


# ─────────────────────────────────────────── 1. реальный зонд
title(1, "1% — ПЕРВЫЙ БАЙТ: ЗОНД НАСТОЯЩЕГО ЖЕЛЕЗА")

print("Капсула профилирования просыпается первой и смотрит, где оказалась:\n")
print(probe_hardware().report())

# ─────────────────────────────────────────── 2. шкала
title(2, "ШКАЛА РОСТА ИЗ СПЕЦИФИКАЦИИ")

print(ladder_table())

# ─────────────────────────────────────────── 3. путь 0→100
title(3, "ЖИВОЙ ПУТЬ: ОТ ГОЛОГО ЖЕЛЕЗА ДО ПОЛНОГО РОЯ")

p = GrowingProteus.bare()
print(f"  {'%':>5}  {'стадия':<30}{'капсул':>7}{'параметров':>12}{'память':>10}  событие")


def show(event: str) -> None:
    print(f"  {p.percent:>5.0f}  {p.stage.title:<30}"
          f"{len(p.active_capsules()):>5}/48{p.active_params() / 1e6:>10.0f}M"
          f"{p.memory_bytes() / MB:>8.0f}МБ  {event}")


show("голое железо, Протея нет")
p.probe(simulate="phone")
show("зонд: смартфон, 8 ГБ")
p.install()
show("установка: базовая семёрка")

for reqs, docs, label in [(15, 5, "первые вопросы"),
                          (60, 300, "первые документы"),
                          (200, 5_000, "ежедневное использование"),
                          (600, 60_000, "месяц работы"),
                          (1500, 400_000, "полгода, вся переписка")]:
    p.use(requests=reqs, documents=docs)
    show(label)

for dev, label in [("laptop", "+ ноутбук"), ("smartwatch", "+ часы"),
                   ("earbuds", "+ наушники"), ("tablet", "+ планшет"),
                   ("esp32", "+ ESP32 в замке")]:
    p.add_node(dev)
    p.use(requests=200, documents=50_000)
    show(f"{label} ({p.state.nodes} узлов)")

print(f"\n  оси роста:")
print(p.axes_bar())
print(f"\n{p.status()}")

# ─────────────────────────────────────────── 4. разное железо
title(4, "ОДИН КОД — РАЗНОЕ ЖЕЛЕЗО: ПОТОЛОК ЧЕСТНО РАЗНЫЙ")

print(f"  {'устройство':<20}{'RAM':>9}{'потолок':>9}{'капсул':>8}{'параметров':>12}  стадия")
for key in ("smartcard", "esp32", "ring", "earbuds", "smartwatch", "tv",
            "phone", "laptop", "workstation", "server"):
    g = GrowingProteus.install_on(key)
    for _ in range(30):
        g.use(requests=300, documents=80_000)
    ram = g.profile.ram_bytes
    r = f"{ram / GB:.0f} ГБ" if ram >= GB else (
        f"{ram / MB:.0f} МБ" if ram >= MB else f"{ram / 1024:.0f} КБ")
    print(f"  {g.profile.name:<20}{r:>9}{g.ceiling():>8.0f}%"
          f"{len(g.active_capsules()):>6}/48{g.active_params() / 1e6:>10.0f}M"
          f"  {g.stage.title}")
    assert g.fits_hardware(), "перерос железо!"

print("\n  ✓ ни одна конфигурация не переросла своё железо")

# ─────────────────────────────────────────── 5. рой спасает слабых
title(5, "РОЙ ПОДНИМАЕТ ПОТОЛОК: ЧАСЫ + НОУТБУК = ОДНО ТЕЛО")

solo = GrowingProteus.install_on("smartwatch")
for _ in range(30):
    solo.use(requests=300, documents=80_000)
print(f"  часы одни        : {solo.percent:>5.0f}%  потолок {solo.ceiling():>3.0f}%  "
      f"капсул {len(solo.active_capsules()):>2}  {solo.active_params() / 1e6:>6.0f}M")

pair = GrowingProteus.install_on("smartwatch")
pair.add_node("laptop")
for _ in range(30):
    pair.use(requests=300, documents=80_000)
print(f"  часы + ноутбук   : {pair.percent:>5.0f}%  потолок {pair.ceiling():>3.0f}%  "
      f"капсул {len(pair.active_capsules()):>2}  {pair.active_params() / 1e6:>6.0f}M")
print(f"\n  тяжёлые капсулы живут на сильном узле — часы получают доступ ко всему.")

# ─────────────────────────────────────────── 6. деградация
title(6, "ЧТО ВЫЖИВАЕТ НА КАЖДОМ УРОВНЕ ЖЕЛЕЗА")

for key in ("esp32", "earbuds", "smartwatch", "phone"):
    g = GrowingProteus.install_on(key)
    for _ in range(30):
        g.use(requests=300, documents=80_000)
    groups: dict = {}
    for c in g.active_capsules():
        groups[c.group.value] = groups.get(c.group.value, 0) + 1
    print(f"\n  {g.profile.name} ({g.ceiling():.0f}% потолок, "
          f"{len(g.active_capsules())} капсул):")
    print("    " + ", ".join(f"{k}: {v}" for k, v in sorted(groups.items())))

# ─────────────────────────────────────────── 7. рост по использованию
title(7, "«ПРОТЕЙ НЕ ОБНОВЛЯЕТСЯ. ОН РАСТЁТ»")

g = GrowingProteus.install_on("laptop")
print(f"  {'запросов':>9}{'документов':>12}{'узлов':>7}{'%':>7}{'капсул':>8}  стадия")
for reqs, docs, nodes in [(0, 0, 0), (10, 5, 0), (50, 100, 0), (200, 2_000, 1),
                          (600, 30_000, 2), (1500, 300_000, 3),
                          (2500, 1_000_000, 5)]:
    while g.state.requests < reqs:
        g.use(requests=min(50, reqs - g.state.requests),
              documents=max(0, (docs - g.state.documents) // 5))
    g.state.documents = max(g.state.documents, docs)
    while g.state.nodes < nodes + 1:
        for d in ("phone", "smartwatch", "earbuds", "tablet", "esp32"):
            if g.state.nodes < nodes + 1:
                g.add_node(d)
    print(f"  {g.state.requests:>9,}{g.state.documents:>12,}{g.state.nodes:>7}"
          f"{g.percent:>6.0f}%{len(g.active_capsules()):>6}/48  {g.stage.title}")

print(f"\n{'=' * 78}")
print("ИТОГ: Протей зондирует железо, разворачивается под него и растёт по мере")
print("использования — от 1 капсулы на смарт-карте до всех 48 в полном рое.")
print("Ни одна конфигурация не перерастает память, на которой живёт.")
print("=" * 78)
