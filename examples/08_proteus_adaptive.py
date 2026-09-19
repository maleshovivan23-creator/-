"""Протей v2: эксперты, steering, LoRA, капсулы, рой без лидера, четыре петли.

Запуск:  PYTHONPATH=. python examples/08_proteus_adaptive.py
"""
from __future__ import annotations

import itertools
import time

import numpy as np

import ultranet as un
from ultranet.functional import cross_entropy
from ultranet.optim import Adam
from ultranet.proteus import (Activity, AdaptationLoops, CapsuleRegistry, Context,
                              DeviceState, Level, MatConfig, MatFormer, Place,
                              Proteus, ShardedSwarm, get_device, style_for)


def title(n: int, s: str) -> None:
    print(f"\n{'=' * 74}\n{n}. {s}\n{'=' * 74}")


un.manual_seed(7)

# ─────────────────────────────────────────────────────── 1. эксперты
title(1, "MoE: РАЗНЫЕ 10% НЕЙРОНОВ ДЛЯ РАЗНЫХ ЗАДАЧ")

model = MatFormer(MatConfig(n_layer=4, n_embd=128, head_dim=32, block_size=64,
                            n_experts=6, top_k_experts=2))
print(f"всего параметров      : {model.num_params():,}")
for k in (1, 2, 3, 6):
    model.set_active_experts(k)
    a = model.active_params(1.0)
    print(f"  активно при top-{k}   : {a:>10,}  ({a / model.num_params() * 100:5.1f}% сети)")
model.set_active_experts(None)

print("\nроутер сам отправляет разные входы к разным экспертам:")
moe = model.blocks[0].moe
for name, seed in (("текст A", 1), ("текст B", 2), ("текст C", 3)):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 259, (1, 16))
    model(ids)
    top = np.argsort(-moe.last_load)[:2]
    print(f"  {name}: нагрузка {np.round(moe.last_load, 2)} -> любимые эксперты {list(top)}")

# ─────────────────────────────────────────────────── 2. steering
title(2, "STEERING-ВЕКТОРЫ: НАПРАВЛЕНИЕ МЫШЛЕНИЯ БЕЗ ИЗМЕНЕНИЯ ВЕСОВ")

p = Proteus.with_experts(n_experts=6, top_k=2, device="phone")
lib = p.steering
lib.learn("математика",
          ["2+2=4 посчитай сумму", "реши уравнение x=5", "умножь три на семь"],
          ["луна над морем светит", "ветер шумит в листве"], layer=-2)
lib.learn("поэзия",
          ["луна над морем светит", "сердце поёт о любви", "туман плывёт над рекой"],
          ["def f(x): return x", "import numpy as np"], layer=-2)
lib.learn("простота",
          ["кот сидит на окне", "это просто и понятно"],
          ["имплементация асинхронного планировщика требует изоляции контекста"], layer=1)
lib.learn("детальность",
          ["подробный разбор всех аспектов и деталей системы"],
          ["ага", "да", "нет"], layer=1)

weights_before = [x.data.copy() for x in p.model.parameters()]
print(f"{'вектор':<14}{'слой':>5}{'|v|':>7}{'эффект@1':>11}{'эффект@4':>11}")
for n in lib.names():
    v = lib[n]
    print(f"{n:<14}{v.layer:>5}{np.linalg.norm(v.vector):>7.2f}"
          f"{lib.effect_size(n, strength=1.0):>11.4f}{lib.effect_size(n, strength=4.0):>11.4f}")

print("\nпопарные косинусы (разные концепты — разные направления):")
for a, b in itertools.combinations(lib.names(), 2):
    print(f"  cos(«{a}», «{b}») = {float(np.dot(lib[a].direction, lib[b].direction)):+.3f}")

with lib.active("математика", 3.0):
    p.respond("посчитай", max_new_tokens=5, seed=0)
same = all(np.array_equal(a, b.data) for a, b in zip(weights_before, p.model.parameters()))
print(f"\nвеса модели после применения векторов не изменились: {same}")

# ─────────────────────────────────────────────────── 3. LoRA
title(3, "LoRA: ЯДРО ЗАМОРОЖЕНО, УЧИТСЯ ТОЛЬКО НАДСТРОЙКА")

pa = p.enable_adapter(rank=4)
print(pa.report())
core_before = [x.data.copy() for x in p.model.parameters()]
for _ in range(4):
    p.feedback("отвечай кратко. без воды. по делу. это мой стиль.", reward=1.0)
pa.add_replay(p.tok.encode("общие знания не должны забыться при персонализации"))
t0 = time.time()
hist = p.consolidate(steps=40, lr=5e-3)
print(f"loss адаптера: {hist['loss'][0]:.3f} -> {hist['loss'][-1]:.3f} "
      f"за {time.time() - t0:.1f} с")
print(f"ядро не тронуто: "
      f"{all(np.array_equal(a, b.data) for a, b in zip(core_before, p.model.parameters()))}")
print(f"передать другому устройству нужно лишь {pa.size_bytes() / 1024:.1f} КБ "
      f"вместо {p.model.num_params() * 2 / 8 / 1024:.0f} КБ весов")

print("\nзащита от забывания:")
from ultranet.proteus import LoRALayer
a, b = LoRALayer(64, 64, rank=4), LoRALayer(64, 64, rank=4)
print(f"  пересечение подпространств до ортогонализации : {a.overlap_with(b):.3f}")
b.orthogonalize_to([a])
print(f"  после                                        : {a.overlap_with(b):.3f}")
print(f"  replay-буфер: {len(pa.replay)} примеров | EWC-якорей: {len(pa.anchor)}")

# ─────────────────────────────────────────────────── 4. капсулы
title(4, "НАНОЧАСТИЦЫ: АТОМ -> МОЛЕКУЛА -> КЛЕТКА -> ОРГАН -> ТЕЛО")

reg = p.enable_capsules({0: {"код"}, 1: {"творчество"}, 2: {"математика"},
                         3: {"диалог"}, 4: {"логика"}, 5: {"анализ"}})
print(reg.report())
print("\nсборка тела под задачу (спящие капсулы не занимают ничего):")
for task in (["код"], ["творчество"], ["математика", "логика"],
             ["код", "творчество", "математика", "диалог", "логика", "анализ"]):
    reg.assemble(task)
    print(f"  {str(task):<58} {reg.active_fraction() * 100:5.1f}% "
          f"({reg.active_params():>9,} пар.)")
reg.sleep_all()
print(f"  {'всё спит':<58} {reg.active_fraction() * 100:5.1f}% "
      f"({reg.active_params():>9,} пар.)")

reg.assemble(["код"])
print("\nфрагмент дерева капсул (● активна, ○ спит):")
print("\n".join(reg.root.children[1].tree().splitlines()[:8]))

# ─────────────────────────────────────────────────── 5. рой
title(5, "РОЙ БЕЗ ЛИДЕРА: ШАРДЫ, КОНСЕНСУС, МИГРАЦИЯ")

big = MatFormer(MatConfig(n_layer=8, n_embd=96, head_dim=32, block_size=32))
skills = {0: {"слух"}, 1: {"слух"}, 2: {"контекст"}, 3: {"память"},
          4: {"код"}, 5: {"код"}, 6: {"логика"}, 7: {"генерация"}}
sw = ShardedSwarm(big, [DeviceState(get_device(k))
                        for k in ("earbuds", "smartwatch", "phone", "laptop")],
                  skills_by_layer=skills)
print(sw.topology())
probe = np.array([[1, 2, 3, 4, 5]])
ref = big(probe).data.copy()
out = sw.forward(probe)
print(f"\nвывод роя идентичен монолиту: {np.allclose(out.data, ref, atol=1e-5)}")
print(f"прыжков: {sw.hops}, по сети ушло {sw.bytes_moved / 1024:.1f} КБ активаций "
      f"против {big.num_params() * 2 / 8 / 1024:.0f} КБ весов")

print("\nроутер выбирается под задачу — постоянного лидера нет:")
for need, cx in ((["код"], 0.9), (["слух"], 0.1), (["память"], 0.5)):
    n = sw.elect_router(need, cx)
    print(f"  задача {str(need):<12} -> роутер {n.name:<14} (score {n.router_score:.2f})")

print("\nотказоустойчивость:")
sw.leave("Ноутбук")
print(f"  ноутбук умер   -> покрытие {sw.coverage()}, вывод сохранён "
      f"{np.allclose(sw.forward(probe).data, ref, atol=1e-5)}")
sw.join(DeviceState(get_device("tablet")))
print(f"  пришёл планшет -> покрытие {sw.coverage()}, вывод сохранён "
      f"{np.allclose(sw.forward(probe).data, ref, atol=1e-5)}")
sw.rejoin("Ноутбук")
print(f"  ноутбук вернулся -> покрытие {sw.coverage()}, перестроений: {sw.rebuilds}")
print(f"  консенсус «расшириться»: {sw.consensus('расшириться')}")

# ─────────────────────────────────────────────────── 6. контекст
title(6, "АДАПТИВНОСТЬ К КОНТЕКСТУ")

for ctx in (Context(8, Place.TRANSIT, Activity.DRIVING),
            Context(2, Place.HOME, Activity.SLEEPING, headphones=True),
            Context(14, Place.WORK, Activity.WORKING),
            Context(19, Place.OUTDOORS, Activity.WALKING, headphones=True),
            Context(11, Place.WORK, Activity.MEETING),
            Context(12, Place.HOME, Activity.IDLE)):
    print(f"  {str(ctx):<48}")
    print(f"     -> {style_for(ctx)}")

# ─────────────────────────────────────────────────── 7. петли
title(7, "ЧЕТЫРЕ ПЕТЛИ АДАПТАЦИИ, РАБОТАЮЩИЕ ОДНОВРЕМЕННО")

loops = AdaptationLoops(p, consolidate_steps=3)
for i in range(60):
    loops.tick(Context(hour=i % 24, place=Place.WORK,
                       activity=Activity.DRIVING if i % 3 == 0 else Activity.WORKING),
               prompt="объясни почему так вышло")
print(loops.report())
print("\nсобытия медленных петель:")
for e in loops.log:
    if e.loop in ("часы", "недели"):
        print(f"  тик {e.at_step:>3}: [{e.loop}] {e.what}")

# ─────────────────────────────────────────────────── 8. всё вместе
title(8, "ПОЛНЫЙ ЦИКЛ: ОДИН ПРОТЕЙ, РАЗНЫЕ ЗАДАЧИ")

p.bind_vector("математика", "математика")
p.bind_vector("творчество", "поэзия")
p.remember("проект пользователя называется Протей")
p.remember("любимый язык программирования — Python")

for prompt in ("напиши функцию сортировки", "сочини стих про море",
               "сколько будет 17 умножить на 3", "объясни почему небо голубое",
               "как называется мой проект"):
    r = p.respond(prompt, max_new_tokens=6, seed=1)
    print(f"\n  «{prompt}»")
    print(f"     задача={Proteus.classify_task(prompt):<12} экспертов={r.experts_active} "
          f"капсул={r.capsules_active * 100:.0f}% вектор={r.steering or '—'}")
    print(f"     {r.plan}")
    if r.context_used:
        print(f"     вспомнил: {r.context_used[:60]}")

print(f"\n{'=' * 74}")
print("ИТОГ: одна модель — разные конфигурации под задачу, устройство и контекст.")
print("Ядро ни разу не переобучалось: менялись только маршруты, векторы и надстройки.")
print("=" * 74)
