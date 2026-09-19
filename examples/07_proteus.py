"""Протей: демонстрация всех заявлений спецификации на работающем коде.

Запуск:  PYTHONPATH=. python examples/07_proteus.py
"""
import numpy as np

import ultranet as un
from ultranet.proteus import (
    ByteTokenizer,
    DeviceState,
    LocalMemory,
    MatConfig,
    MatFormer,
    MetaController,
    Proteus,
    Swarm,
    TernaryModel,
    evaluate_widths,
    get_device,
    train_matryoshka,
)

LINE = "=" * 74


def head(n: int, title: str) -> None:
    print(f"\n{LINE}\n  Часть {n}. {title}\n{LINE}")


# --------------------------------------------------------------------------
def part1_language() -> ByteTokenizer:
    head(1, "На каком языке говорит Протей")
    tok = ByteTokenizer()
    print(f"Словаря нет вовсе: vocab_size = {tok.vocab_size} (256 байт + 4 спецтокена)\n")

    samples = {
        "русский": "нейронная сеть",
        "english": "neural network",
        "日本語": "ニューラルネット",
        "العربية": "شبكة عصبية",
        "код": "def f(x): return x**2",
        "ДНК": "ATGCATGCATGC",
        "ноты": "♪ C#m7 ♫ F♯",
        "эмодзи": "🚀🧠✨",
        "руны": "ᚦᚢᚱᛁᛊᚨᛉ",
    }
    print(f"{'язык':<12} {'символов':>9} {'байтов':>8} {'байт/симв':>10}  roundtrip")
    print("-" * 58)
    for name, text in samples.items():
        ok = tok.decode(tok.encode(text)) == text
        print(f"{name:<12} {len(text):>9} {len(text.encode()):>8} "
              f"{tok.bytes_per_char(text):>10.2f}  {'✓' if ok else '✗'}")

    print("\nЧестное уточнение: байты убирают штраф СЛОВАРЯ (нет 'английских' токенов),")
    print("но не штраф UTF-8 — кириллица физически занимает 2 байта, CJK 3 байта.")
    print("Заявление «все языки одинаково дёшевы» верно про словарь, не про кодировку.")
    return tok


# --------------------------------------------------------------------------
def part2_nested(tok: ByteTokenizer):
    head(2, "Вложенность: одна модель — все размеры")
    corpus = ("нейронная сеть учится на данных. gradient flows backward. "
              "模型学习数据。 def f(x): return x*2. ") * 45
    ids = tok.encode(corpus)
    print(f"корпус: {len(ids)} байт, 4 языка в одном потоке\n")

    cfg = MatConfig(n_layer=4, n_embd=96, head_dim=32, block_size=48,
                    widths=(0.33, 0.67, 1.0), exit_layers=(1,))
    model = MatFormer(cfg)
    print(f"ОДИН набор весов: {model.num_params():,} параметров")
    print("Обучаем все размеры одновременно (matryoshka training):")
    train_matryoshka(model, ids, steps=220, batch_size=8, log_every=55)

    print("\nКаждый срез — рабочая модель:")
    res = evaluate_widths(model, ids, [0.33, 0.67, 1.0])
    print(f"{'ширина':>8} {'активно':>10} {'loss':>7} {'ppl':>8} {'бит/байт':>9}")
    print("-" * 46)
    for w, r in res.items():
        print(f"{w:>8.2f} {int(r['active_params']):>10,} {r['loss']:>7.3f} "
              f"{r['ppl']:>8.2f} {r['bpb']:>9.2f}")
    print(f"\nСлучайная модель дала бы loss = {np.log(cfg.vocab_size):.2f}")
    return model, ids


# --------------------------------------------------------------------------
def part3_quant(model: MatFormer):
    head(3, "Квантование 1.58 бит: веса в {-1, 0, +1}")
    q = TernaryModel(model)
    print(q.stats)
    print(f"\nИзмеренная энтропия {q.stats.entropy_bits:.3f} бит — "
          f"это и есть «1.58 бит» (log2(3) = {np.log2(3):.3f}).")
    print("Упаковка: 2 бита на вес физически (4 веса в байт).")


# --------------------------------------------------------------------------
def part4_devices(model: MatFormer):
    head(4, "Протей на всех устройствах: от смарт-карты до кластера")
    mc = MetaController(model)
    keys = ["smartcard", "esp32", "stm32", "ring", "earbuds", "smartwatch",
            "glasses", "phone", "tablet", "laptop", "console", "workstation",
            "server", "cluster"]
    print(f"{'устройство':<22} {'RAM':>8} {'бюджет':>9} {'активно':>10} {'вес':>9} {'слоёв':>6}")
    print("-" * 70)
    for k in keys:
        dev = get_device(k)
        st = DeviceState(dev)
        plan = mc.plan(st, 0.9)
        ram = (f"{dev.ram_bytes / 1024**3:.0f}ГБ" if dev.ram_bytes >= 1024**3
               else f"{dev.ram_bytes / 1024**2:.0f}МБ" if dev.ram_bytes >= 1024**2
               else f"{dev.ram_bytes / 1024:.0f}КБ")
        bud = (f"{st.available_bytes / 1024**2:.0f}МБ" if st.available_bytes >= 1024**2
               else f"{st.available_bytes / 1024:.0f}КБ")
        wt = (f"{plan.weight_bytes / 1024**2:.1f}МБ" if plan.weight_bytes >= 1024**2
              else f"{plan.weight_bytes / 1024:.0f}КБ")
        flag = "" if plan.fits else " ✗"
        print(f"{dev.name:<22} {ram:>8} {bud:>9} {plan.active_params:>10,} {wt:>9} "
              f"{plan.n_layers:>6}{flag}")

    print("\nОдно и то же устройство в разных состояниях:")
    for label, kw in [("заряжен, холодный", dict(battery_pct=100, temperature_c=30)),
                      ("50% заряда", dict(battery_pct=50, temperature_c=40)),
                      ("5% заряда", dict(battery_pct=5, temperature_c=40)),
                      ("перегрев 90°C", dict(battery_pct=80, temperature_c=90))]:
        st = DeviceState(get_device("phone"), **kw)
        p = mc.plan(st, 0.9)
        print(f"  {label:<20} throttle={st.throttle:.2f} -> {p.active_params:>9,} параметров")


# --------------------------------------------------------------------------
def part5_memory(model: MatFormer):
    head(5, "RAG: знание живёт вне весов")
    mem = LocalMemory()
    facts = [
        "Столица Франции — Париж.",
        "Протей никогда не отправляет данные в облако.",
        "日本の首都は東京です。",
        "Вода кипит при 100 градусах Цельсия.",
        "Градиент течёт назад через граф вычислений.",
    ]
    mem.add_many(facts)
    mem.add_many([f"Запись {i}: код {i * 13}." for i in range(300)])

    weights = TernaryModel(model).stats.packed_bytes
    print(f"вес модели  : {weights / 1024:>8.1f} КБ")
    print(f"вес знаний  : {mem.size_bytes() / 1024:>8.1f} КБ ({len(mem)} фактов)")
    print("Знание дешевле параметров — и обновляется без переобучения.\n")
    for q in ["париж", "東京", "Запись 137", "облако"]:
        doc, score = mem.search_scored(q, 1)[0]
        print(f"  «{q}» -> {doc.text[:46]:<48} (score {score:.1f})")


# --------------------------------------------------------------------------
def part6_swarm(model: MatFormer):
    head(6, "Рой: один организм, разлитый по железу")
    x = np.array([[104, 101, 108, 108, 111]])
    ref = model(x).data

    sw = Swarm(model, [DeviceState(get_device(k))
                       for k in ["earbuds", "smartwatch", "phone", "laptop"]])
    print("Наушники слышат, часы думают, телефон помнит, ноутбук считает:")
    print(sw.topology())
    same = np.allclose(sw.forward(x).data, ref, atol=1e-5)
    print(f"\nвывод роя идентичен монолиту: {same}")
    print(sw.transfer_report(x))

    print("\n«Устройство умирает — Протей отступает»:")
    sw.leave("Ноутбук")
    print(sw.topology())
    print(f"  вывод сохранился: {np.allclose(sw.forward(x).data, ref, atol=1e-5)}")

    print("\n«Появляется новое устройство — Протей растекается»:")
    sw.join(DeviceState(get_device("tablet")))
    print(sw.topology())
    print(f"  вывод сохранился: {np.allclose(sw.forward(x).data, ref, atol=1e-5)}")

    print("\n«Когда устройств нет рядом — каждый автономен»:")
    for name in ["Планшет", "Смартфон", "Умные часы"]:
        sw.leave(name)
    print(sw.topology())
    print(f"  наушники держат всю модель: "
          f"{np.allclose(sw.forward(x).data, ref, atol=1e-5)}")


# --------------------------------------------------------------------------
def part7_live(model: MatFormer, ids):
    head(7, "Полный цикл: запрос -> профилирование -> ответ")
    p = Proteus(model=model, device="phone")
    p.remember("Столица Франции — Париж.", "Протей работает офлайн.")

    for prompt, device, kw in [
        ("включи свет", "esp32", {}),
        ("нейронная сеть", "smartwatch", {}),
        ("объясни почему градиент затухает в глубоких сетях", "laptop", {}),
        ("нейронная сеть", "phone", dict(battery_pct=5, temperature_c=85)),
    ]:
        p.attach(device, **kw)
        r = p.respond(prompt, max_new_tokens=24, seed=0)
        print(f"\n[{get_device(device).name}] «{prompt}»")
        print(r.trace())
        print(f"  ответ     : {r.text[:60]!r}")


def main() -> None:
    un.manual_seed(1337)
    tok = part1_language()
    model, ids = part2_nested(tok)
    part3_quant(model)
    part4_devices(model)
    part5_memory(model)
    part6_swarm(model)
    part7_live(model, ids)

    print(f"\n{LINE}")
    print("  Протей говорит на любом языке, думает без языка,".center(74))
    print("  живёт на любом устройстве и становится одним организмом.".center(74))
    print(LINE)


if __name__ == "__main__":
    main()
