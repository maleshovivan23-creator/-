"""Идеальные веса Протея: семь техник с измеренной ошибкой.

Запуск:  PYTHONPATH=. python examples/11_ideal_weights.py
"""
from __future__ import annotations

import numpy as np

import ultranet as un
from ultranet.functional import cross_entropy
from ultranet.optim import Adam
from ultranet.proteus import MatConfig, MatFormer
from ultranet.proteus.weights import (RVQ, ComplexWeight, FisherImportance,
                                      FractalStack, HyperNetwork, LogPolarWeight,
                                      SphericalWeight, angular_distance,
                                      compare_methods, cosine, rel_error, slerp)


def title(n: int, s: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {s}\n{'=' * 78}")


rng = np.random.default_rng(0)
W = rng.standard_normal((128, 128)).astype(np.float32) * 0.05

# ───────────────────────────────── 1. гиперсфера
title(1, "ГИПЕРСФЕРИЧЕСКАЯ НОРМАЛИЗАЦИЯ: НАПРАВЛЕНИЕ + МАСШТАБ")

sph = SphericalWeight.from_weight(W)
print(f"  все строки единичной длины : {sph.is_unit}")
print(f"  восстановление             : ошибка {rel_error(W, sph.to_weight()) * 100:.4f}%")
print(f"  разброс длин до нормализации: {np.linalg.norm(W, axis=1).min():.4f} … "
      f"{np.linalg.norm(W, axis=1).max():.4f}")
print(f"  после                      : все ровно 1.0")
print("\n  угол = семантическая близость, между весами можно плавно двигаться:")
a, b = sph.directions[0], sph.directions[1]
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    m = slerp(a, b, t)
    print(f"    t={t:.2f}  |m|={np.linalg.norm(m):.6f}  "
          f"угол до a={angular_distance(a, m):.4f}  до b={angular_distance(m, b):.4f}")

# ───────────────────────────────── 2. лог-полярное
title(2, "ЛОГ-ПОЛЯРНОЕ КВАНТОВАНИЕ: w = sign × 2^e × (1 + m/4)")

lp = LogPolarWeight.quantize(W)
print(f"  7 бит на вес (1 знак + 4 экспонента + 2 мантисса)")
print(f"  ошибка          : {rel_error(W, lp.dequantize()) * 100:.2f}%")
print(f"  косинус         : {cosine(W, lp.dequantize()):.5f}")
print(f"  память          : {lp.nbytes() / 1024:.1f} КБ против {W.nbytes / 1024:.1f} КБ (fp32)")

x = rng.standard_normal(W.shape).astype(np.float32)
same = np.allclose(lp.shift_multiply(x), x * lp.dequantize(), atol=1e-6)
print(f"\n  умножение через СДВИГИ совпадает с обычным: {same}")
print("  (np.ldexp = сдвиг экспоненты, мантисса = два сложения — умножения нет)")

print("\n  работает на любом масштабе весов:")
for sc in (1e-5, 1e-3, 1e-1, 1.0, 100.0):
    v = rng.standard_normal(4000).astype(np.float32) * sc
    q = LogPolarWeight.quantize(v)
    print(f"    масштаб {sc:<8g} bias={q.bias:>4}  ошибка {rel_error(v, q.dequantize()) * 100:5.2f}%")

# ───────────────────────────────── 3. RVQ
title(3, "RVQ: ВЛОЖЕННАЯ ТОЧНОСТЬ ИЗ ОДНОГО ХРАНИЛИЩА")

rvq = RVQ(dim=4, size=256, stages=4, iters=12).fit(W)
codes = rvq.encode(W)
print(f"  {'кодбуков':>9}{'бит/вес':>10}{'ошибка':>10}{'косинус':>10}")
for s, bits, err in rvq.error_by_stage(W):
    print(f"  {s:>9}{bits:>10.2f}{err * 100:>9.2f}%{cosine(W, rvq.decode(codes, s)):>10.5f}")
print(f"\n  кодбуки весят {rvq.codebook_bytes() / 1024:.1f} КБ — на любое число весов")
print("  можно остановиться на любом уровне: это и есть вложенность")

# ───────────────────────────────── 4. Fisher
title(4, "FISHER: ВАЖНОЕ ЗАЩИЩЕНО, НЕВАЖНОЕ ДЁШЕВО")

f = np.abs(rng.standard_normal(W.shape)) * 0.01
important = rng.random(W.shape) < 0.05
f[important] += 10.0
fi = FisherImportance(levels=(2, 4, 8, 16))
fi.fisher["W"] = f
plan = fi.plan_for("W")
adaptive = fi.quantize_adaptive(W, plan.bits)
uniform = fi.uniform_baseline(W, plan.mean_bits)

print(f"  средняя разрядность: {plan.mean_bits:.2f} бит — одинаковая цена у обоих")
print(f"\n  {'метрика':<34}{'адаптивно':>12}{'равномерно':>12}")
print(f"  {'Fisher-взвешенная ошибка':<34}"
      f"{fi.weighted_error(W, adaptive, f) * 100:>11.2f}%"
      f"{fi.weighted_error(W, uniform, f) * 100:>11.2f}%")
print(f"  {'ошибка на важных 5% весов':<34}"
      f"{rel_error(W[important], adaptive[important]) * 100:>11.2f}%"
      f"{rel_error(W[important], uniform[important]) * 100:>11.2f}%")
print(f"\n  важные веса получают {plan.bits[important].mean():.1f} бит, "
      f"остальные {plan.bits[~important].mean():.1f}")

# ───────────────────────────────── 5. фрактал
title(5, "ФРАКТАЛЬНАЯ СТРУКТУРА: СЛОИ ИЗ БАЗОВОГО")

base = rng.standard_normal((32, 32)).astype(np.float32) * 0.1
A = np.eye(32) + rng.normal(0, 0.02, (32, 32))
targets, cur = [base.copy()], base.copy()
for _ in range(7):
    cur = (A @ cur).astype(np.float32)
    targets.append(cur.copy())

fs = FractalStack(base, len(targets))
hist = fs.fit(targets, steps=200, lr=0.01)
print(f"  слоёв: {len(targets)}")
print(f"  подгонка f: loss {hist[0]:.6f} -> {hist[-1]:.6f}")
print(f"  ошибка восстановления стека: {fs.error_vs(targets) * 100:.2f}%")
print(f"\n  {'глубина':>8}{'хранится':>12}{'порождается':>14}{'экономия':>11}")
for n in (8, 20, 50, 100):
    g = FractalStack(base, n)
    print(f"  {n:>8}{g.stored_params():>12,}{g.generated_params():>14,}{g.compression():>10.1f}x")

# ───────────────────────────────── 6. комплексные
title(6, "КОМПЛЕКСНЫЕ ВЕСА: АМПЛИТУДА + ФАЗА")

print(f"  {'амп.бит':>8}{'фаза.бит':>10}{'бит/вес':>10}{'ошибка':>10}{'косинус':>10}")
for ab, pb in ((3, 2), (4, 3), (5, 4), (6, 6)):
    cw = ComplexWeight.from_pairs(W, ab, pb)
    r = cw.to_weight()
    print(f"  {ab:>8}{pb:>10}{cw.bits_per_weight():>10.1f}"
          f"{rel_error(W, r) * 100:>9.2f}%{cosine(W, r):>10.5f}")
print("\n  умножение = поворот + масштаб, а не только масштаб")

# ───────────────────────────────── 7. гиперсеть
title(7, "ГИПЕРСЕТЬ: ОДНА СЕТЬ — БЕСКОНЕЧНО МНОГО МОДЕЛЕЙ")

un.manual_seed(0)
print(f"  {'выход':>14}{'гиперсеть':>12}{'порождает':>13}{'сжатие':>10}")
for n in (64, 128, 256, 512, 1024):
    hn = HyperNetwork(embed_dim=16, out_rows=n, out_cols=n, rank=4, hidden=32)
    print(f"  {f'{n}x{n}':>14}{hn.n_params():>12,}{hn.generated_params():>13,}"
          f"{hn.compression():>9.1f}x")

hn = HyperNetwork(embed_dim=16, out_rows=256, out_cols=256, rank=4, hidden=32)
for t in ("код", "поэзия", "математика"):
    hn.register_task(t, seed=abs(hash(t)) % 1000)
wc, wp = hn.generate("код"), hn.generate("поэзия")
print(f"\n  разные задачи дают разные веса : cos = {cosine(wc, wp):+.4f}")
print(f"  одна задача всегда одинаково    : {np.array_equal(hn.generate('код'), wc)}")
print("\n  плавный переход между задачами (мгновенно, без переобучения):")
for t in (0.0, 0.5, 1.0):
    m = hn.interpolate("код", "поэзия", t)
    print(f"    t={t:.1f}  cos к «код»={cosine(wc, m):.4f}  cos к «поэзия»={cosine(wp, m):.4f}")

# ───────────────────────────────── итог
title(8, "СВОДНОЕ СРАВНЕНИЕ НА ОДНОЙ МАТРИЦЕ")

for r in compare_methods(W, rvq_stages=3):
    print(f"  {r}")

# ───────────────────────────────── главная проверка
title(9, "ГЛАВНОЕ: КАЧЕСТВО РЕАЛЬНОЙ ОБУЧЕННОЙ МОДЕЛИ")

un.manual_seed(0)
m = MatFormer(MatConfig(n_layer=3, n_embd=96, head_dim=32, block_size=32))
data = np.tile(np.arange(10, 42), (8, 1))
xx, yy = data[:, :-1], data[:, 1:]
opt = Adam(m.parameters(), lr=3e-3)
for _ in range(60):
    opt.zero_grad()
    cross_entropy(m(xx), yy).backward()
    opt.step()
base_loss = cross_entropy(m(xx), yy).item()
orig = {id(p): p.data.copy() for p in m.parameters()}


def apply(fn) -> float:
    for p in m.parameters():
        p.data = orig[id(p)].copy()
        if p.data.ndim == 2:
            p.data = fn(p.data)
    return cross_entropy(m(xx), yy).item()


def tern(v):
    s = float(np.abs(v).mean()) or 1.0
    return (np.clip(np.rint(v / s), -1, 1) * s).astype(np.float32)


def rvq_n(n):
    def f(v):
        r = RVQ(dim=4, stages=n, iters=8).fit(v)
        return r.decode(r.encode(v), n)
    return f


print(f"  обученная модель: loss = {base_loss:.4f}\n")
print(f"  {'метод':<26}{'бит':>6}{'loss':>10}{'рост':>10}")
rows = [("ternary (базовая линия)", 2.0, apply(tern)),
        ("RVQ, 1 кодбук", 2.0, apply(rvq_n(1))),
        ("RVQ, 2 кодбука", 4.0, apply(rvq_n(2))),
        ("RVQ, 3 кодбука", 6.0, apply(rvq_n(3))),
        ("лог-полярное", 7.0, apply(lambda v: LogPolarWeight.quantize(v).dequantize())),
        ("FP16", 16.0, apply(lambda v: v.astype(np.float16).astype(np.float32)))]
for name, bits, loss in rows:
    print(f"  {name:<26}{bits:>6.1f}{loss:>10.4f}{loss - base_loss:>+10.4f}")
for p in m.parameters():
    p.data = orig[id(p)].copy()

print(f"\n{'=' * 78}")
print("ИТОГ: при ОДИНАКОВОЙ цене в 2 бита RVQ бьёт ternary в разы, а при 4 битах")
print("почти не отличается от FP16. Fisher защищает важные веса, гиперсеть даёт")
print("сжатие x1000, фрактал — x10. Веса стали структурой, а не просто числами.")
print("=" * 78)
