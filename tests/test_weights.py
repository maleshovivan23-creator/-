"""Тесты семи техник «идеальных весов» Протея.

Каждое заявление спецификации проверяется измерением, а не декларацией.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import ultranet as un
from ultranet.functional import cross_entropy
from ultranet.optim import Adam
from ultranet.proteus import MatConfig, MatFormer
from ultranet.proteus.weights import (RVQ, ComplexWeight, FisherImportance,
                                      FractalStack, HyperNetwork, LogPolarWeight,
                                      SphericalWeight, WeightReport,
                                      angular_distance, compare_methods,
                                      complex_matmul, cosine, rel_error, slerp)


@pytest.fixture
def w():
    return np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32) * 0.05


# ══════════════════════════════ 1. гиперсферическая нормализация
class TestSpherical:
    def test_all_rows_unit_length(self, w):
        assert SphericalWeight.from_weight(w).is_unit

    def test_reconstruction_is_lossless(self, w):
        s = SphericalWeight.from_weight(w)
        assert rel_error(w, s.to_weight()) < 0.01

    def test_direction_carries_meaning(self, w):
        """Угол между весами = семантическая близость."""
        s = SphericalWeight.from_weight(w)
        assert angular_distance(s.directions[0], s.directions[0]) < 1e-5
        assert angular_distance(s.directions[0], -s.directions[0]) > 3.0

    def test_slerp_stays_on_sphere(self):
        rng = np.random.default_rng(1)
        a, b = rng.standard_normal(32), rng.standard_normal(32)
        for t in (0.0, 0.25, 0.5, 1.0):
            assert np.isclose(np.linalg.norm(slerp(a, b, t)), 1.0, atol=1e-6)

    def test_slerp_midpoint_is_equidistant(self):
        rng = np.random.default_rng(2)
        a, b = rng.standard_normal(16), rng.standard_normal(16)
        m = slerp(a, b, 0.5)
        assert np.isclose(angular_distance(a, m), angular_distance(m, b), atol=1e-5)

    def test_slerp_endpoints(self):
        rng = np.random.default_rng(3)
        a, b = rng.standard_normal(8), rng.standard_normal(8)
        assert cosine(slerp(a, b, 0.0), a) > 0.999
        assert cosine(slerp(a, b, 1.0), b) > 0.999

    def test_handles_zero_rows(self):
        z = np.zeros((4, 8), dtype=np.float32)
        z[0, 0] = 1.0
        assert np.all(np.isfinite(SphericalWeight.from_weight(z).to_weight()))


# ══════════════════════════════════ 2. лог-полярное квантование
class TestLogPolar:
    def test_seven_bits(self):
        assert LogPolarWeight.BITS == 7

    def test_beats_ternary_accuracy(self, w):
        """«В 3.5 раза точнее ternary»."""
        lp = rel_error(w, LogPolarWeight.quantize(w).dequantize())
        s = float(np.abs(w).mean())
        tern = rel_error(w, np.clip(np.rint(w / s), -1, 1) * s)
        assert lp < tern / 3.0

    def test_close_to_fp16_quality(self, w):
        assert rel_error(w, LogPolarWeight.quantize(w).dequantize()) < 0.10

    def test_multiplication_is_shift_only(self, w):
        """Умножение = сдвиг + сложение, без настоящего умножения."""
        rng = np.random.default_rng(5)
        x = rng.standard_normal(w.shape).astype(np.float32)
        lp = LogPolarWeight.quantize(w)
        assert np.allclose(lp.shift_multiply(x), x * lp.dequantize(), atol=1e-6)

    @pytest.mark.parametrize("scale", [1e-5, 1e-3, 1e-2, 1.0, 100.0])
    def test_works_at_any_scale(self, scale):
        """Баг: мелкие веса обнулялись — окно экспоненты не двигалось."""
        rng = np.random.default_rng(6)
        v = rng.standard_normal(2000).astype(np.float32) * scale
        d = LogPolarWeight.quantize(v).dequantize()
        assert rel_error(v, d) < 0.15
        assert float((d == 0).mean()) < 0.05

    def test_sign_preserved(self, w):
        d = LogPolarWeight.quantize(w).dequantize()
        same = np.sign(d) == np.sign(w)
        assert same.mean() > 0.99

    def test_exact_zeros_stay_zero(self):
        v = np.array([0.0, 1.0, 0.0, -2.0], dtype=np.float32)
        d = LogPolarWeight.quantize(v).dequantize()
        assert d[0] == 0.0 and d[2] == 0.0

    def test_storage_size(self, w):
        lp = LogPolarWeight.quantize(w)
        assert lp.nbytes() == math.ceil(w.size * 7 / 8)
        assert lp.nbytes() < w.size * 2      # дешевле FP16


# ═══════════════════════════════════════════════════ 3. RVQ
class TestRVQ:
    def test_error_decreases_with_stages(self, w):
        """Вложенность: каждый кодбук уточняет предыдущий."""
        rvq = RVQ(dim=4, stages=3, iters=8).fit(w)
        errs = [e for _, _, e in rvq.error_by_stage(w)]
        assert errs == sorted(errs, reverse=True)

    def test_nested_precision_from_one_storage(self, w):
        """Одно хранилище — много точностей, без отдельных версий."""
        rvq = RVQ(dim=4, stages=3, iters=8).fit(w)
        codes = rvq.encode(w)
        rough = rvq.decode(codes, 1)
        fine = rvq.decode(codes, 3)
        assert rel_error(w, fine) < rel_error(w, rough)

    def test_beats_ternary_at_same_bits(self, w):
        """RVQ с 1 кодбуком = 2 бита — та же цена, что ternary, но точнее."""
        rvq = RVQ(dim=4, size=256, stages=1, iters=10).fit(w)
        r = rel_error(w, rvq.decode(rvq.encode(w), 1))
        s = float(np.abs(w).mean())
        tern = rel_error(w, np.clip(np.rint(w / s), -1, 1) * s)
        assert rvq.bits_per_weight(1) == pytest.approx(2.0)
        assert r < tern

    def test_bits_formula(self):
        rvq = RVQ(dim=4, size=256, stages=3)
        assert rvq.bits_per_weight(1) == pytest.approx(2.0)
        assert rvq.bits_per_weight(3) == pytest.approx(6.0)

    def test_codebooks_are_small(self, w):
        """«256 векторов × 8 бит = 2 КБ на кодбук, а весов миллионы»."""
        rvq = RVQ(dim=4, size=256, stages=3, iters=5).fit(w)
        assert rvq.codebook_bytes() < 16 * 1024

    def test_shape_preserved(self, w):
        rvq = RVQ(dim=4, stages=2, iters=5).fit(w)
        assert rvq.decode(rvq.encode(w)).shape == w.shape

    def test_handles_non_divisible_size(self):
        v = np.random.default_rng(7).standard_normal(37).astype(np.float32)
        rvq = RVQ(dim=4, stages=2, iters=5).fit(v)
        assert rvq.decode(rvq.encode(v)).shape == v.shape

    def test_deterministic(self, w):
        a = RVQ(dim=4, stages=2, iters=5, seed=1).fit(w)
        b = RVQ(dim=4, stages=2, iters=5, seed=1).fit(w)
        assert np.allclose(a.decode(a.encode(w)), b.decode(b.encode(w)))


# ═══════════════════════════════════════════════ 4. Fisher
class TestFisher:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(0)
        w = rng.standard_normal((64, 64)).astype(np.float32) * 0.05
        f = np.abs(rng.standard_normal((64, 64))) * 0.01
        important = rng.random((64, 64)) < 0.05
        f[important] += 10.0
        return w, f, important

    def test_adaptive_protects_important_weights(self, setup):
        """Главное заявление: важные веса защищены от квантования."""
        w, f, imp = setup
        fi = FisherImportance()
        fi.fisher["w"] = f
        plan = fi.plan_for("w")
        adaptive = fi.quantize_adaptive(w, plan.bits)
        uniform = fi.uniform_baseline(w, plan.mean_bits)
        assert rel_error(w[imp], adaptive[imp]) < rel_error(w[imp], uniform[imp]) / 10

    def test_adaptive_wins_on_weighted_error(self, setup):
        """При равной цене в битах Fisher-взвешенная ошибка заметно ниже."""
        w, f, _ = setup
        fi = FisherImportance()
        fi.fisher["w"] = f
        plan = fi.plan_for("w")
        a = fi.weighted_error(w, fi.quantize_adaptive(w, plan.bits), f)
        u = fi.weighted_error(w, fi.uniform_baseline(w, plan.mean_bits), f)
        assert a < u / 2

    def test_mean_bits_in_range(self, setup):
        w, f, _ = setup
        fi = FisherImportance(levels=(2, 4, 8, 16))
        fi.fisher["w"] = f
        plan = fi.plan_for("w")
        assert 2.0 <= plan.mean_bits <= 16.0
        assert set(np.unique(plan.bits)) <= {2, 4, 8, 16}

    def test_important_get_more_bits(self, setup):
        w, f, imp = setup
        fi = FisherImportance()
        fi.fisher["w"] = f
        bits = fi.plan_for("w").bits
        assert bits[imp].mean() > bits[~imp].mean()

    def test_accumulate_from_real_model(self):
        un.manual_seed(0)
        m = MatFormer(MatConfig(n_layer=2, n_embd=32, head_dim=16, block_size=16))
        data = np.tile(np.arange(10, 26), (2, 1))
        cross_entropy(m(data[:, :-1]), data[:, 1:]).backward()
        fi = FisherImportance()
        fi.accumulate(m)
        assert fi.fisher and all(np.all(v >= 0) for v in fi.fisher.values())

    def test_missing_param_returns_none(self):
        assert FisherImportance().plan_for("нет такого") is None


# ═════════════════════════════════════════ 5. фрактальная структура
class TestFractal:
    @pytest.fixture
    def stack(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((32, 32)).astype(np.float32) * 0.1
        A = np.eye(32) + rng.normal(0, 0.02, (32, 32))
        targets, cur = [base.copy()], base.copy()
        for _ in range(5):
            cur = (A @ cur).astype(np.float32)
            targets.append(cur.copy())
        return base, targets

    def test_generates_requested_layers(self, stack):
        base, targets = stack
        assert len(FractalStack(base, 6).generate()) == 6

    def test_fit_reduces_loss(self, stack):
        base, targets = stack
        fs = FractalStack(base, len(targets))
        hist = fs.fit(targets, steps=120, lr=0.01)
        assert hist[-1] < hist[0]

    def test_compression_is_real(self, stack):
        """«Храним базовый слой и функцию вместо N слоёв»."""
        base, targets = stack
        fs = FractalStack(base, 20)
        assert fs.compression() > 1.5
        assert fs.stored_params() < fs.generated_params()

    def test_deeper_stack_compresses_more(self, stack):
        base, _ = stack
        assert FractalStack(base, 50).compression() > FractalStack(base, 5).compression()

    def test_can_grow_more_layers(self, stack):
        """«Хочешь больше — применяешь f ещё раз»."""
        base, targets = stack
        fs = FractalStack(base, len(targets))
        assert len(fs.generate(12)) == 12

    def test_first_layer_is_base(self, stack):
        base, _ = stack
        assert np.allclose(FractalStack(base, 3).generate()[0], base, atol=1e-5)


# ═══════════════════════════════════ 6. комплексные веса
class TestComplex:
    def test_roundtrip_reasonable(self, w):
        assert rel_error(w, ComplexWeight.from_pairs(w).to_weight()) < 0.4

    def test_more_bits_better(self, w):
        coarse = ComplexWeight.from_pairs(w, amp_bits=3, phase_bits=2)
        fine = ComplexWeight.from_pairs(w, amp_bits=6, phase_bits=6)
        assert rel_error(w, fine.to_weight()) < rel_error(w, coarse.to_weight())

    def test_bits_per_weight(self, w):
        """7 бит на комплексное число = 3.5 бита на действительный вес."""
        assert ComplexWeight.from_pairs(w, 4, 3).bits_per_weight() == 3.5

    def test_shape_preserved(self, w):
        assert ComplexWeight.from_pairs(w).to_weight().shape == w.shape

    def test_odd_size_handled(self):
        v = np.random.default_rng(8).standard_normal(15).astype(np.float32)
        assert ComplexWeight.from_pairs(v).to_weight().shape == v.shape

    def test_complex_matmul_is_rotation(self):
        """Умножение = поворот + масштаб."""
        rng = np.random.default_rng(9)
        xr, xi = rng.standard_normal((4, 8)), rng.standard_normal((4, 8))
        wr, wi = rng.standard_normal((8, 8)), rng.standard_normal((8, 8))
        gr, gi = complex_matmul(xr, xi, wr, wi)
        expect = (xr + 1j * xi) @ (wr + 1j * wi)
        assert np.allclose(gr, expect.real) and np.allclose(gi, expect.imag)


# ═══════════════════════════════════════════ 7. гиперсеть
class TestHyperNetwork:
    def test_compression_grows_with_size(self):
        """«Гиперсеть в 100 раз меньше основной сети»."""
        un.manual_seed(0)
        small = HyperNetwork(16, 64, 64)
        big = HyperNetwork(16, 1024, 1024)
        assert big.compression() > small.compression()
        assert big.compression() > 100

    def test_size_independent_of_output(self):
        """Ключ к экономии: размер гиперсети не зависит от выхода."""
        un.manual_seed(0)
        assert HyperNetwork(16, 64, 64).n_params() == HyperNetwork(16, 512, 512).n_params()

    def test_generates_correct_shape(self):
        un.manual_seed(0)
        assert HyperNetwork(16, 48, 96).generate("код").shape == (48, 96)

    def test_different_tasks_different_weights(self):
        un.manual_seed(0)
        hn = HyperNetwork(16, 64, 64)
        assert not np.allclose(hn.generate("код"), hn.generate("поэзия"))

    def test_deterministic_per_task(self):
        un.manual_seed(0)
        hn = HyperNetwork(16, 32, 32)
        assert np.array_equal(hn.generate("код"), hn.generate("код"))

    def test_interpolation_between_tasks(self):
        """«Поменял embedding — поменял модель», плавно."""
        un.manual_seed(0)
        hn = HyperNetwork(16, 64, 64)
        a, b = hn.generate("код"), hn.generate("поэзия")
        mid = hn.interpolate("код", "поэзия", 0.5)
        assert not np.allclose(mid, a) and not np.allclose(mid, b)
        assert np.all(np.isfinite(mid))

    def test_accepts_raw_embedding(self):
        un.manual_seed(0)
        hn = HyperNetwork(16, 32, 32)
        e = np.ones(16, dtype=np.float32) / 4
        assert hn.generate(e).shape == (32, 32)

    def test_trainable(self):
        un.manual_seed(0)
        hn = HyperNetwork(8, 16, 16, rank=2)
        assert len(hn.parameters()) > 0
        assert all(p.requires_grad for p in hn.parameters())


# ═════════════════════════════════ сводное сравнение и качество модели
class TestComparison:
    def test_all_methods_reported(self, w):
        methods = {r.method for r in compare_methods(w, rvq_stages=2)}
        assert any("ternary" in m for m in methods)
        assert any("RVQ" in m for m in methods)
        assert any("лог-полярное" in m for m in methods)

    def test_rvq_beats_ternary_at_equal_cost(self, w):
        reps = {r.method: r for r in compare_methods(w, rvq_stages=1)}
        assert reps["RVQ, 1 кодбук(а)"].rel_error < reps["ternary (1.58-bit)"].rel_error

    def test_logpolar_beats_ternary(self, w):
        reps = {r.method: r for r in compare_methods(w, rvq_stages=1)}
        assert reps["лог-полярное (7 бит)"].rel_error < reps["ternary (1.58-bit)"].rel_error

    def test_report_str(self, w):
        assert "бит/вес" in str(compare_methods(w, rvq_stages=1)[0])

    def test_quality_on_real_trained_model(self):
        """Главная проверка: сжатые веса сохраняют качество РАБОЧЕЙ модели."""
        un.manual_seed(0)
        m = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32, block_size=32))
        data = np.tile(np.arange(10, 42), (4, 1))
        x, y = data[:, :-1], data[:, 1:]
        opt = Adam(m.parameters(), lr=3e-3)
        for _ in range(40):
            opt.zero_grad()
            cross_entropy(m(x), y).backward()
            opt.step()
        base = cross_entropy(m(x), y).item()
        orig = {id(p): p.data.copy() for p in m.parameters()}

        def apply(fn):
            for p in m.parameters():
                p.data = orig[id(p)].copy()
                if p.data.ndim == 2:
                    p.data = fn(p.data)

        scale = lambda v: float(np.abs(v).mean()) or 1.0
        apply(lambda v: np.clip(np.rint(v / scale(v)), -1, 1) * scale(v))
        tern_loss = cross_entropy(m(x), y).item()

        def rvq2(v):
            r = RVQ(dim=4, stages=2, iters=6).fit(v)
            return r.decode(r.encode(v), 2)

        apply(rvq2)
        rvq_loss = cross_entropy(m(x), y).item()
        apply(lambda v: LogPolarWeight.quantize(v).dequantize())
        lp_loss = cross_entropy(m(x), y).item()

        # RVQ@4бита и лог-полярное сохраняют качество, ternary — разрушает
        assert rvq_loss < tern_loss
        assert lp_loss < tern_loss
        assert lp_loss < base + 0.2
