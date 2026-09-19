"""Тесты пути Протея 0% → 100% и автоматического роста под железо."""
from __future__ import annotations

import pytest

from ultranet.proteus.catalog import CATALOG, BY_KEY
from ultranet.proteus.growth import (LADDER, GrowingProteus, GrowthState,
                                     HardwareProfile, ladder_table, next_stage,
                                     probe_hardware, stage_at)

GB = 1024 ** 3


def grown(device: str, rounds: int = 30, nodes: tuple = ()) -> GrowingProteus:
    p = GrowingProteus.install_on(device)
    for n in nodes:
        p.add_node(n)
    for _ in range(rounds):
        p.use(requests=300, documents=80_000)
    return p


class TestProbe:
    def test_real_probe_works(self):
        """Зонд должен работать на настоящей машине, а не только в симуляции."""
        hp = probe_hardware()
        assert hp.cpu_count >= 1
        assert hp.ram_bytes > 0
        assert hp.max_params() > 0
        assert hp.tier in ("micro", "small", "medium", "large", "datacenter")

    def test_probe_report_readable(self):
        r = probe_hardware().report()
        assert "RAM" in r and "параметров" in r

    def test_simulated_probe_matches_device(self):
        hp = probe_hardware("phone")
        assert hp.ram_bytes == 8 * GB
        assert hp.tier == "medium"

    def test_budget_never_exceeds_available(self):
        hp = probe_hardware("phone")
        assert hp.weight_budget() <= hp.ram_available

    def test_bigger_ram_more_params(self):
        assert (probe_hardware("laptop").max_params()
                > probe_hardware("smartwatch").max_params())

    def test_context_window_scales(self):
        assert (probe_hardware("server").context_window()
                >= probe_hardware("esp32").context_window())

    def test_profile_converts_to_device(self):
        dev = probe_hardware("phone").as_device()
        assert dev.ram_bytes == 8 * GB


class TestLadder:
    def test_nine_stages(self):
        assert len(LADDER) == 9
        assert [s.pct for s in LADDER] == [0, 1, 5, 10, 25, 50, 75, 90, 100]

    @pytest.mark.parametrize("pct,capsules", [(0, 0), (1, 1), (5, 7), (10, 12),
                                              (25, 25), (50, 35), (75, 45),
                                              (90, 48), (100, 48)])
    def test_capsule_counts_match_spec(self, pct, capsules):
        """Сводная таблица спецификации."""
        assert stage_at(pct).capsules == capsules

    def test_stage_lookup(self):
        assert stage_at(0).title == "Абсолютный ноль"
        assert stage_at(7).pct == 5
        assert stage_at(100).title == "Ультимативная форма"

    def test_next_stage(self):
        assert next_stage(0).pct == 1
        assert next_stage(100) is None

    def test_table_renders(self):
        assert "капсул" in ladder_table()


class TestZeroToInstall:
    def test_zero_is_nothing(self):
        """0%: нет модели, нет капсул, нет памяти."""
        p = GrowingProteus.bare()
        assert p.percent == 0.0
        assert p.active_capsules() == []
        assert p.active_params() == 0
        assert p.stage.title == "Абсолютный ноль"

    def test_one_percent_is_only_probe(self):
        """1%: просыпается ТОЛЬКО капсула профилирования."""
        p = GrowingProteus.bare()
        p.probe(simulate="phone")
        assert p.percent == pytest.approx(1.0)
        caps = p.active_capsules()
        assert len(caps) == 1 and caps[0].key == "profiler"

    def test_five_percent_base_build(self):
        """5%: базовая сборка — 7 капсул, около 100M параметров."""
        p = GrowingProteus.install_on("phone")
        assert p.percent == pytest.approx(5.0)
        assert len(p.active_capsules()) == 7
        assert p.active_params() < 300_000_000

    def test_base_build_has_essentials(self):
        p = GrowingProteus.install_on("phone")
        keys = {c.key for c in p.active_capsules()}
        assert {"profiler", "text_in", "gen_text"} <= keys

    def test_install_probes_automatically(self):
        p = GrowingProteus.bare().install()
        assert p.profile is not None and p.percent >= 5.0


class TestGrowth:
    def test_growth_is_monotonic(self):
        p = GrowingProteus.install_on("laptop")
        seen = [p.percent]
        for _ in range(12):
            p.use(requests=100, documents=5000)
            seen.append(p.percent)
        assert seen == sorted(seen)

    def test_capsules_grow_with_usage(self):
        p = GrowingProteus.install_on("laptop")
        start = len(p.active_capsules())
        for _ in range(20):
            p.use(requests=200, documents=50_000)
        assert len(p.active_capsules()) > start

    def test_params_grow_with_usage(self):
        p = GrowingProteus.install_on("laptop")
        start = p.active_params()
        for _ in range(20):
            p.use(requests=200, documents=50_000)
        assert p.active_params() > start

    def test_maturity_scales_capsule_size(self):
        """Растёт не только число капсул, но и каждая капсула."""
        young = GrowingProteus.install_on("laptop")
        old = grown("laptop")
        spec = BY_KEY["gen_text"]
        assert old.capsule_params(spec) > young.capsule_params(spec)

    def test_nodes_increase_growth(self):
        solo = grown("phone")
        swarm = grown("phone", nodes=("laptop", "smartwatch", "earbuds",
                                      "tablet", "esp32"))
        assert swarm.percent > solo.percent

    def test_full_swarm_reaches_hundred(self):
        p = grown("phone", nodes=("laptop", "smartwatch", "earbuds",
                                  "tablet", "esp32"))
        assert p.percent == pytest.approx(100.0)
        assert len(p.active_capsules()) == 48

    def test_history_recorded(self):
        p = grown("phone", rounds=10)
        assert len(p.history) > 2
        assert "зонд" in p.timeline()

    def test_axes_bar(self):
        p = grown("phone", rounds=5)
        bar = p.axes_bar()
        for axis in ("железо", "использование", "знания", "рой"):
            assert axis in bar

    def test_duplicate_node_ignored(self):
        p = GrowingProteus.install_on("phone")
        p.add_node("laptop")
        n = p.state.nodes
        p.add_node("laptop")
        assert p.state.nodes == n

    def test_node_can_leave(self):
        p = GrowingProteus.install_on("phone")
        p.add_node("laptop")
        p.remove_node("Ноутбук")
        assert p.state.nodes == 1

    def test_cannot_remove_last_node(self):
        p = GrowingProteus.install_on("phone")
        p.remove_node(p.nodes[0])
        assert p.state.nodes == 1


class TestHardwareLimits:
    """Главное требование: модель растёт под железо и никогда не перерастает."""

    @pytest.mark.parametrize("device", ["smartcard", "esp32", "ring", "earbuds",
                                        "smartwatch", "tv", "phone", "laptop",
                                        "workstation", "server"])
    def test_never_exceeds_hardware(self, device):
        p = grown(device, rounds=25)
        assert p.fits_hardware(), f"{device}: {p.memory_bytes()} > {p.total_budget()}"

    @pytest.mark.parametrize("device", ["esp32", "smartwatch", "phone", "server"])
    def test_growth_terminates(self, device):
        """Рост обязан упереться в потолок, а не расти бесконечно."""
        p = GrowingProteus.install_on(device)
        for _ in range(60):
            p.use(requests=500, documents=100_000)
        assert p.percent <= 100.0
        assert p.percent <= p.ceiling() + 1e-6

    def test_weak_hardware_has_low_ceiling(self):
        """ESP32 не может стать «полным роем»."""
        assert grown("esp32").ceiling() < 50.0
        assert grown("smartcard").ceiling() < 25.0

    def test_strong_hardware_high_ceiling(self):
        assert grown("laptop", nodes=("phone",)).ceiling() == pytest.approx(100.0)

    def test_ceiling_ordered_by_ram(self):
        ceilings = [grown(d, rounds=20).ceiling()
                    for d in ("esp32", "earbuds", "smartwatch", "phone", "laptop")]
        assert ceilings == sorted(ceilings)

    def test_swarm_raises_weak_node_ceiling(self):
        """Часы в паре с ноутбуком умеют то же, что ноутбук."""
        solo = grown("smartwatch")
        paired = grown("smartwatch", nodes=("laptop",))
        assert paired.ceiling() > solo.ceiling()
        assert len(paired.active_capsules()) > len(solo.active_capsules())

    def test_size_follows_real_ram_not_tier(self):
        """Smart TV (4 ГБ) и ноутбук (32 ГБ) — оба medium, но размеры разные."""
        spec = BY_KEY["gen_text"]
        from ultranet.proteus.devices import get_device
        assert spec.params_for(get_device("laptop")) > spec.params_for(get_device("tv"))

    def test_micro_device_still_runs(self):
        p = grown("esp32")
        assert p.percent >= 1.0 and p.active_capsules()

    def test_status_reports_fit(self):
        assert "влезает" in grown("phone", rounds=5).status()


class TestGrowthState:
    def test_percent_zero_before_probe(self):
        assert GrowthState().percent() == 0.0

    def test_percent_one_after_probe(self):
        assert GrowthState(probed=True).percent() == 1.0

    def test_percent_five_after_install(self):
        assert GrowthState(probed=True, installed=True).percent() == pytest.approx(5.0)

    def test_axes_bounded(self):
        st = GrowthState(probed=True, installed=True, requests=10 ** 9,
                         documents=10 ** 9, adapters=10 ** 6, vectors=10 ** 6,
                         nodes=100)
        assert all(0.0 <= v <= 1.0 for v in st.axes().values())
        assert st.percent() <= 100.0

    def test_log_progress_is_front_loaded(self):
        """Первые документы дают больше прогресса, чем сотые."""
        st = GrowthState._log_frac
        assert st(10, 1000) - st(0, 1000) > st(1000, 1000) - st(990, 1000)
