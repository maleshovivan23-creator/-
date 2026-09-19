"""Тесты каталога из 48 капсул и движка сборки.

Проверяется соответствие спецификации: состав групп, размеры, «где живёт»,
сценарий «напиши код сортировки» и принцип «что не нужно — не активируется».
"""
from __future__ import annotations

import pytest

from ultranet.proteus.assembly import Assembly, ProteusBody, detect_triggers
from ultranet.proteus.catalog import (BY_KEY, CATALOG, GROUP_COUNTS, CapsuleSpec,
                                      Group, Stage, resolve_deps, tier_rank)
from ultranet.proteus.devices import get_device


class TestCatalogStructure:
    def test_exactly_48_capsules(self):
        """Сводная таблица спецификации: всего 48 капсул."""
        assert len(CATALOG) == 48

    @pytest.mark.parametrize("group,expected", list(GROUP_COUNTS.items()))
    def test_group_sizes(self, group, expected):
        """7 / 4 / 6 / 6 / 6 / 5 / 5 / 4 / 5 — как в таблице."""
        assert len([c for c in CATALOG if c.group is group]) == expected

    def test_nine_groups(self):
        assert len({c.group for c in CATALOG}) == 9

    def test_keys_unique(self):
        keys = [c.key for c in CATALOG]
        assert len(set(keys)) == len(keys)

    def test_names_unique(self):
        names = [c.name for c in CATALOG]
        assert len(set(names)) == len(names)

    def test_every_capsule_has_description(self):
        for c in CATALOG:
            assert c.does and len(c.does) > 10, c.key

    def test_size_ranges_valid(self):
        for c in CATALOG:
            assert 0 < c.min_params <= c.max_params, c.key

    def test_dependencies_exist(self):
        keys = {c.key for c in CATALOG}
        for c in CATALOG:
            for d in c.depends_on:
                assert d in keys, f"{c.key} зависит от несуществующей {d}"

    def test_no_self_dependency(self):
        for c in CATALOG:
            assert c.key not in c.depends_on

    def test_no_dependency_cycles(self):
        """Граф зависимостей должен быть ациклическим."""
        state: dict = {}

        def visit(k: str) -> None:
            if state.get(k) == "done":
                return
            assert state.get(k) != "open", f"цикл через {k}"
            state[k] = "open"
            for d in BY_KEY[k].depends_on:
                visit(d)
            state[k] = "done"

        for c in CATALOG:
            visit(c.key)

    def test_private_capsules_are_sensitive(self):
        """Медицина, право, финансы, память о тебе — помечены как приватные."""
        for key in ("expert_med", "expert_law", "expert_fin",
                    "semantic_mem", "episodic_mem", "safety"):
            assert BY_KEY[key].private, key


class TestSizing:
    def test_weak_device_gets_minimum(self):
        spec = BY_KEY["text_in"]
        assert spec.params_for("small") == spec.min_params

    def test_strong_device_gets_maximum(self):
        spec = BY_KEY["text_in"]
        assert spec.params_for("datacenter") == spec.max_params

    def test_size_grows_with_tier(self):
        spec = BY_KEY["expert_code"]
        sizes = [spec.params_for(t) for t in ("medium", "large", "datacenter")]
        assert sizes == sorted(sizes)

    def test_incompatible_tier_gives_zero(self):
        assert BY_KEY["vision_in"].params_for("micro") == 0

    def test_bytes_follow_params(self):
        spec = BY_KEY["logic"]
        assert spec.bytes_for("medium") == int(spec.params_for("medium") * 2 / 8)

    def test_size_range_is_readable(self):
        assert BY_KEY["steering"].size_range() == "100K–10M"
        assert BY_KEY["expert_sci"].size_range() == "500M–14B"


class TestWhereItLives:
    def test_sensor_capsule_runs_on_micro(self):
        """«Где живёт: часы, кольцо, телефон, ESP32»."""
        assert BY_KEY["sensor_in"].fits_on(get_device("esp32"))

    def test_vision_needs_real_device(self):
        assert not BY_KEY["vision_in"].fits_on(get_device("esp32"))
        assert BY_KEY["vision_in"].fits_on(get_device("phone"))

    def test_science_needs_big_ram(self):
        """«Где живёт: ноутбук, рабочая станция» — но не смартфон."""
        assert not BY_KEY["expert_sci"].fits_on(get_device("phone"))
        assert BY_KEY["expert_sci"].fits_on(get_device("laptop"))
        assert BY_KEY["expert_sci"].fits_on(get_device("workstation"))

    def test_steering_lives_everywhere(self):
        """«Где живёт: везде» — даже на смарт-карте."""
        for key in ("steering", "task", "safety", "router", "early_exit"):
            assert BY_KEY[key].fits_on(get_device("smartcard")), key

    def test_more_capsules_on_stronger_hardware(self):
        counts = [len(ProteusBody(d).available())
                  for d in ("smartcard", "earbuds", "phone", "workstation")]
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]

    def test_workstation_fits_everything(self):
        assert len(ProteusBody("workstation").available()) == 48


class TestTriggers:
    @pytest.mark.parametrize("prompt,trigger", [
        ("напиши функцию сортировки", "код"),
        ("сочини стих про море", "творчество"),
        ("посчитай сколько будет 17*3", "математика"),
        ("объясни почему небо голубое", "рассуждение"),
        ("переведи на японский", "перевод"),
        ("у меня болит голова", "медицина"),
        ("проверь договор аренды", "право"),
        ("посчитай налоги", "финансы"),
        ("нарисуй закат", "изображение"),
        ("открой почту", "действие"),
    ])
    def test_keyword_detection(self, prompt, trigger):
        assert trigger in detect_triggers(prompt)

    def test_text_always_present(self):
        assert "текст" in detect_triggers("что угодно")

    def test_unrelated_prompt_has_few_triggers(self):
        assert detect_triggers("привет") == {"текст"}


class TestAssembly:
    @pytest.fixture
    def body(self):
        return ProteusBody("phone")

    def test_specification_scenario(self, body):
        """Сценарий из спецификации: «напиши код сортировки», шаги 1–11."""
        names = body.assemble("напиши код сортировки").names()
        for expected in ("Текстовая", "Интента", "Задачи", "Контекста",
                         "RAG-поиска", "Кодовая экспертная", "Логики",
                         "Кодовая генерация", "Критики", "Безопасности",
                         "Текстовая генерация"):
            assert expected in names, f"не проснулась капсула «{expected}»"

    def test_specification_scenario_sleepers(self, body):
        """«Что не активировалось: медицинская, юридическая, финансовая,
        творчество, голосовая»."""
        names = body.assemble("напиши код сортировки").names()
        for sleeper in ("Медицинская экспертная", "Юридическая экспертная",
                        "Финансовая экспертная", "Творчества", "Голосовая"):
            assert sleeper not in names, f"капсула «{sleeper}» не должна просыпаться"

    def test_uses_fraction_of_capsules(self, body):
        """«Использовано 10% капсул» — в любом случае заметно меньше всех."""
        a = body.assemble("напиши код сортировки")
        assert a.capsule_fraction < 0.75
        assert a.foreground_fraction < 0.6

    def test_different_prompts_different_experts(self, body):
        def experts(q):
            return {c.name for c in body.assemble(q).active
                    if c.group is Group.EXPERT}
        assert experts("напиши код") != experts("у меня болит голова")
        assert "Кодовая экспертная" in experts("напиши код сортировки")
        assert "Медицинская экспертная" in experts("у меня болит голова")
        assert "Финансовая экспертная" in experts("посчитай налоги за год")

    def test_creative_prompt_wakes_creativity(self, body):
        a = body.assemble("сочини стих про море")
        assert "Творчества" in a.names()
        assert "Кодовая экспертная" not in a.names()

    def test_math_prompt_wakes_math(self, body):
        names = body.assemble("сколько будет 17 умножить на 3").names()
        assert "Математики" in names and "Формальная" in names

    def test_always_on_capsules_always_active(self, body):
        a = body.assemble("привет")
        for key in ("task", "context", "safety", "router", "steering", "profiler"):
            assert BY_KEY[key].name in a.names(), key

    def test_dependencies_are_pulled_in(self, body):
        """Кодовая генерация тянет за собой кодовую экспертную."""
        a = body.assemble("напиши код сортировки")
        names = a.names()
        assert "Кодовая генерация" in names and "Кодовая экспертная" in names

    def test_resolve_deps_transitive(self):
        deps = resolve_deps(["gen_code"])
        assert {"gen_code", "expert_code", "code_in"} <= deps

    def test_pipeline_is_ordered(self, body):
        a = body.assemble("напиши код сортировки")
        stages = [st for st, _ in a.pipeline()]
        assert stages == sorted(stages)
        assert Stage.INPUT in stages and Stage.PRODUCE in stages

    def test_assembly_fits_budget(self, body):
        a = body.assemble("напиши код сортировки")
        assert a.fits and a.bytes_used <= a.budget

    def test_weak_device_drops_heavy_capsules(self):
        weak = ProteusBody("smartwatch").assemble("нарисуй закат")
        strong = ProteusBody("laptop").assemble("нарисуй закат")
        assert "Визуальная генерация" not in weak.names()
        assert "Визуальная генерация" in strong.names()
        assert weak.params < strong.params

    def test_unavailable_reported(self):
        a = ProteusBody("smartwatch").assemble("нарисуй закат")
        assert a.unavailable
        assert all(not c.fits_on(get_device("smartwatch")) for c in a.unavailable)

    def test_micro_device_still_works(self):
        """Даже смарт-карта собирает рабочее тело."""
        a = ProteusBody("smartcard").assemble("включи свет")
        assert a.n_active > 0 and a.fits

    def test_explain_is_informative(self, body):
        text = body.assemble("напиши код сортировки").explain()
        assert "Кодовая экспертная" in text and "капсул" in text

    def test_energy_fraction_below_one(self, body):
        a = body.assemble("открой почту")
        assert 0.0 < a.energy_fraction() < 1.0

    def test_simple_request_cheaper_than_complex(self, body):
        simple = body.assemble("открой почту")
        complex_ = body.assemble("напиши код сортировки и объясни алгоритм")
        assert simple.params < complex_.params

    def test_registry_view(self, body):
        a = body.assemble("напиши код сортировки")
        reg = body.to_registry(a)
        assert len(reg.by_level(reg.root.children[0].level)) == 9   # 9 групп-органов
        assert 0.0 < reg.active_fraction() < 1.0

    def test_inventory_lists_all_groups(self, body):
        inv = body.inventory()
        for g in Group:
            assert g.value in inv
        assert "ВСЕГО" in inv
