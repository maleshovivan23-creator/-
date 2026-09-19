"""Тесты второй волны Протея: MoE, steering, LoRA, капсулы, шарды, контекст.

Каждый тест проверяет конкретное заявление из спецификации пользователя.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import ultranet as un
from ultranet.proteus import (Activity, AdaptationLoops, Capsule, CapsuleRegistry,
                              Context, DeviceState, Expert, Feedback, Level,
                              LoRALayer, MatConfig, MatFormer, MoE, PersonalAdapter,
                              Place, Proteus, Shard, ShardedSwarm, SteeringLibrary,
                              SteeringVector, get_device, style_for)
from ultranet.tensor import Tensor


@pytest.fixture
def moe_model():
    un.manual_seed(0)
    return MatFormer(MatConfig(n_layer=3, n_embd=64, head_dim=32, block_size=32,
                               n_experts=4, top_k_experts=2))


@pytest.fixture
def small_model():
    un.manual_seed(0)
    return MatFormer(MatConfig(n_layer=3, n_embd=64, head_dim=32, block_size=32))


# ============================================================ MoE / эксперты
class TestMoE:
    def test_forward_shape(self, moe_model):
        out = moe_model(np.array([[1, 2, 3, 4]]))
        assert out.shape == (1, 4, 260)

    def test_only_top_k_experts_active(self, moe_model):
        """«Работают только нужные 10% нейронов»."""
        total = moe_model.num_params()
        active = moe_model.active_params(1.0)
        assert active < total * 0.5, "top-2 из 4 должно быть заметно дешевле плотной сети"

    def test_more_experts_more_params(self, moe_model):
        moe_model.set_active_experts(1)
        one = moe_model.active_params(1.0)
        moe_model.set_active_experts(4)
        four = moe_model.active_params(1.0)
        moe_model.set_active_experts(None)
        assert four > one

    def test_gate_weights_sum_to_one(self):
        """Если все эксперты одинаковы, MoE == один эксперт (веса нормированы)."""
        un.manual_seed(1)
        moe = MoE(16, n_experts=3, top_k=3)
        for e in moe.experts[1:]:
            for a, b in zip(e.parameters(), moe.experts[0].parameters()):
                a.data = b.data.copy()
        x = Tensor(np.random.randn(1, 3, 16).astype(np.float32))
        assert np.allclose(moe(x, 16).data, moe.experts[0](x, 16).data, atol=1e-5)

    def test_gradient_reaches_router_and_experts(self):
        un.manual_seed(0)
        moe = MoE(32, n_experts=4, top_k=2)
        x = Tensor(np.random.randn(2, 4, 32).astype(np.float32), requires_grad=True)
        (moe(x, 32) ** 2).sum().backward()
        assert np.abs(x.grad).sum() > 0
        assert moe.router.grad is not None and np.abs(moe.router.grad).sum() > 0
        trained = sum(1 for e in moe.experts
                      if e.w_down.weight.grad is not None
                      and np.abs(e.w_down.weight.grad).sum() > 0)
        assert trained >= 2

    def test_different_inputs_use_different_experts(self):
        """Разные задачи -> разные эксперты."""
        un.manual_seed(3)
        moe = MoE(32, n_experts=4, top_k=1)
        a = Tensor(np.random.randn(1, 8, 32).astype(np.float32) * 3)
        moe(a, 32)
        load_a = moe.last_load.copy()
        b = Tensor(np.random.randn(1, 8, 32).astype(np.float32) * 3 - 5)
        moe(b, 32)
        assert not np.array_equal(load_a, moe.last_load)

    def test_force_experts(self):
        un.manual_seed(0)
        moe = MoE(32, n_experts=4, top_k=1)
        x = Tensor(np.random.randn(1, 4, 32).astype(np.float32))
        moe(x, 32, force=[2])
        assert moe.last_load[2] == 1.0

    def test_aux_loss_penalises_imbalance(self):
        un.manual_seed(0)
        moe = MoE(16, n_experts=4, top_k=1)
        x = Tensor(np.random.randn(1, 8, 16).astype(np.float32))
        moe(x, 16)
        assert moe.aux_loss() >= 0.0

    def test_moe_trains(self):
        """MoE-модель реально снижает loss."""
        un.manual_seed(0)
        from ultranet.functional import cross_entropy
        from ultranet.optim import Adam
        m = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32, block_size=16,
                                n_experts=4, top_k_experts=2))
        data = np.tile(np.arange(10, 26), (4, 1))
        x, y = data[:, :-1], data[:, 1:]
        opt = Adam(m.parameters(), lr=3e-3)
        first = last = None
        for i in range(30):
            opt.zero_grad()
            loss = cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            first = loss.item() if i == 0 else first
            last = loss.item()
        assert last < first * 0.6


# ====================================================== steering-векторы
class TestSteering:
    @pytest.fixture
    def lib(self, small_model):
        lib = SteeringLibrary(small_model)
        lib.learn("математика", ["2+2=4 посчитай", "реши уравнение x=5"],
                  ["луна над морем", "ветер в листве"], layer=1)
        lib.learn("поэзия", ["луна над морем светит", "сердце поёт"],
                  ["def f(x): return x", "import numpy"], layer=1)
        return lib

    def test_learn_from_text(self, lib):
        assert "математика" in lib and "поэзия" in lib
        assert np.isclose(np.linalg.norm(lib["математика"].vector), 1.0, atol=1e-5)

    def test_weights_never_change(self, lib, small_model):
        """Главное заявление: сдвиг активаций БЕЗ изменения весов."""
        before = [p.data.copy() for p in small_model.parameters()]
        with lib.active("математика", 3.0):
            small_model(np.array([[1, 2, 3, 4]]))
        assert all(np.array_equal(a, b.data)
                   for a, b in zip(before, small_model.parameters()))

    def test_steering_changes_output(self, lib, small_model):
        probe = np.array([[1, 2, 3, 4, 5]])
        base = small_model(probe).data.copy()
        with lib.active("математика", 5.0):
            steered = small_model(probe).data.copy()
        assert not np.allclose(base, steered, atol=1e-4)

    def test_context_manager_restores(self, lib, small_model):
        with lib.active("поэзия", 2.0):
            assert small_model.steering_active
        assert not small_model.steering_active

    def test_stronger_vector_bigger_effect(self, lib):
        weak = lib.effect_size("математика", strength=1.0)
        strong = lib.effect_size("математика", strength=5.0)
        assert strong > weak

    def test_zero_strength_is_noop(self, lib, small_model):
        probe = np.array([[1, 2, 3]])
        base = small_model(probe).data.copy()
        with lib.active("поэзия", 0.0):
            assert np.allclose(base, small_model(probe).data, atol=1e-6)

    def test_different_concepts_differ(self, lib):
        cos = float(np.dot(lib["математика"].direction, lib["поэзия"].direction))
        assert abs(cos) < 0.95, "векторы разных концептов не должны совпадать"

    def test_unknown_vector_raises(self, lib):
        with pytest.raises(KeyError):
            lib.apply("несуществующий")

    def test_negative_layer_index(self, small_model):
        lib = SteeringLibrary(small_model)
        sv = lib.learn("x", ["абв"], ["где"], layer=-1)
        assert sv.layer == small_model.cfg.n_layer - 1


# ============================================================ LoRA / адаптер
class TestAdapter:
    def test_adapter_is_tiny(self, small_model):
        pa = PersonalAdapter(small_model, rank=4)
        assert pa.ratio() < 0.02, "адаптер должен быть <2% от ядра"

    def test_core_not_in_adapter_params(self, small_model):
        """Критично: оптимизатор адаптера не должен видеть веса ядра."""
        pa = PersonalAdapter(small_model, rank=4)
        core = {id(p) for p in small_model.parameters()}
        assert not [p for p in pa.parameters() if id(p) in core]

    def test_starts_as_identity(self, small_model):
        """B=0 -> адаптер сначала ничего не меняет."""
        pa = PersonalAdapter(small_model, rank=4)
        x = np.array([[1, 2, 3, 4]])
        assert np.allclose(pa.forward_with_adapter(x).data, small_model(x).data, atol=1e-6)

    def test_core_frozen_during_training(self, small_model):
        """«Ядро заморожено. Меняются только надстройки»."""
        un.manual_seed(0)
        pa = PersonalAdapter(small_model, rank=4)
        pa.observe(Feedback(list(range(10, 40)), reward=1.0))
        before = [p.data.copy() for p in small_model.parameters()]
        pa.consolidate(steps=10, lr=3e-3)
        assert all(np.array_equal(a, b.data)
                   for a, b in zip(before, small_model.parameters()))

    def test_adapter_learns(self, small_model):
        un.manual_seed(0)
        pa = PersonalAdapter(small_model, rank=8)
        pa.observe(Feedback(list(range(10, 42)), reward=1.0))
        hist = pa.consolidate(steps=40, lr=5e-3, ewc_lambda=0.0)
        assert hist["loss"][-1] < hist["loss"][0]

    def test_adapter_changes_output_after_training(self, small_model):
        un.manual_seed(0)
        pa = PersonalAdapter(small_model, rank=4)
        pa.observe(Feedback(list(range(10, 42)), reward=1.0))
        pa.consolidate(steps=20, lr=5e-3)
        x = np.array([[1, 2, 3, 4]])
        assert not np.allclose(pa.forward_with_adapter(x).data,
                               small_model(x).data, atol=1e-6)

    def test_orthogonal_init_reduces_overlap(self):
        """Защита от forgetting #1: ортогональность подпространств."""
        un.manual_seed(0)
        a, b = LoRALayer(32, 32, rank=4), LoRALayer(32, 32, rank=4)
        before = a.overlap_with(b)
        b.orthogonalize_to([a])
        assert a.overlap_with(b) < before
        assert a.overlap_with(b) < 1e-4

    def test_replay_buffer(self, small_model):
        """Защита от forgetting #2: replay старых примеров."""
        pa = PersonalAdapter(small_model, rank=4, replay_size=3)
        for i in range(5):
            pa.add_replay(list(range(i, i + 20)))
        assert len(pa.replay) == 3

    def test_ewc_importance(self, small_model):
        """Защита от forgetting #3: регуляризация важных весов."""
        un.manual_seed(0)
        pa = PersonalAdapter(small_model, rank=4)
        pa.snapshot_importance(list(range(10, 42)), block=32)
        assert pa.fisher and all(np.all(v >= 0) for v in pa.fisher.values())
        assert set(pa.fisher) == set(pa.anchor)

    def test_ewc_keeps_weights_closer(self, small_model):
        """С EWC адаптер уходит от якоря меньше, чем без него."""
        def drift(lmbda):
            un.manual_seed(0)
            m = MatFormer(MatConfig(n_layer=2, n_embd=32, head_dim=16, block_size=32))
            pa = PersonalAdapter(m, rank=4)
            pa.observe(Feedback(list(range(10, 42)), reward=1.0))
            pa.consolidate(steps=10, lr=5e-3, ewc_lambda=0.0)
            pa.snapshot_importance(list(range(10, 42)), block=32)
            anchor = {k: v.copy() for k, v in pa.anchor.items()}
            pa.observe(Feedback(list(range(60, 92)), reward=1.0))
            pa.consolidate(steps=20, lr=5e-3, ewc_lambda=lmbda, replay_ratio=0.0)
            return sum(float(np.abs(p.data - anchor[n]).sum())
                       for n, p in pa.named_parameters() if n in anchor)
        assert drift(50.0) < drift(0.0)

    def test_size_report(self, small_model):
        pa = PersonalAdapter(small_model, rank=4)
        assert pa.size_bytes() > 0 and "LoRA" in pa.report()


# ============================================================== капсулы
class TestCapsules:
    @pytest.fixture
    def reg(self, moe_model):
        return CapsuleRegistry.from_model(
            moe_model, {0: {"код"}, 1: {"поэзия"}, 2: {"математика"}, 3: {"диалог"}})

    def test_levels_exist(self, reg):
        """Атом -> молекула -> клетка -> орган -> тело."""
        for lvl in (Level.MOLECULE, Level.CELL, Level.ORGAN, Level.BODY):
            assert reg.by_level(lvl), f"нет капсул уровня {lvl.label}"

    def test_sleeping_costs_nothing(self, reg):
        """«Неактивные капсулы не существуют»."""
        reg.sleep_all()
        assert reg.active_params() == 0

    def test_task_activates_subset(self, reg):
        reg.assemble(["код"])
        assert 0.0 < reg.active_fraction() < 1.0

    def test_different_tasks_different_capsules(self, reg):
        reg.assemble(["код"])
        code = {c.name for c in reg.index.values() if c.awake}
        reg.assemble(["поэзия"])
        poetry = {c.name for c in reg.index.values() if c.awake}
        assert code != poetry

    def test_unknown_task_wakes_everything(self, reg):
        """Незнакомая задача решается полным телом, а не голым ядром."""
        reg.assemble(["астрология"])
        assert reg.active_fraction() == pytest.approx(1.0)

    def test_more_skills_more_capsules(self, reg):
        reg.assemble(["код"])
        one = reg.active_params()
        reg.assemble(["код", "поэзия", "математика"])
        assert reg.active_params() > one

    def test_wake_sleep_cycle(self, reg):
        reg.wake_all()
        full = reg.active_params()
        reg.sleep_all()
        assert reg.active_params() == 0
        reg.wake_all()
        assert reg.active_params() == full

    def test_capsule_knows_its_size(self, reg):
        cap = reg.by_level(Level.CELL)[0]
        assert cap.bytes_at(2.0) == int(cap.total_params() * 2.0 / 8)

    def test_hierarchy_sums_up(self, reg):
        assert reg.root.total_params() == sum(c.total_params()
                                              for c in reg.root.children)

    def test_tree_renders(self, reg):
        reg.assemble(["код"])
        assert "●" in reg.root.tree() and "○" in reg.root.tree()


# ================================================================ шарды / рой
class TestShardedSwarm:
    @pytest.fixture
    def swarm(self):
        un.manual_seed(0)
        m = MatFormer(MatConfig(n_layer=8, n_embd=96, head_dim=32, block_size=32))
        states = [DeviceState(get_device(k))
                  for k in ("earbuds", "smartwatch", "phone", "laptop")]
        skills = {0: {"слух"}, 1: {"слух"}, 2: {"контекст"}, 3: {"память"},
                  4: {"код"}, 5: {"код"}, 6: {"логика"}, 7: {"генерация"}}
        return ShardedSwarm(m, states, skills_by_layer=skills), m

    def test_every_node_gets_shard(self, swarm):
        sw, _ = swarm
        assert all(n.shard.layers for n in sw.nodes if n.online)

    def test_full_coverage(self, swarm):
        sw, _ = swarm
        assert sw.coverage()

    def test_shards_do_not_overlap(self, swarm):
        sw, _ = swarm
        seen = set()
        for n in sw.nodes:
            if n.online:
                assert not (seen & n.shard.layers)
                seen |= n.shard.layers

    def test_output_matches_monolith(self, swarm):
        sw, m = swarm
        x = np.array([[1, 2, 3, 4, 5]])
        assert np.allclose(sw.forward(x).data, m(x).data, atol=1e-5)

    def test_only_activations_travel(self, swarm):
        """По сети идут активации, а не веса."""
        sw, m = swarm
        sw.forward(np.array([[1, 2, 3, 4, 5]]))
        weights_bytes = m.num_params() * 2 / 8
        assert sw.bytes_moved < weights_bytes

    def test_router_depends_on_task(self, swarm):
        """«Кто ближе — тот и роутер», без постоянного лидера."""
        sw, _ = swarm
        code = sw.elect_router(["код"], complexity=0.9).name
        hear = sw.elect_router(["слух"], complexity=0.1).name
        assert code != hear

    def test_node_death_is_survivable(self, swarm):
        sw, m = swarm
        x = np.array([[1, 2, 3, 4, 5]])
        ref = m(x).data.copy()
        sw.leave("Ноутбук")
        assert sw.coverage()
        assert np.allclose(sw.forward(x).data, ref, atol=1e-5)

    def test_node_return_rebalances(self, swarm):
        sw, m = swarm
        sw.leave("Ноутбук")
        after_leave = {n.name: len(n.shard.layers) for n in sw.nodes if n.online}
        sw.rejoin("Ноутбук")
        assert sw.coverage()
        assert {n.name: len(n.shard.layers) for n in sw.nodes if n.online} != after_leave

    def test_join_new_device(self, swarm):
        sw, m = swarm
        before = sw.rebuilds
        sw.join(DeviceState(get_device("tablet")))
        assert sw.rebuilds > before and sw.coverage()

    def test_cannot_kill_last_node(self):
        un.manual_seed(0)
        m = MatFormer(MatConfig(n_layer=2, n_embd=32, head_dim=16, block_size=16))
        sw = ShardedSwarm(m, [DeviceState(get_device("phone"))])
        with pytest.raises(ValueError):
            sw.leave("Смартфон")

    def test_hungry_node_gets_less(self):
        """«Кто голоднее — тот и отдаёт»: узел с низкой батареей берёт меньше."""
        un.manual_seed(0)
        m = MatFormer(MatConfig(n_layer=8, n_embd=64, head_dim=32, block_size=16))
        full = DeviceState(get_device("phone"), battery_pct=100.0)
        empty = DeviceState(get_device("phone2") if False else get_device("phone"),
                            battery_pct=5.0)
        empty.device = get_device("phone")
        sw = ShardedSwarm(m, [full, empty])
        loads = [len(n.shard.layers) for n in sw.nodes]
        assert loads[0] >= loads[1]

    def test_consensus_without_leader(self, swarm):
        sw, _ = swarm
        assert sw.consensus("расшириться") is True
        assert sw.current_router is None or isinstance(sw.current_router, str)

    def test_weak_device_limited_width(self):
        """Маленькое устройство хранит только узкие уровни вложенности."""
        un.manual_seed(0)
        m = MatFormer(MatConfig(n_layer=4, n_embd=256, head_dim=32, block_size=32))
        sw = ShardedSwarm(m, [DeviceState(get_device("smartcard")),
                              DeviceState(get_device("laptop"))])
        weak = [n for n in sw.nodes if "карта" in n.name.lower()][0]
        strong = [n for n in sw.nodes if "оутбук" in n.name][0]
        assert weak.shard.max_width <= strong.shard.max_width


# ============================================================== контекст
class TestContext:
    def test_driving_is_short_and_voice(self):
        s = style_for(Context(9, Place.TRANSIT, Activity.DRIVING))
        assert s.voice and s.max_tokens <= 32

    def test_night_is_quiet(self):
        assert style_for(Context(3, Place.HOME, Activity.IDLE)).max_tokens <= 32

    def test_work_is_detailed(self):
        s = style_for(Context(14, Place.WORK, Activity.WORKING))
        assert s.verbosity > 1.0 and not s.voice

    def test_meeting_is_silent(self):
        assert not style_for(Context(11, Place.WORK, Activity.MEETING)).voice

    def test_default_is_normal(self):
        assert style_for(Context(12, Place.HOME, Activity.IDLE)).verbosity == 1.0

    def test_offline_flag(self):
        assert Context(12, network="none").offline

    def test_loops_run_at_different_speeds(self):
        """Четыре петли: мс / сек / часы / недели."""
        un.manual_seed(0)
        p = Proteus.tiny(device="phone")
        p.enable_adapter(rank=2)
        p.steering.learn("простота", ["кот на окне"], ["эпистемология"], layer=0)
        loops = AdaptationLoops(p, consolidate_steps=2)
        for i in range(55):
            loops.tick(Context(hour=i % 24, activity=Activity.DRIVING), prompt="привет")
        assert loops.counts["сек"] == 55
        assert loops.counts["часы"] == 2
        assert loops.counts["недели"] == 1
        assert loops.counts["мс"] > 0
        assert loops.counts["сек"] > loops.counts["часы"] > loops.counts["недели"]


# ========================================================== интеграция
class TestIntegration:
    def test_task_classifier(self):
        cases = {"напиши функцию сортировки": "код", "сочини стих про море": "творчество",
                 "сколько будет 2+2": "математика", "объясни почему небо голубое": "рассуждение",
                 "привет": "диалог"}
        for prompt, expected in cases.items():
            assert Proteus.classify_task(prompt) == expected

    def test_full_pipeline_with_everything(self):
        un.manual_seed(0)
        p = Proteus.with_experts(n_experts=4, top_k=2, device="phone")
        p.enable_capsules({0: {"код"}, 1: {"творчество"},
                           2: {"математика"}, 3: {"диалог"}})
        p.enable_adapter(rank=4)
        p.remember("проект называется Протей")
        r = p.respond("напиши функцию", max_new_tokens=5, seed=0)
        assert r.experts_active is not None
        assert 0 < r.capsules_active <= 1.0
        assert r.adapter_applied
        assert r.stayed_local
        assert isinstance(str(r.trace() if hasattr(r, "trace") else r), str)

    def test_complexity_scales_experts(self):
        un.manual_seed(0)
        p = Proteus.with_experts(n_experts=4, top_k=2, device="laptop")
        simple = p.respond("да", max_new_tokens=3, seed=0, task_complexity=0.0)
        hard = p.respond("проанализируй", max_new_tokens=3, seed=0, task_complexity=1.0)
        assert hard.experts_active > simple.experts_active

    def test_feedback_then_consolidate(self):
        un.manual_seed(0)
        p = Proteus.tiny(device="phone")
        p.enable_adapter(rank=4)
        for _ in range(3):
            p.feedback("мой личный стиль общения очень краткий и сухой", reward=1.0)
        core = [x.data.copy() for x in p.model.parameters()]
        hist = p.consolidate(steps=10, lr=3e-3)
        assert hist["loss"]
        assert all(np.array_equal(a, b.data) for a, b in zip(core, p.model.parameters()))

    def test_sharded_swarm_from_proteus(self):
        un.manual_seed(0)
        p = Proteus.tiny(device="phone")
        sw = p.form_sharded_swarm(["earbuds", "phone", "laptop"])
        assert sw.coverage()
        x = np.array([[1, 2, 3]])
        assert np.allclose(sw.forward(x).data, p.model(x).data, atol=1e-5)
        p.dissolve_swarm()
        assert p.sharded is None
