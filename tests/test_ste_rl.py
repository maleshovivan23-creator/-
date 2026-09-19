"""Тесты STE-обучения квантованной сети и RL-мета-контроллера."""
import numpy as np
import pytest

import ultranet as un
from ultranet.optim import Adam
from ultranet.tensor import Tensor
from ultranet.proteus import MatConfig, MatFormer, DeviceState, get_device
from ultranet.proteus.quant import TernaryLinear, ste_round
from ultranet.proteus.controller import RLMetaController


def _states():
    return [
        DeviceState(get_device("phone")),
        DeviceState(get_device("phone"), battery_pct=8.0),
        DeviceState(get_device("phone"), temperature_c=88.0),
        DeviceState(get_device("laptop")),
        DeviceState(get_device("smartwatch")),
    ]


def _model():
    return MatFormer(MatConfig(n_layer=6, n_embd=128, head_dim=32, block_size=32))


class TestSTE:
    def test_forward_uses_only_ternary_values(self):
        un.manual_seed(0)
        layer = TernaryLinear(16, 8)
        assert layer.ternary_fraction() == 1.0

    def test_quantized_weight_values_are_three_levels(self):
        un.manual_seed(0)
        layer = TernaryLinear(32, 16)
        w = layer.quantized_weight().data
        assert len(np.unique(np.round(w, 6))) <= 3

    def test_gradient_reaches_shadow_weights(self):
        un.manual_seed(0)
        layer = TernaryLinear(16, 8)
        x = Tensor(np.random.randn(4, 16).astype(np.float32))
        (layer(x) ** 2).sum().backward()
        assert layer.weight.grad is not None
        assert np.abs(layer.weight.grad).sum() > 0

    def test_gradient_is_not_blocked_by_rounding(self):
        """STE: градиент по округлённому значению равен градиенту по исходному."""
        un.manual_seed(0)
        x = Tensor(np.array([[0.3, -0.7, 0.1]], dtype=np.float32), requires_grad=True)
        (ste_round(x) * Tensor(np.ones((1, 3), dtype=np.float32))).sum().backward()
        assert np.allclose(x.grad, 1.0)

    def test_ste_round_output_is_quantized(self):
        x = Tensor(np.array([[0.3, -0.7, 0.1]], dtype=np.float32))
        vals = np.unique(ste_round(x, levels=3).data)
        assert set(np.round(vals, 6)).issubset({-1.0, 0.0, 1.0})

    def test_bits_per_weight_is_log2_of_three(self):
        assert TernaryLinear(8, 8).bits_per_weight() == pytest.approx(1.585, abs=1e-3)

    def test_network_trains_in_ternary_form(self):
        un.manual_seed(0)
        net = [TernaryLinear(8, 32), TernaryLinear(32, 8)]
        opt = Adam([p for m in net for p in m.parameters()], lr=0.02)
        X = np.random.randn(64, 8).astype(np.float32)
        Y = X @ np.random.randn(8, 8).astype(np.float32)
        first = last = None
        for i in range(150):
            opt.zero_grad()
            loss = ((net[1](net[0](Tensor(X)).silu()) - Tensor(Y)) ** 2).mean()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        assert last < first / 5

    def test_weights_stay_ternary_after_training(self):
        un.manual_seed(0)
        layer = TernaryLinear(8, 8)
        opt = Adam(layer.parameters(), lr=0.05)
        X = Tensor(np.random.randn(16, 8).astype(np.float32))
        for _ in range(30):
            opt.zero_grad()
            (layer(X) ** 2).mean().backward()
            opt.step()
        assert layer.ternary_fraction() == 1.0

    def test_bias_option_participates(self):
        un.manual_seed(0)
        layer = TernaryLinear(8, 4, bias=True)
        x = Tensor(np.random.randn(2, 8).astype(np.float32))
        layer(x).sum().backward()
        assert layer.bias.grad is not None


class TestRLMetaController:
    def test_policy_is_a_distribution(self):
        rl = RLMetaController(_model(), seed=0)
        p = rl.policy(_states()[0], 0.5)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()

    def test_policy_never_picks_action_that_does_not_fit(self):
        rl = RLMetaController(_model(), seed=0)
        state = DeviceState(get_device("smartcard"))
        p = rl.policy(state, 0.5)
        for i, prob in enumerate(p):
            w, layers = rl.actions[i]
            if prob > 1e-6:
                assert (rl.base.weight_bytes(w, layers) <= state.available_bytes
                        or i == 0)

    def test_reward_penalises_latency_and_energy(self):
        """Тот же размер модели на севшей батарее стоит дороже."""
        rl = RLMetaController(_model(), seed=0)
        full = DeviceState(get_device("phone"))
        empty = DeviceState(get_device("phone"), battery_pct=5.0)
        biggest = len(rl.actions) - 1
        assert rl.reward(biggest, empty, 0.5) < rl.reward(biggest, full, 0.5)

    @pytest.mark.slow
    def test_training_improves_reward(self):
        rl = RLMetaController(_model(), lr=0.3, seed=0)
        states = _states()
        before = rl.mean_reward(states)
        rl.train(states, steps=4000)
        assert rl.mean_reward(states) > before

    @pytest.mark.slow
    def test_training_reaches_most_of_the_optimum(self):
        rl = RLMetaController(_model(), lr=0.3, seed=0)
        states = _states()
        optimum = np.mean([max(rl.reward(i, st, cx) for i in range(len(rl.actions)))
                           for st in states for cx in (0.1, 0.5, 0.9)])
        rl.train(states, steps=6000)
        assert rl.mean_reward(states) >= 0.93 * optimum

    @pytest.mark.slow
    def test_learned_policy_beats_random_policy(self):
        rl = RLMetaController(_model(), lr=0.3, seed=0)
        states = _states()
        rng = np.random.default_rng(0)
        random_reward = np.mean([rl.reward(int(rng.integers(len(rl.actions))), st, cx)
                                 for st in states for cx in (0.1, 0.5, 0.9)
                                 for _ in range(20)])
        rl.train(states, steps=4000)
        assert rl.mean_reward(states) > random_reward

    @pytest.mark.slow
    def test_hot_device_gets_smaller_model_than_cool_one(self):
        rl = RLMetaController(_model(), lr=0.3, seed=0)
        rl.train(_states(), steps=6000)
        cool = rl.plan(DeviceState(get_device("phone")), 0.5)
        hot = rl.plan(DeviceState(get_device("phone"), temperature_c=95.0), 0.5)
        assert hot.active_params <= cool.active_params

    @pytest.mark.slow
    def test_plan_respects_memory_budget(self):
        rl = RLMetaController(_model(), lr=0.3, seed=0)
        states = _states()
        rl.train(states, steps=3000)
        for st in states:
            plan = rl.plan(st, 0.7)
            assert plan.weight_bytes <= st.available_bytes

    def test_training_is_deterministic_for_a_seed(self):
        states = _states()
        model = _model()
        a = RLMetaController(model, lr=0.3, seed=7)
        b = RLMetaController(model, lr=0.3, seed=7)
        a.train(states, steps=500)
        b.train(states, steps=500)
        assert np.allclose(a.theta, b.theta)

    def test_history_records_every_step(self):
        rl = RLMetaController(_model(), seed=0)
        rl.train(_states(), steps=120)
        assert len(rl.history) == 120

    @pytest.mark.slow
    def test_greedy_plan_is_reproducible(self):
        rl = RLMetaController(_model(), lr=0.3, seed=0)
        rl.train(_states(), steps=2000)
        st = _states()[0]
        assert rl.plan(st, 0.5).width == rl.plan(st, 0.5).width
