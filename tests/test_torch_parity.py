"""Эквивалентность PyTorch-порта и NumPy-референса.

NumPy-движок — источник истины. Если torch-версия разойдётся с ним,
на GPU обучится не та модель, которую мы проверяли здесь.
"""
import numpy as np
import pytest

import ultranet as un
from ultranet.functional import cross_entropy
from ultranet.models import GPT, GPTConfig
from ultranet.optim import Adam

torch = pytest.importorskip("torch")
from ultranet.torch_port import TorchGPT, convert_weights, max_abs_diff  # noqa: E402

GPT2_STYLE = dict(rope=False, norm="layer", swiglu=False, tie_weights=False)
LLAMA_STYLE = dict(rope=True, norm="rms", swiglu=True, tie_weights=True)


def make_pair(**kw):
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=4, n_embd=64,
                    block_size=16, dropout=0.0, **kw)
    un.manual_seed(0)
    nm = GPT(cfg)
    nm.eval()
    tm = TorchGPT(cfg)
    convert_weights(nm, tm)
    tm.eval()
    return cfg, nm, tm


@pytest.mark.parametrize("style,name", [(GPT2_STYLE, "gpt2"), (LLAMA_STYLE, "llama")])
class TestParity:
    def test_same_parameter_count(self, style, name):
        _, nm, tm = make_pair(**style)
        assert sum(p.data.size for p in nm.parameters()) == \
               sum(p.numel() for p in tm.parameters())

    def test_logits_match(self, style, name):
        _, nm, tm = make_pair(**style)
        idx = np.random.default_rng(0).integers(0, 64, (4, 12))
        with torch.no_grad():
            t = tm(torch.tensor(idx)).numpy()
        assert max_abs_diff(nm(idx).data, t) < 1e-4

    def test_loss_matches(self, style, name):
        _, nm, tm = make_pair(**style)
        rng = np.random.default_rng(0)
        x, y = rng.integers(0, 64, (4, 12)), rng.integers(0, 64, (4, 12))
        ln = cross_entropy(nm(x), y).item()
        with torch.no_grad():
            _, lt = tm(torch.tensor(x), torch.tensor(y))
        assert abs(ln - lt.item()) < 1e-5

    def test_gradients_match(self, style, name):
        _, nm, tm = make_pair(**style)
        rng = np.random.default_rng(0)
        x, y = rng.integers(0, 64, (4, 12)), rng.integers(0, 64, (4, 12))
        cross_entropy(nm(x), y).backward()
        _, lt = tm(torch.tensor(x), torch.tensor(y))
        lt.backward()
        g_np = nm.tok_emb.weight.grad
        g_t = tm.tok_emb.weight.grad.numpy()
        rel = max_abs_diff(g_np, g_t) / max(float(np.abs(g_np).max()), 1e-12)
        assert rel < 1e-4

    def test_training_curves_match(self, style, name):
        """12 шагов Adam: расхождение loss не должно накапливаться."""
        _, nm, tm = make_pair(**style)
        rng = np.random.default_rng(0)
        x, y = rng.integers(0, 64, (8, 12)), rng.integers(0, 64, (8, 12))
        nopt = Adam(nm.parameters(), lr=1e-3)
        topt = torch.optim.Adam(tm.parameters(), lr=1e-3)
        tm.train()
        for _ in range(12):
            nopt.zero_grad()
            ln = cross_entropy(nm(x), y)
            ln.backward()
            nopt.step()
            topt.zero_grad()
            _, lt = tm(torch.tensor(x), torch.tensor(y))
            lt.backward()
            topt.step()
        assert abs(ln.item() - lt.item()) < 1e-4

    def test_greedy_generation_identical(self, style, name):
        _, nm, tm = make_pair(**style)
        seq, tseq = [1, 2, 3, 4], [1, 2, 3, 4]
        for _ in range(15):
            seq.append(int(nm(np.array([seq[-16:]])).data[0, -1].argmax()))
        with torch.no_grad():
            for _ in range(15):
                tseq.append(int(tm(torch.tensor([tseq[-16:]]))[0, -1].argmax()))
        assert seq == tseq


class TestNoDoubleCounting:
    """Регрессия: parameters() отдавал общие веса по нескольку раз."""

    def test_parameters_are_unique(self):
        un.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=64, n_layer=3, n_head=4, n_embd=64, block_size=16))
        params = m.parameters()
        assert len({id(p) for p in params}) == len(params)

    def test_named_parameters_are_unique(self):
        un.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=64, n_layer=3, n_head=4, n_embd=64, block_size=16))
        seen = [id(p) for _, p in m.named_parameters()]
        assert len(set(seen)) == len(seen)

    def test_tied_head_counted_once(self):
        un.manual_seed(0)
        tied = GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=4, n_embd=64,
                             block_size=16, tie_weights=True))
        un.manual_seed(0)
        untied = GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=4, n_embd=64,
                               block_size=16, tie_weights=False))
        assert tied.num_params() < untied.num_params()

    def test_optimizer_does_not_step_twice(self):
        """Дубли в parameters() = удвоенный lr для общих весов."""
        un.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=64, n_layer=2, n_head=4, n_embd=64, block_size=16))
        before = m.tok_emb.weight.data.copy()
        opt = Adam(m.parameters(), lr=1e-2)
        rng = np.random.default_rng(0)
        x, y = rng.integers(0, 64, (4, 8)), rng.integers(0, 64, (4, 8))
        opt.zero_grad()
        cross_entropy(m(x), y).backward()
        opt.step()
        moved = np.abs(m.tok_emb.weight.data - before).max()
        # один шаг Adam сдвигает вес примерно на lr, не на 2*lr
        assert moved < 1.5e-2
