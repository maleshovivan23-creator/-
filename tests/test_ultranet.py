"""Тесты: численная проверка градиентов + сквозное обучение."""
import numpy as np
import pytest

import ultranet as un
from ultranet import nn
from ultranet.tensor import Tensor


def numgrad(f, x, eps=1e-3):
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x.data[i]
        x.data[i] = orig + eps
        a = f().item()
        x.data[i] = orig - eps
        b = f().item()
        x.data[i] = orig
        g[i] = (a - b) / (2 * eps)
        it.iternext()
    return g


def check(f, *params, tol=2e-2):
    loss = f()
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        ng = numgrad(f, p)
        assert np.allclose(p.grad, ng, atol=tol, rtol=tol), f"grad mismatch:\n{p.grad}\n{ng}"


np.random.seed(0)


def test_basic_ops_grad():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 2), requires_grad=True)
    check(lambda: ((a @ b) ** 2).sum(), a, b)


def test_activations_grad():
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    for fn in ["relu", "tanh", "sigmoid", "gelu", "exp"]:
        check(lambda fn=fn: getattr(x, fn)().sum(), x)


def test_softmax_and_ce_grad():
    logits = Tensor(np.random.randn(6, 4), requires_grad=True)
    y = np.array([0, 1, 2, 3, 1, 0])
    check(lambda: un.cross_entropy(logits, y), logits)


def test_reductions_and_reshape():
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    check(lambda: (x.mean(axis=1) ** 2).sum(), x)
    check(lambda: x.transpose(0, 2, 1).reshape(2, 12).sum(axis=1).sum(), x)
    check(lambda: (x.max(axis=-1) * 3).sum(), x)


def test_layernorm_grad():
    ln = nn.LayerNorm(5)
    x = Tensor(np.random.randn(3, 5), requires_grad=True)
    check(lambda: (ln(x) ** 2).sum(), x, ln.gamma, ln.beta)


def test_linear_grad():
    lin = nn.Linear(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    check(lambda: lin(x).sum(), x, lin.weight, lin.bias)


def test_conv2d_grad():
    conv = nn.Conv2d(2, 3, 3, padding=1)
    x = Tensor(np.random.randn(2, 2, 5, 5), requires_grad=True)
    check(lambda: (conv(x) ** 2).sum(), x, conv.weight, conv.bias, tol=5e-2)


def test_maxpool_grad():
    pool = nn.MaxPool2d(2)
    x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)
    check(lambda: (pool(x) ** 2).sum(), x)


def test_attention_grad():
    attn = nn.MultiHeadAttention(8, 2, causal=True)
    x = Tensor(np.random.randn(2, 4, 8) * 0.5, requires_grad=True)
    check(lambda: (attn(x) ** 2).sum(), x, attn.proj.weight, tol=5e-2)


def test_causal_mask_is_causal():
    attn = nn.MultiHeadAttention(8, 2, causal=True).eval()
    x = np.random.randn(1, 5, 8).astype(np.float32)
    y1 = attn(Tensor(x)).data
    x2 = x.copy()
    x2[0, 4] += 10.0  # меняем последний токен
    y2 = attn(Tensor(x2)).data
    assert np.allclose(y1[0, :4], y2[0, :4], atol=1e-5)


def test_mlp_learns_spirals():
    x, y = un.make_spirals(200, 3, seed=1)
    xtr, ytr, xte, yte = un.train_test_split(x, y, 0.25, seed=1)
    model = un.MLP([2, 64, 64, 3])
    tr = un.Trainer(model, un.Adam(model.parameters(), lr=3e-3), verbose=False)
    tr.fit(un.DataLoader(xtr, ytr, 32), epochs=60)
    _, acc = tr.evaluate(un.DataLoader(xte, yte, 64, shuffle=False))
    assert acc > 0.9, acc


def test_convnet_learns_shapes():
    x, y = un.data.make_digits_like(200, 8, 4, seed=2)
    model = un.ConvNet(1, 4, width=8, img_size=8)
    tr = un.Trainer(model, un.Adam(model.parameters(), lr=3e-3), verbose=False)
    tr.fit(un.DataLoader(x, y, 32), epochs=8)
    assert tr.history["metric"][-1] > 0.85


def test_gpt_overfits_tiny_text():
    text = "привет мир! " * 40
    tok = un.CharTokenizer(text)
    ids = tok.encode(text)
    cfg = un.GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = un.GPT(cfg)
    opt = un.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(120):
        xb, yb = un.make_lm_batches(ids, 16, 8)
        loss = un.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
    out = model.generate(tok.encode("привет"), 20, temperature=0.8, seed=0)
    assert len(out) == len("привет") + 20


def test_save_load_roundtrip(tmp_path):
    m1 = un.MLP([4, 8, 3])
    p = tmp_path / "m.pkl"
    m1.save(str(p))
    m2 = un.MLP([4, 8, 3])
    m2.load(str(p))
    x = np.random.randn(2, 4).astype(np.float32)
    assert np.allclose(m1(Tensor(x)).data, m2(Tensor(x)).data)


def test_optimizers_reduce_loss():
    for opt_cls in [un.SGD, un.Adam, un.RMSprop]:
        np.random.seed(0)
        model = un.MLP([2, 16, 2])
        x, y = un.make_moons(200, seed=3)
        opt = opt_cls(model.parameters(), lr=1e-2)
        first = last = None
        for _ in range(50):
            loss = un.cross_entropy(model(Tensor(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
            last = loss.item()
        assert last < first, opt_cls.__name__


def test_scheduler_warmup_cosine():
    m = un.MLP([2, 4, 2])
    opt = un.Adam(m.parameters(), lr=1.0)
    sch = un.CosineWarmup(opt, warmup=5, total=20)
    lrs = [sch.step() for _ in range(20)]
    assert lrs[0] < lrs[4] and abs(lrs[4] - 1.0) < 1e-6 and lrs[-1] < lrs[5]


def test_dropout_eval_is_identity():
    d = nn.Dropout(0.5).eval()
    x = Tensor(np.ones((4, 4)))
    assert np.allclose(d(x).data, 1.0)
