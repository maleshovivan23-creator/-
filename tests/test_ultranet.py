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


# ======================================================================
#                    Тесты v2: новые возможности
# ======================================================================

def test_no_grad_disables_graph():
    from ultranet import no_grad
    x = Tensor(np.random.randn(3, 3), requires_grad=True)
    with no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad
    z = (x * 2).sum()
    assert z.requires_grad


def test_fused_softmax_grad():
    x = Tensor(np.random.randn(4, 6), requires_grad=True)
    check(lambda: (x.softmax(axis=-1) * Tensor(np.random.RandomState(0).randn(4, 6))).sum(), x)


def test_fused_log_softmax_matches_naive():
    x = Tensor(np.random.randn(5, 7))
    naive = x.data - np.log(np.exp(x.data - x.data.max(-1, keepdims=True)).sum(-1, keepdims=True)) \
        - x.data.max(-1, keepdims=True)
    assert np.allclose(x.log_softmax(-1).data, naive, atol=1e-5)


def test_fused_cross_entropy_grad():
    logits = Tensor(np.random.randn(8, 5), requires_grad=True)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    check(lambda: un.cross_entropy(logits, y), logits)


def test_cross_entropy_ignore_index():
    logits = Tensor(np.random.randn(4, 3), requires_grad=True)
    y = np.array([0, 1, -100, 2])
    loss = un.cross_entropy(logits, y, ignore_index=-100)
    loss.backward()
    assert np.allclose(logits.grad[2], 0.0), "игнорируемая позиция не должна давать градиент"


def test_new_activations_grad():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    for fn in ["silu", "sin", "cos"]:
        check(lambda fn=fn: getattr(x, fn)().sum(), x)


def test_rmsnorm_grad():
    rn = nn.RMSNorm(6)
    x = Tensor(np.random.randn(3, 6), requires_grad=True)
    check(lambda: (rn(x) ** 2).sum(), x, rn.gamma)


def test_rope_preserves_norm_and_is_relative():
    rope = nn.RotaryEmbedding(8, 64)
    x = Tensor(np.random.randn(1, 2, 6, 8).astype(np.float32))
    y = rope(x)
    assert np.allclose(np.linalg.norm(x.data, axis=-1), np.linalg.norm(y.data, axis=-1), atol=1e-4)
    # скалярное произведение зависит только от разности позиций
    q = np.random.randn(1, 1, 1, 8).astype(np.float32)
    k = np.random.randn(1, 1, 1, 8).astype(np.float32)
    d1 = float((rope(Tensor(q), 2).data * rope(Tensor(k), 5).data).sum())
    d2 = float((rope(Tensor(q), 10).data * rope(Tensor(k), 13).data).sum())
    assert abs(d1 - d2) < 1e-3


def test_rope_grad():
    rope = nn.RotaryEmbedding(4, 16)
    x = Tensor(np.random.randn(1, 1, 3, 4), requires_grad=True)
    check(lambda: (rope(x) ** 2).sum(), x)


def test_swiglu_grad():
    m = nn.SwiGLU(8, hidden=16)
    x = Tensor(np.random.randn(2, 8) * 0.5, requires_grad=True)
    check(lambda: m(x).sum(), x, m.w_down.weight, tol=5e-2)


def test_avgpool_grad():
    p = nn.AvgPool2d(2)
    x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)
    check(lambda: (p(x) ** 2).sum(), x)


def test_lstm_cell_shapes_and_grad():
    cell = nn.LSTMCell(3, 4)
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    h, c = Tensor(np.zeros((2, 4))), Tensor(np.zeros((2, 4)))
    check(lambda: cell(x, (h, c))[0].sum(), x, tol=5e-2)


def test_kv_cache_matches_full_forward():
    cfg = un.GPTConfig(vocab_size=17, block_size=16, n_layer=2, n_head=2, n_embd=32)
    m = un.GPT(cfg).eval()
    seq = [1, 5, 9, 3, 7]
    full = m(np.array([seq])).data[0, -1]
    m.reset_cache()
    m(np.array([seq[:-1]]), use_cache=True)
    inc = m(np.array([[seq[-1]]]), use_cache=True, pos_offset=len(seq) - 1).data[0, -1]
    assert np.allclose(full, inc, atol=1e-4), np.abs(full - inc).max()


def test_generate_cache_equals_nocache():
    cfg = un.GPTConfig(vocab_size=13, block_size=12, n_layer=2, n_head=2, n_embd=32)
    m = un.GPT(cfg).eval()
    a = m.generate([2, 3], 12, temperature=0.9, top_k=5, use_cache=True, seed=11)
    b = m.generate([2, 3], 12, temperature=0.9, top_k=5, use_cache=False, seed=11)
    assert a == b


def test_greedy_is_deterministic_and_stop_tokens():
    cfg = un.GPTConfig(vocab_size=11, block_size=12, n_layer=1, n_head=2, n_embd=16)
    m = un.GPT(cfg).eval()
    a = m.generate([1], 10, temperature=0.0)
    b = m.generate([1], 10, temperature=0.0)
    assert a == b
    out = m.generate([1], 30, temperature=0.0, stop_tokens=[a[1]])
    assert out[-1] == a[1] and len(out) == 2


def test_weight_tying_reduces_params():
    kw = dict(vocab_size=500, block_size=16, n_layer=2, n_head=2, n_embd=64)
    tied = un.GPT(un.GPTConfig(tie_weights=True, **kw)).num_params()
    untied = un.GPT(un.GPTConfig(tie_weights=False, **kw)).num_params()
    assert untied - tied == 500 * 64


def test_llama_style_gpt_trains():
    text = "данные учат модель. " * 30
    tok = un.CharTokenizer(text)
    ids = tok.encode(text)
    cfg = un.GPTConfig.llama_style(tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32)
    assert cfg.rope and cfg.norm == "rms" and cfg.swiglu
    m = un.GPT(cfg)
    opt = un.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for _ in range(80):
        xb, yb = un.make_lm_batches(ids, 16, 8)
        loss = un.cross_entropy(m(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(1.0)
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.6, (first, last)


def test_rope_extrapolates_beyond_training_length():
    """RoPE-модель принимает последовательности длиннее block_size без ошибок."""
    cfg = un.GPTConfig.llama_style(12, block_size=8, n_layer=1, n_head=2, n_embd=16)
    m = un.GPT(cfg).eval()
    out = m(np.random.randint(0, 12, (1, 20)))
    assert out.shape == (1, 20, 12)


def test_resnet_learns():
    x, y = un.data.make_digits_like(300, 8, 4, seed=5)
    m = un.ResNet(1, 4, width=8, n_blocks=1, img_size=8)
    tr = un.Trainer(m, un.Adam(m.parameters(), lr=3e-3), verbose=False)
    tr.fit(un.DataLoader(x, y, 32), epochs=6)
    assert tr.history["metric"][-1] > 0.8


def test_text_classifier_learns():
    rng = np.random.default_rng(0)
    n, L, V = 300, 10, 12
    x = rng.integers(2, V, (n, L))
    y = rng.integers(0, 2, n)
    x[y == 1, 0] = 1  # маркерный токен в начале
    x[y == 0, 0] = 0
    m = un.TextClassifier(V, 2, n_embd=32, n_layer=1, n_head=2, max_len=L)
    tr = un.Trainer(m, un.Adam(m.parameters(), lr=3e-3), verbose=False)
    tr.fit(un.DataLoader(x, y, 32), epochs=10)
    assert tr.history["metric"][-1] > 0.9


def test_new_optimizers_reduce_loss():
    for opt_cls in [un.Lion, un.Adagrad]:
        np.random.seed(0)
        model = un.MLP([2, 16, 2])
        x, y = un.make_moons(200, seed=3)
        opt = opt_cls(model.parameters(), lr=1e-2)
        first = last = None
        for _ in range(60):
            loss = un.cross_entropy(model(Tensor(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
            last = loss.item()
        assert last < first, opt_cls.__name__


def test_lookahead_wraps_base():
    model = un.MLP([2, 8, 2])
    base = un.Adam(model.parameters(), lr=1e-2)
    opt = un.Lookahead(base, k=3)
    x, y = un.make_moons(120, seed=1)
    first = last = None
    for _ in range(30):
        loss = un.cross_entropy(model(Tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first


def test_ema_tracks_weights():
    model = un.MLP([2, 4, 2])
    ema = un.EMA(model.parameters(), decay=0.5, warmup=False)
    p = model.parameters()[0]
    orig = p.data.copy()
    p.data += 1.0
    ema.update()
    ema.apply()
    assert np.allclose(p.data, orig + 0.5, atol=1e-5)
    ema.restore()
    assert np.allclose(p.data, orig + 1.0)


def test_onecycle_and_plateau_schedulers():
    m = un.MLP([2, 4, 2])
    opt = un.Adam(m.parameters(), lr=1.0)
    oc = un.OneCycleLR(opt, max_lr=1.0, total=100, pct_start=0.3)
    lrs = [oc.step() for _ in range(100)]
    assert lrs[0] < max(lrs) and lrs[-1] < lrs[0]
    assert abs(max(lrs) - 1.0) < 1e-6

    opt2 = un.Adam(m.parameters(), lr=1.0)
    pl = un.ReduceLROnPlateau(opt2, factor=0.5, patience=1)
    for _ in range(5):
        pl.step(1.0)  # метрика не улучшается
    assert opt2.lr < 1.0


def test_grad_accumulation_equivalence():
    """Накопление градиентов по 2 микробатчам == один большой батч."""
    np.random.seed(0)
    x, y = un.make_moons(64, seed=0)
    m1 = un.MLP([2, 8, 2])
    sd = m1.state_dict()
    m2 = un.MLP([2, 8, 2])
    m2.load_state_dict(sd)

    loss = un.cross_entropy(m1(Tensor(x)), y)
    m1.zero_grad()
    loss.backward()
    g_full = m1.parameters()[0].grad.copy()

    m2.zero_grad()
    for half in range(2):
        xb, yb = x[half * 32:(half + 1) * 32], y[half * 32:(half + 1) * 32]
        (un.cross_entropy(m2(Tensor(xb)), yb) * 0.5).backward()
    assert np.allclose(g_full, m2.parameters()[0].grad, atol=1e-5)


def test_trainer_accum_and_ema_and_checkpoint(tmp_path):
    x, y = un.make_moons(256, seed=2)
    m = un.MLP([2, 16, 2])
    ckpt = tmp_path / "best.pkl"
    tr = un.Trainer(m, un.Adam(m.parameters(), lr=5e-3), verbose=False,
                    accum_steps=2, ema_decay=0.9, checkpoint_path=str(ckpt), grad_clip=1.0)
    hist = tr.fit(un.DataLoader(x, y, 32), un.DataLoader(x, y, 64, shuffle=False), epochs=5)
    assert ckpt.exists()
    assert len(hist["lr"]) == 5 and len(hist["grad_norm"]) == 5
    assert hist["loss"][-1] < hist["loss"][0]


def test_trainer_plot_history_ascii():
    x, y = un.make_moons(120, seed=4)
    m = un.MLP([2, 8, 2])
    tr = un.Trainer(m, un.Adam(m.parameters(), lr=1e-2), verbose=False)
    tr.fit(un.DataLoader(x, y, 32), un.DataLoader(x, y, 64, shuffle=False), epochs=4)
    plot = tr.plot_history()
    assert "train" in plot and "эпохи" in plot and len(plot.splitlines()) > 5


def test_trainer_callback_can_stop():
    x, y = un.make_moons(120, seed=4)
    m = un.MLP([2, 8, 2])

    def cb(trainer, epoch):
        if epoch == 2:
            trainer.stop()

    tr = un.Trainer(m, un.Adam(m.parameters(), lr=1e-2), verbose=False, callbacks=[cb])
    hist = tr.fit(un.DataLoader(x, y, 32), epochs=20)
    assert len(hist["loss"]) == 2


def test_regression_losses():
    x, y = un.make_regression(200, 1, seed=0)
    m = un.MLP([1, 32, 32, 1], activation="tanh")
    opt = un.Adam(m.parameters(), lr=1e-2)
    first = last = None
    for _ in range(300):
        loss = un.mse_loss(m(Tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.2
    assert un.mae_loss(m(Tensor(x)), y).item() < 0.5
    assert un.huber_loss(m(Tensor(x)), y).item() >= 0


def test_huber_and_bce_logits_grad():
    pred = Tensor(np.random.randn(6, 1), requires_grad=True)
    tgt = np.random.randn(6, 1).astype(np.float32)
    check(lambda: un.huber_loss(pred, tgt, delta=1.0), pred)

    logits = Tensor(np.random.randn(6, 1), requires_grad=True)
    ybin = (np.random.rand(6, 1) > 0.5).astype(np.float32)
    check(lambda: un.bce_with_logits(logits, ybin), logits)


def test_focal_loss_runs_and_grads():
    logits = Tensor(np.random.randn(8, 4), requires_grad=True)
    y = np.random.randint(0, 4, 8)
    loss = un.focal_loss(logits, y, gamma=2.0)
    loss.backward()
    assert logits.grad is not None and np.isfinite(loss.item())


def test_metrics():
    logits = Tensor(np.array([[3.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.0, 0.1, 5.0]]))
    y = np.array([0, 1, 2])
    assert un.accuracy(logits, y) == 1.0
    assert un.top_k_accuracy(logits, y, k=2) == 1.0
    assert un.f1_score(logits, y) == 1.0
    cm = un.confusion_matrix(logits, y)
    assert np.array_equal(cm, np.eye(3, dtype=int))
    assert abs(un.perplexity(0.0) - 1.0) < 1e-6


def test_dataset_and_normalize():
    x, y = un.make_moons(60, seed=0)
    ds = un.Dataset(x, y)
    assert len(ds) == 60 and ds[0][0].shape == (2,)
    xb, yb = next(iter(ds.loader(16)))
    assert xb.shape == (16, 2)
    xn, mean, std = un.normalize(x)
    assert abs(float(xn.mean())) < 1e-5 and abs(float(xn.std()) - 1.0) < 1e-2


def test_word_tokenizer_roundtrip():
    tok = un.WordTokenizer("кот сидит на окне кот спит")
    ids = tok.encode("кот спит")
    assert tok.decode(ids) == "кот спит"
    assert tok.encode("неизвестное")[0] == 1  # <unk>


def test_no_grad_speeds_up_and_no_graph():
    from ultranet import no_grad
    m = un.MLP([16, 64, 16])
    x = Tensor(np.random.randn(32, 16))
    with no_grad():
        out = m(x)
    assert not out.requires_grad and out._prev == ()


def test_ema_warmup_tracks_early_updates():
    """На первых шагах EMA не должна сильно отставать от весов."""
    m = un.MLP([2, 4, 2])
    p = m.parameters()[0]
    ema = un.EMA(m.parameters(), decay=0.999, warmup=True)
    target = p.data.copy() + 5.0
    p.data = target.copy()
    for _ in range(5):
        ema.update()
    ema.apply()
    # с прогревом среднее заметно продвинулось к target (без прогрева было бы ~0.5%)
    assert np.abs(p.data - target).mean() < np.abs(target).mean() * 0.7
    ema.restore()


def test_cli_demo_and_bench_run():
    from ultranet.cli import main
    main(["demo"])  # не должно падать


def test_cli_train_and_generate_roundtrip(tmp_path, capsys):
    from ultranet.cli import main
    out = tmp_path / "cli.pkl"
    main(["train-text", "--steps", "20", "--layers", "1", "--embd", "32",
          "--block", "16", "--out", str(out), "--tokens", "20"])
    assert out.exists() and (tmp_path / "cli.pkl.json").exists()
    capsys.readouterr()
    main(["generate", "--checkpoint", str(out), "--prompt", "не", "--tokens", "15", "--seed", "0"])
    text = capsys.readouterr().out.strip()
    assert len(text) > 5


def test_benchmark_timeit_helper():
    from ultranet.benchmark import timeit
    ms = timeit(lambda: np.zeros((10, 10)), n=3, warmup=1)
    assert ms >= 0


# ======================================================================
#              Тесты v3: корректность оптимизаций ядра
# ======================================================================

def test_no_grad_buffer_aliasing():
    """copy-on-write в _accum не должен связывать градиенты разных тензоров."""
    a = Tensor(np.ones((2, 2)), requires_grad=True)
    b = Tensor(np.ones((2, 2)), requires_grad=True)
    (a + b).sum().backward()
    assert np.allclose(a.grad, 1.0) and np.allclose(b.grad, 1.0)
    a.grad += 100.0              # мутируем градиент одного
    assert np.allclose(b.grad, 1.0), "градиент b пострадал от алиасинга"


def test_shared_node_accumulates_twice():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    (x * x).sum().backward()     # d/dx x^2 = 2x
    assert np.allclose(x.grad, [4.0, 6.0])


def test_repeated_backward_accumulates():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * 3).sum().backward()
    (x * 3).sum().backward()
    assert np.allclose(x.grad, [6.0, 6.0]), "градиенты должны складываться"


def test_clip_grad_norm_does_not_corrupt_shared_buffer():
    m = un.MLP([4, 8, 2])
    x, y = np.random.randn(6, 4).astype(np.float32), np.random.randint(0, 2, 6)
    opt = un.Adam(m.parameters(), lr=1e-3)
    un.cross_entropy(m(Tensor(x)), y).backward()
    before = [p.grad.copy() for p in m.parameters()]
    total = opt.clip_grad_norm(1e9)      # порог заведомо не срабатывает
    assert total > 0
    for p, b in zip(m.parameters(), before):
        assert np.allclose(p.grad, b)


def test_basic_index_grad_matches_fancy():
    """Быстрый путь для срезов даёт тот же градиент, что и np.add.at."""
    x = Tensor(np.random.randn(4, 6), requires_grad=True)
    check(lambda: (x[:, 2:5] ** 2).sum(), x)
    check(lambda: (x[1] * 3).sum(), x)
    check(lambda: (x[..., 0] ** 2).sum(), x)


def test_fancy_index_still_accumulates():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    x[np.array([0, 0, 2])].sum().backward()
    assert np.allclose(x.grad, [2.0, 0.0, 1.0]), "повторные индексы должны суммироваться"


def test_attention_qkv_split_correct():
    """Reshape-split q/k/v эквивалентен раздельным срезам."""
    attn = nn.MultiHeadAttention(16, 4, causal=True).eval()
    x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
    b, t, c = x.shape
    qkv = attn.qkv(x)
    fast = qkv.reshape(b, t, 3, attn.n_heads, attn.head_dim).transpose(2, 0, 3, 1, 4)
    slow_q = qkv[:, :, :c].reshape(b, t, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    assert np.allclose(fast.data[0], slow_q.data, atol=1e-6)


# ======================================================================
#          Тесты v3: воспроизводимость, BPE, чекпоинты, gradcheck
# ======================================================================

def test_manual_seed_full_reproducibility():
    un.manual_seed(123)
    m1 = un.MLP([4, 16, 3], dropout=0.3)
    x = np.random.randn(8, 4).astype(np.float32)
    out1 = m1(Tensor(x)).data

    un.manual_seed(123)
    m2 = un.MLP([4, 16, 3], dropout=0.3)
    x2 = np.random.randn(8, 4).astype(np.float32)
    out2 = m2(Tensor(x2)).data
    assert np.allclose(out1, out2), "manual_seed должен давать полный детерминизм"


def test_manual_seed_dropout_deterministic():
    un.manual_seed(5)
    a = Tensor(np.ones((6, 6))).dropout(0.5, training=True).data
    un.manual_seed(5)
    b = Tensor(np.ones((6, 6))).dropout(0.5, training=True).data
    assert np.allclose(a, b)


def test_full_training_run_reproducible():
    def run():
        un.manual_seed(77)
        x, y = un.make_moons(200, seed=0)
        m = un.MLP([2, 16, 2], dropout=0.1)
        tr = un.Trainer(m, un.Adam(m.parameters(), lr=1e-2), verbose=False)
        tr.fit(un.DataLoader(x, y, 32, shuffle=False), epochs=3)
        return tr.history["loss"]

    assert np.allclose(run(), run()), "весь цикл обучения должен воспроизводиться"


# ---------------------------------------------------------------- BPE
def test_bpe_roundtrip_and_compression():
    text = "нейронная сеть учится на данных. " * 30
    tok = un.BPETokenizer.train(text, vocab_size=400)
    assert tok.vocab_size > 256
    s = "нейронная сеть"
    assert tok.decode(tok.encode(s)) == s
    assert tok.compression_ratio(text) > 2.0


def test_bpe_handles_arbitrary_unicode():
    tok = un.BPETokenizer.train("abc abc abc", vocab_size=300)
    for s in ["日本語", "🚀 emoji", "ø ñ ü", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_save_load(tmp_path):
    tok = un.BPETokenizer.train("привет мир привет мир " * 20, vocab_size=320)
    p = tmp_path / "bpe.json"
    tok.save(str(p))
    tok2 = un.BPETokenizer.load(str(p))
    s = "привет мир"
    assert tok2.encode(s) == tok.encode(s) and tok2.decode(tok2.encode(s)) == s


def test_bpe_trains_gpt():
    text = "модель учится на данных. " * 40
    tok = un.BPETokenizer.train(text, vocab_size=280)  # умеренное сжатие
    ids = tok.encode(text)
    assert len(ids) > 40, f"ожидали достаточно токенов, получили {len(ids)}"
    cfg = un.GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=2, n_head=2, n_embd=32)
    m = un.GPT(cfg)
    opt = un.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for _ in range(60):
        xb, yb = un.make_lm_batches(ids, 8, 8)
        loss = un.cross_entropy(m(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first


def test_make_lm_batches_clear_error_on_short_data():
    with pytest.raises(ValueError, match="слишком мало"):
        un.make_lm_batches([1, 2, 3], block_size=32, batch_size=4)


# ------------------------------------------------------- чекпоинты
def test_checkpoint_restores_optimizer_state(tmp_path):
    un.manual_seed(1)
    m = un.MLP([4, 8, 2])
    opt = un.Adam(m.parameters(), lr=1e-3)
    x, y = np.random.randn(6, 4).astype(np.float32), np.random.randint(0, 2, 6)
    for _ in range(3):
        opt.zero_grad()
        un.cross_entropy(m(Tensor(x)), y).backward()
        opt.step()

    p = tmp_path / "ck.pkl"
    un.save_checkpoint(str(p), m, opt, epoch=2, step=opt.t, history={"loss": [0.5]})

    m2 = un.MLP([4, 8, 2])
    opt2 = un.Adam(m2.parameters(), lr=99.0)
    ck = un.load_checkpoint(str(p), m2, opt2)
    assert ck["epoch"] == 2 and ck["history"]["loss"] == [0.5]
    assert opt2.lr == 1e-3 and opt2.t == opt.t
    assert all(np.allclose(a.data, b.data) for a, b in zip(m.parameters(), m2.parameters()))
    assert all(np.allclose(a, b) for a, b in zip(opt.m, opt2.m))


def test_checkpoint_strict_detects_mismatch(tmp_path):
    p = tmp_path / "c.pkl"
    un.save_checkpoint(str(p), un.MLP([4, 8, 2]))
    with pytest.raises(ValueError):
        un.load_checkpoint(str(p), un.MLP([4, 16, 2]))


def test_checkpoint_rejects_wrong_optimizer(tmp_path):
    m = un.MLP([4, 8, 2])
    p = tmp_path / "c.pkl"
    un.save_checkpoint(str(p), m, un.Adam(m.parameters()))
    with pytest.raises(ValueError):
        un.load_checkpoint(str(p), m, un.SGD(m.parameters()))


def test_trainer_resume_continues_training(tmp_path):
    un.manual_seed(3)
    x, y = un.make_moons(200, seed=1)
    loader = un.DataLoader(x, y, 32, shuffle=False)
    ck = tmp_path / "best.pkl"

    m = un.MLP([2, 16, 2])
    tr = un.Trainer(m, un.Adam(m.parameters(), lr=5e-3), verbose=False,
                    checkpoint_path=str(ck))
    tr.fit(loader, epochs=3)
    assert ck.exists()

    m2 = un.MLP([2, 16, 2])
    tr2 = un.Trainer(m2, un.Adam(m2.parameters(), lr=5e-3), verbose=False)
    epoch = tr2.resume(str(ck))
    assert epoch >= 1 and len(tr2.history["loss"]) >= 1
    hist = tr2.fit(loader, epochs=2)
    assert len(hist["loss"]) >= 3   # история продолжилась, а не началась заново


# ------------------------------------------------------- gradcheck API
def test_public_gradcheck_passes_and_detects_bug():
    un.manual_seed(0)
    lin = nn.Linear(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    assert un.gradcheck(lambda: lin(x).sum(), [x, lin.weight, lin.bias])

    # слой с намеренно неверным backward должен быть пойман
    bad = Tensor(np.random.randn(3, 3), requires_grad=True)

    def broken():
        out = bad._make(bad.data * 2.0, (bad,), "broken")
        out._backward = lambda: bad._accum(np.ones_like(bad.data) * 5.0)  # неверно: должно быть 2
        return out.sum()

    with pytest.raises(AssertionError):
        un.gradcheck(broken, [bad])


def test_numeric_grad_matches_known_derivative():
    x = Tensor(np.array([[2.0, 3.0]]), requires_grad=True)
    g = un.numeric_grad(lambda: (x ** 2).sum(), x)
    assert np.allclose(g, [[4.0, 6.0]], atol=1e-2)


# ======================================================================
#              Тесты v3: понятные ошибки и инспекция модели
# ======================================================================

def test_linear_shape_error_is_helpful():
    with pytest.raises(ValueError, match="Linear.*последней размерностью 4"):
        nn.Linear(4, 3)(Tensor(np.random.randn(2, 7)))


def test_embedding_out_of_range_error():
    with pytest.raises(IndexError, match=r"Embedding.*\[0, 9\]"):
        nn.Embedding(10, 4)(np.array([[99]]))


def test_gpt_accepts_1d_sequence():
    m = un.GPT(un.GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16))
    assert m(np.array([1, 2, 3])).shape == (1, 3, 10)


def test_gpt_rejects_3d_input():
    m = un.GPT(un.GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16))
    with pytest.raises(ValueError, match=r"\(batch, seq\)"):
        m(np.zeros((2, 3, 4), dtype=int))


def test_attention_head_divisibility_error():
    with pytest.raises(AssertionError, match="делиться"):
        nn.MultiHeadAttention(10, 4)


def test_model_summary_reports_totals():
    m = un.MLP([4, 8, 2])
    s = m.summary()
    assert "ИТОГО" in s and "память" in s
    total = sum(p.size for p in m.parameters())
    assert f"{total:,}" in s
