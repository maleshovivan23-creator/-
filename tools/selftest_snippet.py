# Эта часть дословно вставляется в конец бандла (tools/make_bundle.py).
# Держим её отдельным файлом, а не строкой внутри генератора: экранировать
# кавычки и переносы внутри строки в строке слишком легко сломать.


# ═════════════════════════════════════════════════════════════════════
# Самопроверка: что внутри и работает ли оно
# ═════════════════════════════════════════════════════════════════════
def selftest() -> None:
    """Обучить крошечный GPT и показать, что упаковано в файл."""
    import numpy as _np

    import ultranet as _un
    from ultranet.functional import cross_entropy as _ce
    from ultranet.models import GPT as _GPT
    from ultranet.models import GPTConfig as _Cfg
    from ultranet.optim import Adam as _Adam
    from ultranet.tokenizer import BPETokenizer as _BPE

    print(f"Протей {_un.__version__} — самопроверка")
    mods = sorted(m for m in sys.modules if m.startswith("ultranet"))
    print(f"модулей загружено: {len(mods)}")

    print("\n1. Токенизатор")
    text = "нейронная сеть учится на данных. " * 40
    tok = _BPE.train(text, vocab_size=400)
    assert tok.decode(tok.encode("нейронная сеть")) == "нейронная сеть"
    print(f"   словарь {tok.vocab_size}, сжатие x{tok.compression_ratio(text):.1f}, "
          f"roundtrip OK")

    print("\n2. Обучение GPT на NumPy (без torch)")
    _un.manual_seed(0)
    ids = tok.encode(text)
    need = 32 * 8 + 1
    while len(ids) < need:
        ids = ids + ids
    cfg = _Cfg(vocab_size=tok.vocab_size, block_size=32, n_layer=2,
               n_head=4, n_embd=64, rope=True, norm="rms", swiglu=True)
    model = _GPT(cfg)
    opt = _Adam(model.parameters(), lr=3e-3)
    data = _np.array(ids[:32 * 8], dtype=int).reshape(8, 32)
    tgt = _np.array(ids[1:32 * 8 + 1], dtype=int).reshape(8, 32)
    first = last = None
    for i in range(40):
        opt.zero_grad()
        loss = _ce(model(data), tgt)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    print(f"   {model.num_params():,} параметров, loss {first:.3f} -> {last:.3f}")
    assert last < first, "модель не обучается"

    print("\n3. Механизмы Протея: адаптация под железо")
    from ultranet.proteus import (DeviceState, MatConfig, MatFormer,
                                  MetaController, get_device)
    mf = MatFormer(MatConfig(n_layer=12, n_embd=1024, head_dim=64, block_size=64))
    ctrl = MetaController(mf)
    # различие видно там, где память реально жмёт: свободная доля RAM
    for frac in [0.002, 0.01, 0.1, 1.0]:
        st = DeviceState(get_device("phone"), ram_free_frac=frac)
        plan = ctrl.plan(st, 0.5)
        fits = "" if plan.fits else "  (не влезает)"
        print(f"   свободно {st.available_bytes / 1024 ** 2:>7,.0f} МБ  "
              f"ширина {plan.width:.2f}, слоёв {plan.n_layers:>2}, "
              f"{plan.active_params:>12,} активных{fits}")

    print("\n4. Сжатие весов")
    from ultranet.proteus import RVQ
    w = _np.random.default_rng(0).standard_normal((64, 64)).astype(_np.float32) * 0.05
    rvq = RVQ(dim=4, size=256, stages=3, iters=4).fit(w)
    codes = rvq.encode(w)
    rel = float(_np.abs(rvq.decode(codes).reshape(w.shape) - w).mean()
                / _np.abs(w).mean())
    print(f"   RVQ, 3 кодбука по {rvq.bits_per_weight():.2f} бит/вес: "
          f"средняя относительная ошибка {rel * 100:.1f}%")

    print("\nВсё работает. Обучение на TinyStories:")
    print("    !pip install -q datasets")
    print("    !python proteus_colab.py --steps 200")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        selftest()
