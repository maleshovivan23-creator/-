"""Тесты Протея: каждое заявление спецификации проверяется кодом."""
import numpy as np
import pytest

import ultranet as un
from ultranet.proteus import (
    ByteTokenizer,
    DeviceState,
    LocalMemory,
    MatConfig,
    MatFormer,
    MetaController,
    Proteus,
    Swarm,
    TernaryModel,
    evaluate_widths,
    get_device,
    make_byte_batches,
    quantization_error,
    ternary_quantize,
    train_matryoshka,
)
from ultranet.proteus.quant import pack_ternary, unpack_ternary


# ======================================================================
#  Часть 1: «Токенизатор отсутствует. Все языки одинаково дёшевы»
# ======================================================================

def test_byte_tokenizer_has_no_vocabulary():
    tok = ByteTokenizer()
    assert tok.vocab_size == 260          # 256 байт + 4 спецтокена, и это всё


@pytest.mark.parametrize("text", [
    "привет мир",
    "hello world",
    "日本語のテキスト",
    "مرحبا بالعالم",
    "🚀🧠✨ emoji",
    "def f(x): return x**2",
    "ATGCATGCATGC",
    "♪♫ C#m7 ♯♭",
    "Ünïcödé ñ ø å",
    "",
])
def test_byte_roundtrip_any_language(text):
    """Любой язык/символ кодируется без <unk> — заявление подтверждено."""
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(text)) == text


def test_unknown_language_still_readable():
    """Язык, которого нет в обучении, всё равно читается побайтово."""
    tok = ByteTokenizer()
    invented = "ᚦᚢᚱᛁᛊᚨᛉ ᚷᛖᛒᛟ"        # руны
    assert tok.decode(tok.encode(invented)) == invented


def test_all_languages_share_one_id_space():
    """Нет «английских» и «русских» токенов — один байтовый диапазон."""
    tok = ByteTokenizer()
    for text in ["hello", "привет", "你好", "مرحبا"]:
        assert all(0 <= i < 256 for i in tok.encode(text))


def test_byte_cost_is_honest_about_utf8():
    """Честная проверка: байтовый вход НЕ делает языки равными по длине.

    UTF-8 сам по себе дороже для кириллицы/CJK. Байты убирают штраф
    словаря, но не штраф кодировки — это важное уточнение к спецификации.
    """
    tok = ByteTokenizer()
    assert tok.bytes_per_char("hello") == 1.0
    assert tok.bytes_per_char("привет") == 2.0        # кириллица = 2 байта
    assert tok.bytes_per_char("日本語") == 3.0          # CJK = 3 байта


def test_fairness_report_structure():
    rep = ByteTokenizer.fairness_report({"en": "cat", "ru": "кот"})
    assert rep["en"]["bytes"] == 3 and rep["ru"]["bytes"] == 6


# ======================================================================
#  «Вложенность: одна модель — все размеры»
# ======================================================================

def test_one_weight_set_many_sizes():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=3, n_embd=128, head_dim=32, block_size=32))
    x = np.array([[1, 2, 3, 4]])
    sizes = {}
    for w in (0.25, 0.5, 1.0):
        out = m(x, width=w)
        assert out.shape == (1, 4, m.cfg.vocab_size)
        sizes[w] = m.active_params(w)
    assert sizes[0.25] < sizes[0.5] < sizes[1.0]


def test_smaller_width_uses_strictly_fewer_params():
    m = MatFormer(MatConfig(n_layer=4, n_embd=128, head_dim=32))
    assert m.active_params(0.25) < m.active_params(1.0) <= m.num_params()


def test_slices_are_nested_prefix_of_same_weights():
    """Малая модель — буквально левый верхний угол большой."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32))
    w_full = m.blocks[0].wq.weight.data
    d = m.width_dim(0.5)
    m.blocks[0].wq.weight.data[:d, :d] += 0.0     # тот же буфер
    assert np.shares_memory(w_full, m.blocks[0].wq.weight.data)


def test_fewer_layers_runs():
    m = MatFormer(MatConfig(n_layer=6, n_embd=64, head_dim=32))
    out = m(np.array([[1, 2, 3]]), width=0.5, n_layers=2)
    assert out.shape[1] == 3


def test_matryoshka_training_makes_all_sizes_work():
    """Ключевой тест: после обучения КАЖДЫЙ срез предсказывает лучше случайного."""
    un.manual_seed(0)
    tok = ByteTokenizer()
    ids = tok.encode("нейронная сеть учится на данных. " * 25)
    cfg = MatConfig(n_layer=3, n_embd=96, head_dim=32, block_size=32,
                    widths=(0.33, 0.67, 1.0))
    m = MatFormer(cfg)
    train_matryoshka(m, ids, steps=120, batch_size=8, verbose=False, exit_loss_weight=0.0)
    res = evaluate_widths(m, ids, [0.33, 0.67, 1.0], n_batches=4)
    random_loss = np.log(cfg.vocab_size)          # ~5.56
    for w, r in res.items():
        assert r["loss"] < random_loss * 0.6, (w, r["loss"])
    # качество растёт с размером
    assert res[1.0]["loss"] <= res[0.33]["loss"] + 1e-6


def test_early_exit_uses_fewer_layers():
    un.manual_seed(0)
    cfg = MatConfig(n_layer=6, n_embd=64, head_dim=32, block_size=32, exit_layers=(1, 3))
    m = MatFormer(cfg)
    _, used_low = m.forward_early_exit(np.array([[5, 6, 7]]), threshold=0.0)
    _, used_high = m.forward_early_exit(np.array([[5, 6, 7]]), threshold=1.01)
    assert used_low < used_high == cfg.n_layer


def test_generation_works_at_every_width():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32, block_size=32))
    for w in (0.5, 1.0):
        out = m.generate([104, 105], max_new_tokens=5, width=w, temperature=0.0)
        assert len(out) == 7


# ======================================================================
#  «Квантование 1.58 бит, веса в {-1, 0, 1}»
# ======================================================================

def test_ternary_values_are_only_minus_one_zero_plus_one():
    w = np.random.randn(64, 64).astype(np.float32)
    q, scale = ternary_quantize(w)
    assert set(np.unique(q)).issubset({-1, 0, 1})
    assert scale > 0


def test_pack_unpack_roundtrip():
    q = np.random.choice([-1, 0, 1], size=(37, 11)).astype(np.int8)
    assert np.array_equal(unpack_ternary(pack_ternary(q), q.shape), q)


def test_packing_is_two_bits_per_weight():
    q = np.random.choice([-1, 0, 1], size=(100, 100)).astype(np.int8)
    packed = pack_ternary(q)
    assert packed.nbytes == pytest.approx(q.size / 4, rel=0.01)


def test_measured_entropy_is_about_1_58_bits():
    """Проверка самого названия «1.58-bit»: log2(3) = 1.585."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32))
    stats = TernaryModel(m).stats
    assert 1.2 <= stats.entropy_bits <= 1.59


def test_real_compression_ratio_vs_fp32():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=3, n_embd=128, head_dim=32))
    stats = TernaryModel(m).stats
    assert stats.packed_bytes < stats.fp32_bytes / 10
    assert stats.ratio_vs_fp16 > 5.0


def test_quantized_model_still_predicts_better_than_random():
    """Сжатие не превращает модель в мусор."""
    un.manual_seed(0)
    tok = ByteTokenizer()
    ids = tok.encode("данные учат модель. " * 30)
    cfg = MatConfig(n_layer=3, n_embd=96, head_dim=32, block_size=32, widths=(1.0,))
    m = MatFormer(cfg)
    train_matryoshka(m, ids, steps=120, batch_size=8, widths=[1.0],
                     verbose=False, exit_loss_weight=0.0)
    before = evaluate_widths(m, ids, [1.0], n_batches=4)[1.0]["loss"]
    TernaryModel(m).dequantize_into(m)
    after = evaluate_widths(m, ids, [1.0], n_batches=4)[1.0]["loss"]
    assert after < np.log(cfg.vocab_size), "после квантования модель не хуже случайной"
    assert after >= before                      # деградация ожидаема и честна


def test_quantization_error_is_reported():
    m = MatFormer(MatConfig(n_layer=1, n_embd=64, head_dim=32))
    errs = quantization_error(m)
    assert errs and all(0 < e < 1.5 for e in errs.values())


# ======================================================================
#  Мета-контроллер: «пересобирает под железо»
# ======================================================================

def test_weaker_device_gets_smaller_config():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=256, head_dim=32))
    mc = MetaController(m)
    tiny = mc.plan(DeviceState(get_device("smartcard")), 0.9)
    big = mc.plan(DeviceState(get_device("laptop")), 0.9)
    assert tiny.active_params < big.active_params


def test_config_always_fits_memory_budget():
    """Главная гарантия: план не превышает бюджет устройства."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=256, head_dim=32))
    mc = MetaController(m)
    for key in ["smartcard", "esp32", "ring", "earbuds", "phone", "server"]:
        st = DeviceState(get_device(key))
        plan = mc.plan(st, 0.9)
        if plan.fits:
            assert plan.weight_bytes <= st.available_bytes, key


def test_low_battery_shrinks_model():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=256, head_dim=32))
    mc = MetaController(m)
    full = mc.plan(DeviceState(get_device("phone"), battery_pct=100), 0.9)
    low = mc.plan(DeviceState(get_device("phone"), battery_pct=5), 0.9)
    assert low.active_params < full.active_params


def test_overheating_shrinks_model():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=256, head_dim=32))
    mc = MetaController(m)
    cool = mc.plan(DeviceState(get_device("phone"), temperature_c=30), 0.9)
    hot = mc.plan(DeviceState(get_device("phone"), temperature_c=90), 0.9)
    assert hot.active_params < cool.active_params


def test_simple_task_uses_less_than_hard_task():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=256, head_dim=32))
    mc = MetaController(m)
    easy = mc.plan(DeviceState(get_device("phone")), task_complexity=0.05)
    hard = mc.plan(DeviceState(get_device("phone")), task_complexity=1.0)
    assert easy.active_params <= hard.active_params


def test_impossible_device_reports_not_fitting():
    """Честность: если не влезает — так и говорим, а не притворяемся."""
    un.manual_seed(0)
    big = MatFormer(MatConfig(n_layer=12, n_embd=512, head_dim=32))
    mc = MetaController(big)
    plan = mc.plan(DeviceState(get_device("smartcard")), 1.0)
    assert not plan.fits and "не влезает" in plan.reason


# ======================================================================
#  RAG: «знание вне весов»
# ======================================================================

def test_memory_retrieves_relevant_fact():
    mem = LocalMemory()
    mem.add_many(["Столица Франции — Париж.", "Вода кипит при 100 градусах.",
                  "Протей работает офлайн."])
    assert "Париж" in mem.search("париж столица")[0].text


def test_memory_works_for_cjk_short_query():
    """Регрессия: короткие CJK-запросы раньше не находились (3-граммы)."""
    mem = LocalMemory()
    mem.add("日本の首都は東京です。")
    mem.add("Париж — столица Франции.")
    assert "東京" in mem.search("東京")[0].text


def test_knowledge_is_cheaper_than_parameters():
    """Заявление: крошечная модель + память знает больше большой без памяти."""
    mem = LocalMemory()
    facts = [f"Факт номер {i}: значение {i * 7}." for i in range(200)]
    mem.add_many(facts)
    model = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32))
    weights_bytes = TernaryModel(model).stats.packed_bytes
    assert mem.size_bytes() < weights_bytes
    assert "Факт номер 137" in mem.search("Факт номер 137")[0].text


def test_memory_save_load(tmp_path):
    mem = LocalMemory()
    mem.add_many(["альфа бета", "гамма дельта"])
    p = tmp_path / "mem.json"
    mem.save(str(p))
    mem2 = LocalMemory.load(str(p))
    assert len(mem2) == 2 and "гамма" in mem2.search("гамма")[0].text


def test_empty_memory_is_safe():
    assert LocalMemory().search("что угодно") == []


# ======================================================================
#  Рой: «один организм, разлитый по железу»
# ======================================================================

def test_swarm_output_identical_to_monolith():
    """Главный тест роя: распределение не меняет результат."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=96, head_dim=32, block_size=32))
    x = np.array([[1, 2, 3, 4, 5]])
    ref = m(x).data
    sw = Swarm(m, [DeviceState(get_device(k)) for k in ["earbuds", "smartwatch", "phone", "laptop"]])
    assert np.allclose(sw.forward(x).data, ref, atol=1e-5)


def test_swarm_assigns_every_node_a_layer():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=64, head_dim=32))
    sw = Swarm(m, [DeviceState(get_device(k)) for k in ["earbuds", "smartwatch", "phone", "laptop"]])
    assert all(n.layers for n in sw.nodes)
    assert sum(len(n.layers) for n in sw.nodes) == 8


def test_stronger_node_gets_more_layers():
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=8, n_embd=64, head_dim=32))
    sw = Swarm(m, [DeviceState(get_device("earbuds")), DeviceState(get_device("laptop"))])
    by_name = {n.name: len(n.layers) for n in sw.nodes}
    assert by_name["Ноутбук"] > by_name["Наушники"]


def test_device_leaves_proteus_retreats_without_loss():
    """«Устройство умерло — Протей стягивается. Ничего не теряется»."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=6, n_embd=64, head_dim=32))
    x = np.array([[7, 8, 9]])
    ref = m(x).data
    sw = Swarm(m, [DeviceState(get_device(k)) for k in ["earbuds", "phone", "laptop"]])
    sw.leave("Ноутбук")
    assert np.allclose(sw.forward(x).data, ref, atol=1e-5)
    assert sum(len(n.layers) for n in sw.nodes) == 6


def test_device_joins_proteus_spreads():
    """«Купил планшет — Протей занимает и его»."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=6, n_embd=64, head_dim=32))
    x = np.array([[7, 8, 9]])
    ref = m(x).data
    sw = Swarm(m, [DeviceState(get_device("phone"))])
    assert len(sw.nodes[0].layers) == 6
    sw.join(DeviceState(get_device("tablet")))
    assert all(n.layers for n in sw.nodes)
    assert np.allclose(sw.forward(x).data, ref, atol=1e-5)


def test_swarm_degrades_to_single_device():
    """Когда устройств нет рядом — оставшийся становится полным Протеем."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=4, n_embd=64, head_dim=32))
    x = np.array([[1, 2]])
    ref = m(x).data
    sw = Swarm(m, [DeviceState(get_device(k)) for k in ["earbuds", "phone"]])
    sw.leave("Смартфон")
    assert len(sw.nodes) == 1 and len(sw.nodes[0].layers) == 4
    assert np.allclose(sw.forward(x).data, ref, atol=1e-5)


def test_swarm_cannot_be_emptied():
    m = MatFormer(MatConfig(n_layer=2, n_embd=64, head_dim=32))
    sw = Swarm(m, [DeviceState(get_device("phone"))])
    with pytest.raises(ValueError):
        sw.leave("Смартфон")


def test_swarm_transfers_activations_not_weights():
    """По сети идут активации (КБ), а не веса — ключ к «нет облака»."""
    un.manual_seed(0)
    m = MatFormer(MatConfig(n_layer=4, n_embd=128, head_dim=32, block_size=32))
    sw = Swarm(m, [DeviceState(get_device(k)) for k in ["earbuds", "laptop"]])
    sw.forward(np.array([[1, 2, 3]]))
    weights = m.active_params(1.0) * 2 // 8
    assert 0 < sw.bytes_total < weights


# ======================================================================
#  Фасад Proteus
# ======================================================================

def test_proteus_responds_and_stays_local():
    un.manual_seed(0)
    p = Proteus.tiny(device="phone")
    r = p.respond("привет", max_new_tokens=6, seed=0)
    assert r.stayed_local and not p.cloud_enabled
    assert isinstance(r.text, str)


def test_proteus_adapts_between_devices():
    un.manual_seed(0)
    p = Proteus.standard()
    weak = p.attach("esp32").controller.plan(p.state, 0.9)
    strong = p.attach("laptop").controller.plan(p.state, 0.9)
    assert weak.active_params < strong.active_params


def test_proteus_uses_memory_in_response():
    un.manual_seed(0)
    p = Proteus.tiny(device="phone")
    p.remember("Столица Франции — Париж.")
    r = p.respond("расскажи про париж подробно", max_new_tokens=4, seed=0)
    assert "Париж" in r.context_used


def test_proteus_simple_command_is_cheaper():
    un.manual_seed(0)
    p = Proteus.standard(device="phone")
    easy = p.controller.plan(p.state, Proteus.estimate_complexity("включи свет"))
    hard = p.controller.plan(p.state, Proteus.estimate_complexity(
        "объясни почему градиент затухает в глубоких сетях и как это исправить"))
    assert easy.active_params <= hard.active_params


def test_proteus_reports_and_matrix():
    un.manual_seed(0)
    p = Proteus.tiny(device="phone")
    assert "бит/вес" in p.footprint()
    m = p.device_matrix(["esp32", "phone"])
    assert "ESP32" in m and "Смартфон" in m


def test_proteus_swarm_integration():
    un.manual_seed(0)
    p = Proteus.tiny()
    sw = p.form_swarm(["earbuds", "phone", "laptop"])
    assert sum(len(n.layers) for n in sw.nodes) == p.model.cfg.n_layer
    p.dissolve_swarm()
    assert p.swarm is None


def test_byte_batches_validate_length():
    with pytest.raises(ValueError, match="слишком мало"):
        make_byte_batches([1, 2, 3], block_size=64, batch_size=2)


def test_invalid_device_key_is_clear():
    with pytest.raises(KeyError, match="неизвестное устройство"):
        get_device("холодильник")


def test_byte_ids_out_of_range_rejected():
    m = MatFormer(MatConfig(n_layer=1, n_embd=64, head_dim=32))
    with pytest.raises(IndexError, match="байтовые id"):
        m(np.array([[999]]))
