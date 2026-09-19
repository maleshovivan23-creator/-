"""Регрессия скорости и согласованности BPE.

encode был O(число_слияний x длина текста): токенизация TinyStories
заняла бы 19 часов. Эти тесты фиксируют, что так больше не будет.
"""
import time

import pytest

from ultranet.tokenizer import BPETokenizer


@pytest.fixture(scope="module")
def tok():
    text = ("нейронная сеть учится на данных. the quick brown fox jumps "
            "over the lazy dog. привет мир! ") * 200
    return BPETokenizer.train(text, vocab_size=1024)


class TestCorrectness:
    @pytest.mark.parametrize("s", [
        "привет мир", "The quick brown fox!", "日本語テスト", "🚀 emoji",
        "", " ", "a", "\n\n  двойные   пробелы\t\tи табы", "ø ñ ü",
    ])
    def test_roundtrip(self, tok, s):
        assert tok.decode(tok.encode(s)) == s

    def test_roundtrip_long_text(self, tok):
        s = "нейронная сеть учится на данных. " * 50
        assert tok.decode(tok.encode(s)) == s

    def test_empty_and_whitespace(self, tok):
        assert tok.encode("") == []
        assert tok.decode(tok.encode("   ")) == "   "

    def test_cache_does_not_change_result(self, tok):
        s = "повторяющийся кусок текста"
        first = tok.encode(s)
        assert tok.encode(s) == first        # второй раз — из кэша
        assert tok.decode(first) == s


class TestCompression:
    def test_compression_beats_bytes(self, tok):
        text = "нейронная сеть учится на данных. " * 30
        assert tok.compression_ratio(text) > 2.0

    def test_training_matches_encoding(self):
        """train и encode должны работать по одним и тем же кускам.

        Иначе словарь копит слияния через пробел, encode их применить
        не может, и реальное сжатие оказывается хуже заявленного.
        """
        text = "нейронная сеть учится на данных. " * 30
        t = BPETokenizer.train(text, vocab_size=400)
        assert t.compression_ratio(text) > 5.0


class TestSpeed:
    @pytest.mark.slow
    def test_encode_is_not_quadratic(self, tok):
        """Время должно расти линейно, а не квадратично."""
        base = "нейронная сеть учится на данных. the quick brown fox. "
        small, big = base * 40, base * 400          # x10 длины

        def measure(s):
            t0 = time.perf_counter()
            tok.encode(s)
            return time.perf_counter() - t0

        measure(small)
        t_small = min(measure(small) for _ in range(3))
        t_big = min(measure(big) for _ in range(3))
        # при квадратичном росте было бы ~x100; допускаем x30 на шум
        assert t_big < t_small * 30

    @pytest.mark.slow
    def test_throughput_is_reasonable(self, tok):
        """Не меньше 1M символов/с — иначе корпус не токенизировать."""
        text = "нейронная сеть учится на данных. the quick brown fox. " * 2000
        tok.encode(text[:100])
        t0 = time.perf_counter()
        tok.encode(text)
        speed = len(text) / (time.perf_counter() - t0)
        assert speed > 1_000_000, f"только {speed:,.0f} символов/с"
