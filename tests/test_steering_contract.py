"""Контракт SteeringLibrary.learn: какие входы принимаются, какие отвергаются."""
import numpy as np
import pytest

import ultranet as un
from ultranet.proteus import MatConfig, MatFormer
from ultranet.proteus.steering import SteeringLibrary


@pytest.fixture
def lib():
    un.manual_seed(0)
    model = MatFormer(MatConfig(n_layer=4, n_embd=64, head_dim=16, block_size=32))
    return SteeringLibrary(model)


class TestAcceptedInputs:
    def test_list_of_strings(self, lib):
        sv = lib.learn("простота", ["просто и ясно", "коротко"],
                       ["сложно и запутанно", "длинно"], layer=2)
        assert sv.dim == 64
        assert np.isfinite(sv.vector).all()

    def test_list_of_bytes(self, lib):
        assert lib.learn("b", [b"abc"], [b"xyz"], layer=1).dim == 64

    def test_list_of_token_ids(self, lib):
        assert lib.learn("c", [[1, 2, 3]], [[4, 5, 6]], layer=1).dim == 64

    def test_list_of_numpy_arrays(self, lib):
        sv = lib.learn("d", [np.array([1, 2, 3])], [np.array([4, 5, 6])], layer=1)
        assert np.isfinite(sv.vector).all()

    def test_mixed_strings_and_ids(self, lib):
        sv = lib.learn("mix", ["текст", [1, 2, 3]], ["другое", [4, 5]], layer=1)
        assert np.isfinite(sv.vector).all()


class TestRejectedInputs:
    def test_learn_rejects_bare_string(self, lib):
        """Голая строка разобралась бы посимвольно — молча неверный вектор."""
        with pytest.raises(TypeError, match="списком примеров"):
            lib.learn("x", "строка без списка", ["другое"], layer=1)

    def test_learn_rejects_bare_string_as_negative(self, lib):
        with pytest.raises(TypeError, match="списком примеров"):
            lib.learn("x", ["ок"], "строка", layer=1)

    def test_learn_rejects_bare_bytes(self, lib):
        with pytest.raises(TypeError, match="списком примеров"):
            lib.learn("x", b"bytes", [b"ok"], layer=1)

    def test_learn_rejects_empty_positive(self, lib):
        with pytest.raises(ValueError, match="пустыми"):
            lib.learn("x", [], [[1, 2]], layer=1)

    def test_learn_rejects_empty_negative(self, lib):
        with pytest.raises(ValueError, match="пустыми"):
            lib.learn("x", [[1, 2]], [], layer=1)

    def test_error_message_suggests_the_fix(self, lib):
        with pytest.raises(TypeError, match=r"\[positive\]"):
            lib.learn("x", "строка", ["ок"], layer=1)


class TestVectorSemantics:
    def test_normalized_vector_is_unit_length(self, lib):
        sv = lib.learn("n", ["раз", "два"], ["три", "четыре"], layer=2, normalize=True)
        assert np.linalg.norm(sv.vector) == pytest.approx(1.0, abs=1e-5)

    def test_identical_sets_give_near_zero_vector(self, lib):
        same = ["одинаково", "одно и то же"]
        sv = lib.learn("z", same, same, layer=2, normalize=False)
        assert np.abs(sv.vector).max() < 1e-5

    def test_swapping_sets_flips_direction(self, lib):
        a = lib.learn("a", ["просто"], ["сложно"], layer=2).vector.copy()
        b = lib.learn("b", ["сложно"], ["просто"], layer=2).vector
        assert np.allclose(a, -b, atol=1e-5)

    def test_negative_layer_counts_from_the_end(self, lib):
        assert lib.learn("l", ["x"], ["y"], layer=-1).layer == 3

    def test_learned_vector_is_registered(self, lib):
        lib.learn("имя", ["a"], ["b"], layer=1)
        assert "имя" in lib and len(lib) == 1

    def test_apply_unknown_vector_raises(self, lib):
        with pytest.raises(KeyError, match="не выучен"):
            lib.apply("нет такого")
