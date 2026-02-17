# /// script
# requires-python = ">=3.14"
# dependencies = [rustputo]
# ///

import numpy as np

from rustputo.pymodel.modeling.matlaws import NeoHookean
from rustputo.rust.constitutive_laws import NeoHookean as RustNeoHookean
from rustputo.rust.testing import sum_as_string

BENCHMARK = np.array([[1, 0.0, 0.0], [0.0, -1, 0.0], [0.0, 0.0, -1]]) / 3 * (2 ** (-1 / 3))


def test_neohookean_array() -> None:
    model = NeoHookean(k=1.0)
    right_c = np.array(
        [
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    stress = model.simulate_3d(right_c)
    assert np.allclose(
        stress,
        BENCHMARK,
        atol=1e-6,
    )


def test_neohookean() -> None:
    model = NeoHookean(k=1.0)
    right_c = np.array(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    stress = model.simulate_3d(right_c)
    assert np.allclose(stress, BENCHMARK, atol=1e-6)


def test_rust_neohookean() -> None:
    model = RustNeoHookean(1.0)
    right_c = np.array(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    stress = model.simulate(right_c)
    assert np.allclose(
        stress,
        BENCHMARK,
        atol=1e-6,
    )


def test_sum_as_string() -> None:
    s = sum_as_string(1, 123)
    assert s == "124"
    assert isinstance(s, str)
    assert s + "okay" == "124okay"
