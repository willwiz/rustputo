# /// script
# requires-python = ">=3.14"
# dependencies = [rustputo]
# ///

import numpy as np

from rustputo.pymodel.modeling.hydrostatics import solve_hydrostatics_3d
from rustputo.rust.constitutive_laws import NeoHookean, solve_hydrostatics, solve_hydrostatics_array


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
    stress = np.array([model.simulate(c) for c in right_c])
    py_pressure = solve_hydrostatics_3d(stress, right_c, np.array([0.0, 0.0, 1.0]))
    rust_pressure = solve_hydrostatics_array(stress, right_c, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(py_pressure, rust_pressure, atol=1e-6)


def test_neohookean() -> None:
    model = NeoHookean(k=1.0)
    right_c = np.array(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    stress = model.simulate(right_c)
    py_pressure = solve_hydrostatics_3d(stress, right_c, np.array([0.0, 0.0, 1.0]))
    rust_pressure = solve_hydrostatics(stress, right_c, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(py_pressure, rust_pressure, atol=1e-6)
