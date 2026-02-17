from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, A2, A3


class HydrostaticResidual(Protocol):
    def __call__(self, p: float) -> float: ...


def _solve_pressure(func: HydrostaticResidual) -> float:
    res = minimize_scalar(func, tol=1e-14)
    return res.x


def solve_hydrostatics_3d[F: np.floating](
    stress: A2[F], c_inv: A2[F], free_sufs: Sequence[int]
) -> float:
    def residual(p: float) -> float:
        res2 = stress - p * c_inv
        return sum((res2[:, i] * res2[:, i]).sum() for i in free_sufs)

    return _solve_pressure(residual)


def solve_hydrostatics_3d_array[F: np.floating](
    stress: A3[F], c_inv: A3[F], free_sufs: Sequence[int]
) -> A1[F]:
    return np.fromiter(
        (solve_hydrostatics_3d(s, c, free_sufs) for s, c in zip(stress, c_inv, strict=True)),
        dtype=stress.dtype,
    )


def add_hydrostatics_3d_array[F: np.floating](
    stress: A3[F], c_inv: A3[F], free_sufs: Sequence[int]
) -> A3[F]:
    pressure = solve_hydrostatics_3d_array(stress, c_inv, free_sufs)
    return (stress - pressure[:, None, None] * c_inv).astype(stress.dtype)
