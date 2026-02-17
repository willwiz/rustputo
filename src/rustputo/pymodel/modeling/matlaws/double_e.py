import dataclasses as dc
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numpy as np

from ._traits import ITriaxialModel

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, S2D, S3D, Arr


class DoubleEParameters(TypedDict, total=True):
    k_ff: float
    k_ss: float
    k_nn: float
    b_i1: float


class DoubleETuple(NamedTuple):
    k_ff: float
    k_ss: float
    k_nn: float
    b_i1: float


@dc.dataclass(slots=True)
class DoubleE[T: np.floating](ITriaxialModel):
    k: float
    mxm: A2[T]
    sxs: A2[T]
    nxn: A2[T]

    def __init__(self, par: DoubleEParameters, h: A2[T]) -> None:
        if h.shape != (3, 3):
            msg = "h must have shape (3, 3)"
            raise ValueError(msg)
        self.p = DoubleETuple(par["k_ff"], par["k_ss"], par["k_nn"], par["b_i1"])
        self.mxm = h[0][:, None] * h[0][None, :]
        self.sxs = h[1][:, None] * h[1][None, :]
        self.nxn = h[2][:, None] * h[2][None, :]

    def simulate_3d[S: S2D | S3D, F: np.floating](self, right_c: Arr[S, F]) -> Arr[S, F]:
        return _simulate_double_e_3d(self, right_c)


def _simulate_double_e_3d[S: S2D | S3D, T: np.floating, F: np.floating](
    model: DoubleE[T], right_c: Arr[S, F]
) -> Arr[S, F]:
    dtype = right_c.dtype
    i_1 = np.linalg.trace(right_c)
    i_ff: A1[F] = np.einsum("...ij,ij->...", right_c, model.mxm, dtype=dtype)
    i_ss: A1[F] = np.einsum("...ij,ij->...", right_c, model.sxs, dtype=dtype)
    i_nn: A1[F] = np.einsum("...ij,ij->...", right_c, model.nxn, dtype=dtype)
    w_1 = np.exp(model.p.b_i1 * (i_1 - 3))
    return (
        np.einsum("...,ij->...ij", w_1 * i_ff - 1.0, model.p.k_ff * model.mxm)
        + np.einsum("...,ij->...ij", w_1 * i_ss - 1.0, model.p.k_ss * model.sxs)
        + np.einsum("...,ij->...ij", w_1 * i_nn - 1.0, model.p.k_nn * model.nxn)
    ).astype(dtype)
