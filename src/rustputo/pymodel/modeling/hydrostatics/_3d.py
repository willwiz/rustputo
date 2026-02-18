from typing import TYPE_CHECKING, Protocol, cast, overload

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, A3


class HydrostaticResidual(Protocol):
    def __call__(self, p: float) -> float: ...


@overload
def solve_hydrostatics_3d[F: np.floating](stress: A2[F], c_inv: A2[F], *surf: A1[F]) -> F: ...
@overload
def solve_hydrostatics_3d[F: np.floating](stress: A3[F], c_inv: A3[F], *surf: A1[F]) -> A1[F]: ...
def solve_hydrostatics_3d[F: np.floating](
    stress: A2[F] | A3[F], c_inv: A2[F] | A3[F], *surf: A1[F]
) -> F | A1[F]:
    nc = [c_inv.dot(n) for n in surf]
    ns = [stress.dot(n) for n in surf]
    nccn = [np.einsum("...i,...i->...", v, v) for v in nc]
    ncsn = [np.einsum("...i,...i->...", v, w) for v, w in zip(nc, ns, strict=True)]
    res = [rhs / lhs for rhs, lhs in zip(ncsn, nccn, strict=True)]
    return cast("A1[F]|F", sum(res) / len(surf))


def add_hydrostatics_3d_array[F: np.floating](stress: A3[F], c_inv: A3[F], *surf: A1[F]) -> A3[F]:
    pressure = solve_hydrostatics_3d(stress, c_inv, *surf)
    return (stress - np.einsum("...,...ij->...ij", pressure, c_inv)).astype(stress.dtype)
