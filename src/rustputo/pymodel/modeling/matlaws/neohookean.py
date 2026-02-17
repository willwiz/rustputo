import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np

from ._traits import ITriaxialModel

if TYPE_CHECKING:
    from pytools.arrays import S2D, S3D, Arr


@dc.dataclass(slots=True)
class NeoHookean(ITriaxialModel):
    k: float

    def simulate_3d[S: S2D | S3D, F: np.floating](self, right_c: Arr[S, F]) -> Arr[S, F]:
        return _simulate_neohookean_3d(self, right_c)


def _simulate_neohookean_3d[S: S2D | S3D, F: np.floating](
    model: NeoHookean, right_c: Arr[S, F]
) -> Arr[S, F]:
    c_inv = np.linalg.inv(right_c)
    i_1 = np.linalg.trace(right_c)
    j = np.linalg.det(right_c) ** (-1.0 / 3.0)
    iso_c = np.eye(3) - np.einsum("...,...ij->...ij", i_1 / 3.0, c_inv)
    return np.einsum("...,...ij->...ij", model.k * j, iso_c).astype(right_c.dtype)
