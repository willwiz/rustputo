import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np

from ._traits import ITriaxialModel

if TYPE_CHECKING:
    from pytools.arrays import S2D, S3D, Arr


@dc.dataclass(slots=True)
class IsoExponential(ITriaxialModel):
    k: float
    b: float

    def simulate_3d[S: S2D | S3D, F: np.floating](self, right_c: Arr[S, F]) -> Arr[S, F]:
        return _simulate_isotropic_exponential_3d(self, right_c)


def _simulate_isotropic_exponential_3d[S: S2D | S3D, F: np.floating](
    model: IsoExponential, right_c: Arr[S, F]
) -> Arr[S, F]:
    c_inv = np.linalg.inv(right_c)
    i_1 = np.linalg.trace(right_c)
    w = model.k * np.exp(model.b * (i_1 - 3))
    iso_c = np.eye(3) - np.einsum("...,...ij->...ij", i_1 / 3.0, c_inv)
    return np.einsum("...,...ij->...ij", w, iso_c, dtype=right_c.dtype)
