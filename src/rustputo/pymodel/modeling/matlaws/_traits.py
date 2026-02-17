import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import S2D, S3D, Arr


class ITriaxialModel(abc.ABC):
    @abc.abstractmethod
    def simulate_3d[S: S2D | S3D, F: np.floating](self, right_c: Arr[S, F]) -> Arr[S, F]: ...
