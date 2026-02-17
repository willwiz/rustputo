import dataclasses as dc
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    import optype.numpy as opn
    from pytools.arrays import A1, A2, A3, DType


@dc.dataclass(slots=True)
class TriaxialKinematics[F: np.floating]:
    f: A2[F]
    c: A2[F]
    c_inv: A2[F]

    @classmethod
    def from_stretch(
        cls, stretch: opn.ToFloat, *, axis: Literal[0, 1, 2], dtype: DType[F] = np.float64
    ) -> TriaxialKinematics[F]:
        stretch_inv = 1.0 / np.sqrt(stretch, dtype=dtype)
        f = np.zeros((3, 3), dtype=dtype)
        off_axis = {0, 1, 2} - {axis}
        f[:, axis, axis] = stretch
        for i in off_axis:
            f[:, i, i] = stretch_inv
        c = f.T @ f
        c_inv = np.linalg.inv(c).astype(dtype)
        return TriaxialKinematics(f=f, c=c, c_inv=c_inv)


@dc.dataclass(slots=True)
class TriaxialKinematicArray[F: np.floating]:
    f: A3[F]
    c: A3[F]
    c_inv: A3[F]

    @classmethod
    def from_stretch(cls, stretch: A1[F], *, axis: Literal[0, 1, 2]) -> TriaxialKinematicArray[F]:
        stretch_inv = 1.0 / np.sqrt(stretch)
        f = np.zeros((stretch.shape[0], 3, 3), dtype=stretch.dtype)
        off_axis = {0, 1, 2} - {axis}
        f[:, axis, axis] = stretch
        for i in off_axis:
            f[:, i, i] = stretch_inv
        c: A3[F] = np.einsum("nji,njk->nik", f, f, dtype=stretch.dtype)
        c_inv = np.linalg.inv(c).astype(stretch.dtype)
        return TriaxialKinematicArray(f=f, c=c, c_inv=c_inv)

    def __getitem__(self, key: int) -> TriaxialKinematics[F]:
        return TriaxialKinematics(f=self.f[key], c=self.c[key], c_inv=self.c_inv[key])

    def __iter__(self) -> Generator[TriaxialKinematics[F]]:
        for k in range(self.f.shape[0]):
            yield self[k]
