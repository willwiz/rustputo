from typing import TYPE_CHECKING, TypedDict

import numpy as np

from rustputo.pymodel.prony_viscoelasticity.approximate import linear_approximation_array
from rustputo.pymodel.prony_viscoelasticity.types import IViscoelasticModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype.numpy as opn
    from pytools.arrays import A1, A3

    from rustputo.pymodel.modeling.matlaws import ITriaxialModel


class MaxwellParameters(TypedDict, total=True):
    tau: float


class MaxwellBranch(IViscoelasticModel):
    __slots__ = ("_laws", "tau")
    _laws: Sequence[ITriaxialModel]
    tau: float
    n: int = 1

    def __init__(self, parameters: MaxwellParameters, *law: ITriaxialModel) -> None:
        self.tau = parameters["tau"]
        self._laws = law

    @property
    def laws(self) -> Sequence[ITriaxialModel]:
        return self._laws

    def precompute_linear(self, dt: opn.ToFloat) -> tuple[Sequence[float], Sequence[float]]:
        return [1.0], [float(self.tau / (self.tau + dt))]

    def simulate_linear[F: np.floating](self, c: A3[F], dt: A1[F]) -> A3[F]:
        pk2 = sum(m.simulate_3d(c) for m in self._laws) + np.zeros_like(c)
        pk2 = pk2.astype(c.dtype)
        state = linear_approximation_array(self, pk2, dt)
        return state.stress
