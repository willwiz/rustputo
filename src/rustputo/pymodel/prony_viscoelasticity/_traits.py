import abc
import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype.numpy as opn
    from pytools.arrays import A1, A2, A3

    from rustputo.pymodel.modeling.matlaws import ITriaxialModel


class IViscoelasticModel(abc.ABC):
    n: int

    @property
    @abc.abstractmethod
    def laws(self) -> Sequence[ITriaxialModel]: ...

    @abc.abstractmethod
    def precompute_linear(self, dt: opn.ToFloat) -> tuple[Sequence[float], Sequence[float]]: ...

    @abc.abstractmethod
    def simulate_linear[F: np.floating](self, c: A3[F], dt: A1[F]) -> A3[F]: ...


@dc.dataclass(slots=True)
class StressState[F: np.floating]:
    fn: A2[F]
    prony: Sequence[A2[F]]

    @property
    def stress(self) -> A2[F]:
        return (sum(self.prony) + np.zeros_like(self.fn)).astype(self.fn.dtype)


@dc.dataclass(slots=True)
class StressStateArray[F: np.floating]:
    fn: A3[F]
    prony: Sequence[A3[F]]

    @property
    def stress(self) -> A3[F]:
        return (sum(self.prony) + np.zeros_like(self.fn)).astype(self.fn.dtype)

    def __getitem__(self, idx: opn.ToInt) -> StressState[F]:
        return StressState(fn=self.fn[idx].view(), prony=[p[idx].view() for p in self.prony])
