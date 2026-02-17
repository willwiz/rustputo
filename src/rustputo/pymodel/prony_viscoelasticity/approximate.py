from typing import TYPE_CHECKING, Protocol

import numpy as np

from ._traits import StressState, StressStateArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype.numpy as opn
    from pytools.arrays import A1, A2, A3


class HasLinearApproximation(Protocol):
    n: int

    def precompute_linear(self, dt: opn.ToFloat) -> tuple[Sequence[float], Sequence[float]]: ...


class HasQuadraticApproximation(Protocol):
    n: int

    def precompute_quadratic(self, dt: opn.ToFloat) -> tuple[Sequence[float], Sequence[float]]: ...


def linear_approximation[F: np.floating](
    model: HasLinearApproximation, f: A2[F], dt: opn.ToFloat, store: StressState[F]
) -> StressState[F]:
    beta, decay = model.precompute_linear(dt)
    prony = [
        (d * (p + b * (f - store.fn))).astype(f.dtype)
        for b, d, p in zip(beta, decay, store.prony, strict=True)
    ]
    return StressState(f, prony)


def linear_approximation_array[F: np.floating](
    model: HasLinearApproximation, f: A3[F], dt: A1[F]
) -> StressStateArray[F]:
    n_p = model.n
    prony = [np.zeros_like(f) for _ in range(n_p)]
    for i in range(1, len(f)):
        state = linear_approximation(
            model, f[i], dt[i], StressState(f[i - 1], [p[i - 1] for p in prony])
        )
        for k in range(n_p):
            prony[k][i] = state.prony[k]
    return StressStateArray(f, prony)


def quadratic_approximation[F: np.floating](
    model: HasQuadraticApproximation, f: A2[F], dt: float, store: StressState[F]
) -> StressState[F]:
    beta, decay = model.precompute_quadratic(dt)
    prony = [
        (d * (p + b * (f - store.fn))).astype(f.dtype)
        for b, d, p in zip(beta, decay, store.prony, strict=True)
    ]
    return StressState(f, prony)


def quadratic_approximation_array[F: np.floating](
    model: HasQuadraticApproximation, f: A3[F], dt: A1[F]
) -> StressStateArray[F]:
    n_p = model.n
    prony = [np.zeros_like(f) for _ in range(n_p)]
    for i in range(1, len(f)):
        state = quadratic_approximation(
            model, f[i], dt[i], StressState(f[i - 1], [p[i - 1] for p in prony])
        )
        for k in range(n_p):
            prony[k][i] = state.prony[k]
    return StressStateArray(f, prony)
