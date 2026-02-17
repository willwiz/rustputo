import numpy as np
from numpy.typing import NDArray
from pytools.arrays import A1, A2

def sum_as_string(a: int, b: int) -> str: ...
def axpy[T: (np.integer, np.float64, np.float32, np.str_, np.bool_)](
    a: float | NDArray[T],
    x: A1[T],
    y: A1[T],
) -> A1[T]: ...
def mult[T: (np.integer, np.float64, np.float32, np.str_, np.bool_)](
    a: float,
    x: A1[T],
) -> None: ...
def lgres_mat[T: (np.integer, np.float64, np.float32, np.str_, np.bool_)](
    x: A2[T],
    b: A1[T],
) -> A1[T]: ...
