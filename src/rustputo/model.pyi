import numpy as np
from arraystubs import Arr1, Arr2
from numpy.typing import NDArray

def axpy[T: (np.integer, np.float64, np.float32, np.str_, np.bool_)](
    a: float | NDArray[T],
    x: Arr1[T],
    y: Arr1[T],
) -> Arr1[T]: ...
def mult[T: (np.integer, np.float64, np.float32, np.str_, np.bool_)](
    a: float,
    x: Arr1[T],
) -> None: ...
def sum_as_string(a: int, b: int) -> str: ...
def lgres_mat[T: (np.integer, np.float64, np.float32, np.str_, np.bool_)](
    x: Arr2[T],
    b: Arr1[T],
) -> Arr1[T]: ...
def simulate_aorta_uniaxial_response(
    parameters: Arr1[np.float64],
    constants: Arr1[np.float64],
    strain: Arr1[np.float64],
) -> Arr1[np.float64]: ...
