# ruff: noqa: PYI021
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
def simulate_aorta_he_uniaxial_response(
    parameters: Arr1[np.float64],
    constants: Arr1[np.float64],
    strain: Arr1[np.float64],
) -> Arr1[np.float64]:
    """Simulate the hyperelastic response of the aorta under uniaxial loading.

    Parameters
    ----------
    parameters : Arr1[np.float64]
        Model parameters for the hyperelastic material. [kg, ke, kc, bc]

    constants : Arr1[np.float64]
        Constants in the model. [Unused in this model, but kept for consistency in signature]

    strain : Arr1[np.float64]
        Applied strain values.

    Returns
    -------
    Arr1[np.float64]
        Computed stress values corresponding to the input strain.

    """

def simulate_aorta_ve_uniaxial_response(
    parameters: Arr1[np.float64],
    constants: Arr1[np.float64],
    strain: Arr1[np.float64],
    dt: Arr1[np.float64],
) -> Arr1[np.float64]:
    """Simulate the hyperelastic response of the aorta under uniaxial loading.

    Parameters
    ----------
    parameters : Arr1[np.float64]
        Model parameters for the hyperelastic material. [kg, ke, kc, bc, alpha]

    constants : Arr1[np.float64]
        Constants: Tf (time scale of the simulation)

    strain : Arr1[np.float64]
        Applied strain values.

    dt : Arr1[np.float64]
        Time step values corresponding to each strain value.

    Returns
    -------
    Arr1[np.float64]
        Computed stress values corresponding to the input strain.

    """
