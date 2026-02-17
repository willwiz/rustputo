# ruff: noqa: PYI021
import numpy as np
from numpy.typing import NDArray
from pytools.arrays import A1, A2

# from rustputo.rust.models import constitutive_laws as constitutive_laws
# from rustputo.rust.models import testing as testing

__all__ = [
    "axpy",
    "lgres_mat",
    "mult",
    "simulate_aorta_he_uniaxial_response",
    "simulate_aorta_ve_uniaxial_response",
    "sum_as_string",
]

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
def simulate_aorta_he_uniaxial_response(
    parameters: A1[np.float64],
    constants: A1[np.float64],
    strain: A1[np.float64],
) -> A1[np.float64]:
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
    parameters: A1[np.float64],
    constants: A1[np.float64],
    strain: A1[np.float64],
    dt: A1[np.float64],
) -> A1[np.float64]:
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
