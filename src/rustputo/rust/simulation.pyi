# ruff: noqa: PYI021
import numpy as np
from pytools.arrays import A1

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
