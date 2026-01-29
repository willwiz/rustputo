pub mod uniaxial_model;
use crate::tissues_1d::aorta::uniaxial_model::AortaUniaxialViscoelastic;
use crate::tissues_1d::simulation::{
    simulate_hyperelastic_uniaxial_response, simulate_viscoelastic_uniaxial_response,
};
use numpy::PyReadonlyArray1;
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::{Bound, PyResult, Python};
use uniaxial_model::AortaUniaxial;
#[pyo3::pyfunction]
#[pyo3(name = "simulate_aorta_he_uniaxial_response")]
pub fn simulate_aorta_he_uniaxial_response<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    _constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    const PARAMETER_VEC_SIZE: usize = 4;
    if parameters.shape() != &[PARAMETER_VEC_SIZE] {
        return Err(PyValueError::new_err(format!(
            "Array has incorrect dimensions. Expected [matrix_k, elastin_k, collagen_k, collagen_b], got {}",
            parameters.len()
        )));
    }
    let p = parameters.as_array();
    let aorta_model = AortaUniaxial::new(p[0], p[1], p[2], p[3]);
    Ok(simulate_hyperelastic_uniaxial_response(&aorta_model, &strain.as_array()).into_pyarray(py))
}

/// Compute the viscoelastic response of the aorta under uniaxial loading.
/// # Arguments
/// * `parameters` - A 1D array of model parameters:
///     - matrix_k: stiffness of the ground matrix
///     - elastin_k: stiffness of the elastin fibers
///     - collagen_k: stiffness of the collagen fibers
///     - collagen_b: viscosity of the collagen fibers
///     - alpha: fractional order of the viscoelastic model
/// * `constants` - A 1D array of constant parameters:
///     - T_f: The time scale of the simulation
/// * `strain` - A 1D array of strain values applied to the aorta
/// * `dt` - A 1D array of time step values for the simulation
#[pyo3::pyfunction]
#[pyo3(name = "simulate_aorta_ve_uniaxial_response")]
pub fn simulate_aorta_ve_uniaxial_response<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
    dt: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    const PARAMETER_VEC_SIZE: usize = 5;
    if parameters.shape() != &[PARAMETER_VEC_SIZE] {
        return Err(PyValueError::new_err(format!(
            "Array has incorrect dimensions. Expected [matrix_k, elastin_k, collagen_k, collagen_b], got {}",
            parameters.len()
        )));
    }
    let p = parameters.as_array();

    let model_constants = match constants.get(0) {
        Some(val) => *val,
        None => {
            return Err(PyValueError::new_err(
                "ValueError: constants array is empty.",
            ));
        }
    };
    let mut aorta_model =
        match AortaUniaxialViscoelastic::new(p[0], p[1], p[2], p[3], p[4], model_constants) {
            Ok(model) => model,
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Failed to create AortaUniaxialViscoelastic model: {}",
                    e
                )));
            }
        };
    match simulate_viscoelastic_uniaxial_response(
        &mut aorta_model,
        &strain.as_array(),
        &dt.as_array(),
    ) {
        Ok(result) => Ok(result.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(format!("Simulation failed: {}", e))),
    }
}
