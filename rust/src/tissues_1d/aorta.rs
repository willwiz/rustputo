pub mod uniaxial_model;
use crate::tissues_1d::aorta::uniaxial_model::AortaUniaxialViscoelastic;
use crate::tissues_1d::simulation_1d::{
    simulate_hyperelastic_response, simulate_viscoelastic_response,
};
use numpy::PyReadonlyArray1;
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
use pyo3::{Bound, Python};
use uniaxial_model::AortaUniaxial;

#[pyo3::pyfunction]
#[pyo3(name = "simulate_aorta_he_uniaxial_response")]
pub fn simulate_aorta_he_uniaxial_response<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    _constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    if parameters.shape() != &[4] {
        println!(
            "Array has incorrect dimensions. Expected {:?}, got {:?}",
            &[4],
            parameters.shape()
        );
        println!("parameters expected: [matrix_k, elastin_k, collagen_k, collagen_b]");
        panic!("ValueError: Array has incorrect dimensions.");
    }
    let aorta_model = AortaUniaxial::new(
        *parameters.get(0).unwrap(),
        *parameters.get(1).unwrap(),
        *parameters.get(2).unwrap(),
        *parameters.get(3).unwrap(),
    );
    simulate_hyperelastic_response(&aorta_model, &strain.as_array()).into_pyarray(py)
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
) -> Bound<'py, PyArray1<f64>> {
    if parameters.shape() != &[5] {
        println!(
            "Array has incorrect dimensions. Expected {}, got {}",
            5,
            parameters.len()
        );
        println!("parameters expected: [matrix_k, elastin_k, collagen_k, collagen_b]");
        panic!("ValueError: Array has incorrect dimensions.");
    }
    let mut aorta_model = AortaUniaxialViscoelastic::new(
        *parameters.get(0).unwrap(),
        *parameters.get(1).unwrap(),
        *parameters.get(2).unwrap(),
        *parameters.get(3).unwrap(),
        *parameters.get(4).unwrap(),
        *constants.get(0).unwrap(),
    );
    simulate_viscoelastic_response(&mut aorta_model, &strain.as_array(), &dt.as_array())
        .into_pyarray(py)
}
