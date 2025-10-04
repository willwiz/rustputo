pub mod uniaxial_model;
use crate::tissues_1d::simulation_1d::simulate_tissue_response;
use numpy::PyReadonlyArray1;
use numpy::{IntoPyArray, PyArray1, PyUntypedArrayMethods};
use pyo3::{Bound, Python};
use uniaxial_model::AortaUniaxial;

#[pyo3::pyfunction]
#[pyo3(name = "simulate_aorta_uniaxial_response")]
pub fn simulate_aorta_uniaxial_response<'py>(
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
    simulate_tissue_response(aorta_model, &strain.as_array()).into_pyarray(py)
}
