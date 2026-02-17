pub mod uniaxial_model;
use crate::simulation::pymodeling::{
    pymodel_hyperelastic_uniaxial_response, pymodel_viscoelastic_uniaxial_response,
};
use crate::tissues_1d::aorta::uniaxial_model::AortaUniaxialViscoelastic;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{Bound, PyResult, Python};
use uniaxial_model::AortaUniaxial;

#[pyo3::pyfunction]
#[pyo3(name = "simulate_aorta_he_uniaxial_response")]
pub fn simulate_aorta_he_uniaxial_response<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    pymodel_hyperelastic_uniaxial_response::<AortaUniaxial>(py, parameters, constants, strain)
}

#[pyo3::pyfunction]
#[pyo3(name = "simulate_aorta_ve_uniaxial_response")]
pub fn simulate_aorta_ve_uniaxial_response<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
    dt: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    pymodel_viscoelastic_uniaxial_response::<AortaUniaxialViscoelastic>(
        py, parameters, constants, strain, dt,
    )
}
