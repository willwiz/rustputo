pub mod pymodeling;
pub mod simulate;
use crate::utils::errors::PyError;

use ndarray::ArrayView1;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use pyo3::{Bound, PyResult, Python};

pub trait FromNumpy {
    fn from_np(pars: &ArrayView1<f64>, constants: &ArrayView1<f64>) -> Result<Self, PyError>
    where
        Self: Sized;
}

pub type HyperelasticUniaxialFunction<'py> = fn(
    Python<'py>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>>;

pub type ViscoelasticUniaxialFunction<'py> = fn(
    Python<'py>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>>;
