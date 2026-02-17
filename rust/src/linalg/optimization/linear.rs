pub mod linear_regression;

use linear_regression::lgres_mat;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{Bound, Python};
#[pyfunction]
#[pyo3(name = "lgres_mat")]
pub fn lgres_mat_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array();
    let b = b.as_array();
    match lgres_mat(x, b) {
        Ok(result) => Ok(result.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(format!(
            "Linear regression failed: {}",
            e
        ))),
    }
}
