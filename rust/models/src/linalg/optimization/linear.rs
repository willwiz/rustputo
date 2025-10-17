pub mod linear_regression;

use linear_regression::lgres_mat;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{Bound, Python};
#[pyfunction]
#[pyo3(name = "lgres_mat")]
pub fn lgres_mat_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let b = b.as_array();
    let z = lgres_mat(x, b);
    z.into_pyarray(py)
}
