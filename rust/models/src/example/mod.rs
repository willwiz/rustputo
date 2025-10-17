pub mod arrays;

use arrays::{axpy, mult};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, Python};
#[pyfunction]
#[pyo3(name = "mult")]
pub fn mult_py<'py>(a: f64, x: &Bound<'py, PyArrayDyn<f64>>) {
    let x = unsafe { x.as_array_mut() };
    mult(a, x);
}

#[pyfunction]
#[pyo3(name = "axpy")]
pub fn axpy_py<'py>(
    py: Python<'py>,
    a: f64,
    x: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let z = axpy(a, x, y);
    z.into_pyarray(py)
}

#[pyfunction]
pub fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
