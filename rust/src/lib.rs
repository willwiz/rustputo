use numpy::ndarray::{Array1, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD};
use numpy::{
    IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

// example using a mutable borrow to modify an array in-place
fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
    x *= a;
}

fn lgres_mat(x: ArrayView2<'_, f64>, b: ArrayView1<'_, f64>) -> Array1<f64> {
    // This is a placeholder for the actual implementation of the lgres_mat function.
    // For now, it just returns the result of axpy with a = 1.0.
    x.dot(&b)
}

#[pyfunction]
#[pyo3(name = "lgres_mat")]
fn lgres_mat_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let b = b.as_array();
    let z = lgres_mat(x, b);
    z.into_pyarray(py)
}

// wrapper of `axpy`
#[pyfunction]
#[pyo3(name = "axpy")]
fn axpy_py<'py>(
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

// wrapper of `mult`
#[pyfunction]
#[pyo3(name = "mult")]
fn mult_py<'py>(a: f64, x: &Bound<'py, PyArrayDyn<f64>>) {
    let x = unsafe { x.as_array_mut() };
    mult(a, x);
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "model")]
mod rustputo {
    use super::*;

    #[pymodule_export]
    use super::axpy_py;
    #[pymodule_export]
    use super::lgres_mat_py;
    #[pymodule_export]
    use super::mult_py;
    #[pymodule_export]
    use super::sum_as_string;

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
