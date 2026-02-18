mod _3d;

use _3d::{solve_hydrostatics_3d, solve_hydrostatics_3d_array};
use numpy::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray1};
use numpy::{PyReadonlyArray1, PyReadonlyArray3};
use pyo3::{Bound, PyResult, Python};

#[pyo3::pyfunction]
#[pyo3(name = "solve_hydrostatics")]
pub fn pysolve_hydrostatics<'py>(
    _py: Python<'py>,
    stress: PyReadonlyArray2<'py, f64>,
    c_inv: PyReadonlyArray2<'py, f64>,
    surf: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    Ok(solve_hydrostatics_3d(
        &stress.as_array(),
        &c_inv.as_array(),
        &surf.as_array(),
    ))
}

#[pyo3::pyfunction]
#[pyo3(name = "solve_hydrostatics_array")]
pub fn pysolve_hydrostatics_array<'py>(
    py: Python<'py>,
    stress: PyReadonlyArray3<'py, f64>,
    c_inv: PyReadonlyArray3<'py, f64>,
    surf: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(
        solve_hydrostatics_3d_array(&stress.as_array(), &c_inv.as_array(), &surf.as_array())
            .into_pyarray(py),
    )
}
