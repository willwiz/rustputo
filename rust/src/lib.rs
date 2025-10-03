mod biomechanics;
mod fractional;
mod linalg;
mod tests;

use linalg::optimization::linear::lgres_mat_py;

use pyo3::{pymodule, types::PyModule, Bound, PyResult};
use tests::{axpy_py, mult_py, sum_as_string};
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
