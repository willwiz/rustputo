pub mod biomechanics;
pub mod example;
pub mod fractional;
pub mod kinematics;
pub mod linalg;
pub mod tissues_1d;
pub mod utils;

use linalg::optimization::linear::lgres_mat_py;

use example::{axpy_py, mult_py, sum_as_string};
use pyo3::{pymodule, types::PyModule, Bound, PyResult};
use tissues_1d::aorta::{simulate_aorta_he_uniaxial_response, simulate_aorta_ve_uniaxial_response};

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
#[pyo3(name = "models")]
mod rustputo {
    use pyo3::types::PyAnyMethods;

    use super::*;

    #[pymodule_export]
    use super::axpy_py;
    #[pymodule_export]
    use super::lgres_mat_py;
    #[pymodule_export]
    use super::mult_py;
    #[pymodule_export]
    use super::simulate_aorta_he_uniaxial_response;
    #[pymodule_export]
    use super::simulate_aorta_ve_uniaxial_response;
    #[pymodule_export]
    use super::sum_as_string;

    #[pymodule]
    mod testing {
        use super::*;
        use pyo3::Python;
        #[pymodule_export]
        use sum_as_string;
        #[pymodule_init]
        fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            Python::with_gil(|py| {
                py.import("sys")?
                    .getattr("modules")?
                    .set_item("rustputo.model.testing", m)
            })
        }
    }

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
