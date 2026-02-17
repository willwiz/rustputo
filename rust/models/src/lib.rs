pub mod biomechanics;
pub mod example;
pub mod kinematics;
pub mod linalg;
pub mod simulation;
pub mod tissues_1d;
pub mod tissues_3d;
pub mod utils;
pub mod viscoelasticity;
use linalg::optimization::linear::lgres_mat_py;
use pyo3::types::PyAnyMethods;
use pyo3::Python;

use crate::tissues_3d::neohookean::PyNeoHookean;
use example::{axpy_py, mult_py, sum_as_string};
use pyo3::{pymodule, types::PyModule, Bound, PyResult};
use tissues_1d::aorta::{simulate_aorta_he_uniaxial_response, simulate_aorta_ve_uniaxial_response};
/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
#[pyo3(name = "models")]
mod rustputo {

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
    mod constitutive_laws {
        #[pymodule_export]
        use super::PyNeoHookean;
        use super::{Bound, PyAnyMethods, PyModule, PyResult, Python};
        #[pymodule_init]
        pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            Python::with_gil(|py| {
                py.import("sys")?
                    .getattr("modules")?
                    .set_item("rustputo.rust.models.constitutive_laws", m)
            })
        }
    }

    #[pymodule]
    mod testing {
        #[pymodule_export]
        use super::sum_as_string;
        use super::{Bound, PyAnyMethods, PyModule, PyResult, Python};
        #[pymodule_init]
        pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            Python::with_gil(|py| {
                py.import("sys")?
                    .getattr("modules")?
                    .set_item("rustputo.rust.models.testing", m)
            })
        }
    }

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        // constitutive_laws::init(m)?;
        // testing::init(m)?;
        Ok(())
    }
}
