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

use crate::example::{axpy_py, mult_py, sum_as_string};
use crate::tissues_1d::aorta::{
    simulate_aorta_he_uniaxial_response, simulate_aorta_ve_uniaxial_response,
};
use crate::tissues_3d::neohookean::PyNeoHookean;
use pyo3::{pymodule, types::PyModule, Bound, PyResult};
/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
#[pyo3(name = "rust")]
mod rust {

    use super::*;

    #[pymodule_export]
    use super::axpy_py;
    #[pymodule_export]
    use super::constitutive_laws;
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
    #[pymodule_export]
    use super::testing;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let modules = PyModule::import(m.py(), "sys")?.getattr("modules")?;
        modules.set_item("rustputo.rust", m)?;
        modules.set_item(
            "rustputo.rust.constitutive_laws",
            m.getattr("constitutive_laws")?,
        )?;
        modules.set_item("rustputo.rust.testing", m.getattr("testing")?)?;
        Ok(())
    }
}

#[pymodule(submodule)]
mod constitutive_laws {
    #[pymodule_export]
    use super::PyNeoHookean;
}
#[pymodule(submodule)]
mod testing {
    #[pymodule_export]
    use super::sum_as_string;
}
