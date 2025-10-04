mod biomechanics;
mod example;
mod fractional;
mod kinematics;
mod linalg;
mod tissues_1d;

use linalg::optimization::linear::lgres_mat_py;

use example::{axpy_py, mult_py, sum_as_string};
use pyo3::{pymodule, types::PyModule, Bound, PyResult};

#[cfg(test)]
static GLB_PRES_TOL: f64 = 1.0e-12;

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

#[cfg(test)]
mod tests {
    use crate::fractional::{
        caputo_store::caputo_internal::CaputoInternal, derivatives::LinearDerivative,
    };

    use super::GLB_PRES_TOL;

    #[test]
    fn caputo_works() {
        let mut caputo_init = CaputoInternal::<3, 9>::new(0.2, 0.0, 1.0);
        let fvals = [1.0, 2.0, 3.0];
        let dfvals = caputo_init
            .init_with_dt_lin(0.1)
            .caputo_derivative_lin(&fvals, 0.1);
        print!("dfvals: {:?}\n", dfvals);
        let benchmark = [0.2855326161084972, 0.5710652322169943, 0.8565978483254915];
        let check = benchmark
            .iter()
            .zip(dfvals.iter())
            .all(|(a, b)| (a - b).abs() < GLB_PRES_TOL);
        assert_eq!(check, true);
    }
}
