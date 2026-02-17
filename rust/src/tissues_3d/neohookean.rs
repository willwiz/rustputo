use crate::biomechanics::model_traits::ComputeHyperelasticTriaxialPK2;
use crate::kinematics::Precomputable;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
#[pyclass]
#[pyo3(name = "NeoHookean")]
pub struct PyNeoHookean {
    k: crate::biomechanics::matlaw_general::NeoHookean,
}

#[pymethods]
impl PyNeoHookean {
    #[new]
    pub fn new(k: f64) -> Self {
        Self {
            k: crate::biomechanics::matlaw_general::NeoHookean::new(k),
        }
    }

    fn simulate<'py>(
        &self,
        py: Python<'py>,
        f: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = f.as_array();
        let mut deformation = crate::kinematics::deformation::TriaxialDeformation::new();
        deformation.precompute_from(&f)?;
        let stress = self.k.pk2(&deformation)?;
        Ok(PyArray2::from_owned_array(py, stress.stress))
    }
}
