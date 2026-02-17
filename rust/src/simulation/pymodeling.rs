use super::FromNumpy;
use crate::biomechanics::model_traits::{
    ComputeHyperelasticUniaxialPK2, ComputeViscoelasticUniaxialPK2,
};
use crate::simulation::simulate::{
    simulate_hyperelastic_uniaxial_response, simulate_viscoelastic_uniaxial_response,
};
use numpy::PyReadonlyArray1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::{Bound, PyResult, Python};

pub fn pymodel_hyperelastic_uniaxial_response<'py, M>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    M: FromNumpy + ComputeHyperelasticUniaxialPK2,
{
    let p = parameters.as_array();
    let c = constants.as_array();
    let aorta_model = M::from_np(&p, &c)?;
    match simulate_hyperelastic_uniaxial_response(&aorta_model, &strain.as_array()) {
        Ok(result) => Ok(result.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(format!("Simulation failed: {}", e))),
    }
}

pub fn pymodel_viscoelastic_uniaxial_response<'py, M>(
    py: Python<'py>,
    parameters: PyReadonlyArray1<'py, f64>,
    constants: PyReadonlyArray1<'py, f64>,
    strain: PyReadonlyArray1<'py, f64>,
    dt: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    M: FromNumpy + ComputeViscoelasticUniaxialPK2,
{
    let p = parameters.as_array();
    let c = constants.as_array();
    let mut model = M::from_np(&p, &c)?;
    match simulate_viscoelastic_uniaxial_response(&mut model, &strain.as_array(), &dt.as_array()) {
        Ok(result) => Ok(result.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(format!("Simulation failed: {}", e))),
    }
}
