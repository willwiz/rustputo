use std::fmt;

use ndarray::ShapeError;
use ndarray_linalg::error::LinalgError;
use pyo3::PyErr;
#[derive(Debug)]
pub enum PyError {
    Linalg(String),
    Math(String),
    Shape(String),
}

impl From<LinalgError> for PyError {
    fn from(err: LinalgError) -> PyError {
        PyError::Math(err.to_string())
    }
}

impl From<ShapeError> for PyError {
    fn from(err: ShapeError) -> PyError {
        PyError::Shape(err.to_string())
    }
}

impl From<String> for PyError {
    fn from(err: String) -> PyError {
        PyError::Shape(err)
    }
}

impl fmt::Display for PyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyError::Linalg(msg) => write!(f, "LinAlg Error: {}", msg),
            PyError::Math(msg) => write!(f, "Math Error: {}", msg),
            PyError::Shape(msg) => write!(f, "Shape Mismatch: {}", msg),
        }
    }
}

impl From<PyError> for PyErr {
    fn from(err: PyError) -> PyErr {
        match err {
            PyError::Math(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(msg),
            PyError::Shape(msg) => PyErr::new::<pyo3::exceptions::PyIndexError, _>(msg),
            PyError::Linalg(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
        }
    }
}
