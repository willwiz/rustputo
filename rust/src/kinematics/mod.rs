pub mod deformation;
pub mod invariants;
use crate::utils::errors::PyError;
use ndarray::ArrayView2;

pub trait Precomputable {
    fn precompute_from(&mut self, t_c: &ArrayView2<f64>) -> Result<(), PyError>;
}
