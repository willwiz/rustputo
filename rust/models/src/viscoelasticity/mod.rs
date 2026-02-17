pub mod caputo;
pub mod caputo_data;
pub mod derivatives;
pub mod maxwell_branch;
pub mod utils;
use ndarray::{Array, Ix2};
use std::usize;
pub struct StressState<const N: usize> {
    pub fprev: Array<f64, Ix2>,
    pub prony: [Array<f64, Ix2>; N],
}
