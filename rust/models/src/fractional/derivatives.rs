use ndarray::{Array, Dimension};

use crate::fractional::caputo::caputo_ndarray::CaputoStore;

pub trait LinearDerivative<const FDIM: usize> {
    fn init_with_dt_lin(&mut self, dt: f64) -> &mut Self;
    fn caputo_derivative_lin(&mut self, f: &[f64; FDIM], dt: f64) -> [f64; FDIM];
}

pub trait NDArrayLinearDerivative<D: Dimension, const NP: usize> {
    fn init_with_dt_lin(&mut self, dt: f64);
    fn caputo_derivative_lin(
        &mut self,
        f: &Array<f64, D>,
        dt: f64,
        store: &mut CaputoStore<D, NP>,
    ) -> Array<f64, D>;
}
