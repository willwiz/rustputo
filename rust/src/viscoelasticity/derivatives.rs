use ndarray::{Array, Dimension, ShapeBuilder};
use std::array;

pub struct StressHistory<D: Dimension, const NP: usize> {
    pub qk: [Array<f64, D>; NP],
    pub fprev: Array<f64, D>,
}

pub trait LinearDerivative<const FDIM: usize> {
    fn init_with_dt_lin(&mut self, dt: f64) -> &mut Self;
    fn derivative_linear(&mut self, f: &[f64; FDIM], dt: f64) -> [f64; FDIM];
}

pub trait NDArrayLinearDerivative<D: Dimension, const NP: usize> {
    fn init_with_dt_lin(&mut self, dt: f64);
    fn derivative_linear(
        &mut self,
        f: &Array<f64, D>,
        dt: f64,
        store: &mut StressHistory<D, NP>,
    ) -> Array<f64, D>;
}

impl<D: Dimension, const NP: usize> StressHistory<D, NP> {
    pub fn new<Sh: ShapeBuilder<Dim = D> + Copy>(shape: Sh) -> Self {
        {
            Self {
                qk: array::from_fn(|_| Array::<f64, D>::zeros(shape)),
                fprev: Array::<f64, D>::zeros(shape),
            }
        }
    }
}
