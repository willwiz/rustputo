use ndarray::{ArrayView1, ArrayView2};

pub trait PseudoInvariants {
    fn i_4(&self, a0: &ArrayView1<f64>) -> f64;
    fn i_h(&self, a0: &ArrayView2<f64>) -> f64;
}
