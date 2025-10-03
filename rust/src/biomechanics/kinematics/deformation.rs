use ndarray::Array2;
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::trace::Trace;
use ndarray_linalg::Determinant;
pub struct Deformation {
    pub r_c: Array2<f64>,
    pub i_c: Array2<f64>,
    pub i_n: f64,
    pub i_1: f64,
    pub det: f64,
}

pub trait DeformUpdate {
    fn deform(&mut self, v_f: &Array2<f64>);
}

impl DeformUpdate for Deformation {
    fn deform(&mut self, v_f: &Array2<f64>) {
        self.r_c = v_f.t().dot(v_f);
        self.i_c = self.r_c.inv().unwrap();
        self.det = self.r_c.det().unwrap();
        self.i_1 = self.i_c.trace().unwrap();
        self.i_n = 1.0 / self.det;
    }
}
