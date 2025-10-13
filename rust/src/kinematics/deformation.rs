use super::invariants::PseudoInvariants;
use ndarray::{Array2, ArrayView1, ArrayView2};
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::trace::Trace;
use ndarray_linalg::Determinant;

pub struct TriaxialDeformation {
    pub c: Array2<f64>,
    pub c_inv: Array2<f64>,
    pub i_1: f64,
    pub i_2: f64,
    pub i_3: f64,
    pub j_23: f64,
    pub det: f64,
}

pub struct BiaxialDeformation {
    pub c: Array2<f64>,
    pub c_inv: Array2<f64>,
    pub i_1: f64,
    pub i_2: f64,
    pub i_3: f64,
    pub i_n: f64,
    pub det: f64,
}

pub struct UniaxialDeformation {
    pub c: f64,
    pub c_inv: f64,
    pub i_1: f64,
    pub i_2: f64,
    pub i_3: f64,
    pub i_n: f64,
    pub det: f64,
}

impl TriaxialDeformation {
    pub fn new() -> Self {
        Self {
            c: Array2::eye(3),
            c_inv: Array2::eye(3),
            i_1: 3.0,
            i_2: 3.0,
            i_3: 1.0,
            j_23: 1.0,
            det: 1.0,
        }
    }

    pub fn precompute_from(&mut self, t_c: &ArrayView2<f64>) {
        self.c = t_c.to_owned();
        self.c_inv = t_c.inv().unwrap();
        self.det = t_c.det().unwrap();
        self.i_1 = self.c.trace().unwrap();
        self.i_2 = 0.5 * (self.i_1 * self.i_1 - (self.c.dot(&self.c)).trace().unwrap());
        self.i_3 = self.det;
        self.j_23 = 1.0 / self.det.powf(2.0 / 3.0);
    }
}

impl BiaxialDeformation {
    pub fn new() -> Self {
        Self {
            c: Array2::eye(2),
            c_inv: Array2::eye(2),
            i_1: 3.0,
            i_2: 3.0,
            i_3: 1.0,
            i_n: 1.0,
            det: 1.0,
        }
    }

    pub fn precompute_from(&mut self, t_c: &ArrayView2<f64>) {
        self.c = t_c.to_owned();
        self.c_inv = t_c.inv().unwrap();
        self.det = t_c.det().unwrap();
        self.i_n = 1.0 / self.det;
        self.i_1 = self.c.trace().unwrap() + self.i_n;
        self.i_2 = 0.5 * (self.i_1 * self.i_1 - (self.c.dot(&self.c)).trace().unwrap());
        self.i_3 = 1.0;
    }
}

impl UniaxialDeformation {
    pub fn new() -> Self {
        Self {
            c: 1.0,
            c_inv: 1.0,
            i_1: 3.0,
            i_2: 3.0,
            i_3: 1.0,
            i_n: 1.0,
            det: 1.0,
        }
    }

    pub fn precompute_from(&mut self, t_c: f64) {
        self.c = t_c;
        self.c_inv = 1.0 / self.c;
        self.det = self.c;
        self.i_n = 1.0 / self.det.sqrt();
        self.i_1 = self.c + 2.0 * self.i_n;
        self.i_2 = 2.0 * self.c * self.i_n + self.i_n * self.i_n;
        self.i_3 = 1.0;
    }
}

impl PseudoInvariants for TriaxialDeformation {
    fn i_4(&self, a0: &ArrayView1<f64>) -> f64 {
        let a0_c = self.c.dot(a0);
        a0.dot(&a0_c)
    }

    fn i_h(&self, a0: &ArrayView2<f64>) -> f64 {
        (&self.c * a0).sum()
    }
}
