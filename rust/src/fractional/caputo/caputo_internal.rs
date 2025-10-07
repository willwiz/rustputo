use crate::fractional::caputo_data::CaputoData;
use crate::fractional::derivatives::LinearDerivative;
use core::array;

pub struct CaputoInternal<const FDIM: usize, const NP: usize> {
    pub(super) caputo: CaputoData<NP>,
    pub delta: f64,
    pub dt: f64,
    pub(super) c0: f64,
    pub(super) k0: f64,
    pub(super) k1: f64,
    pub(super) e2: [f64; NP],
    pub(super) bek: [f64; NP],
    pub(super) qk: [[f64; FDIM]; NP],
    pub(super) fprev: [f64; FDIM],
}

impl<const FDIM: usize> CaputoInternal<FDIM, 9> {
    pub fn new(alpha: f64, delta: f64, tf: f64) -> Self {
        Self {
            caputo: CaputoData::<9>::new(alpha, tf),
            delta: delta,
            dt: 0.0,
            c0: 0.0,
            k0: 0.0,
            k1: 1.0,
            e2: [0.0; 9],
            bek: [0.0; 9],
            qk: [[0.0; FDIM]; 9],
            fprev: [0.0; FDIM],
        }
    }
}

impl<const FDIM: usize> CaputoInternal<FDIM, 15> {
    pub fn new(alpha: f64, delta: f64, tf: f64) -> Self {
        Self {
            caputo: CaputoData::<15>::new(alpha, tf),
            delta: delta,
            dt: 0.0,
            c0: 0.0,
            k0: 0.0,
            k1: 1.0,
            e2: [0.0; 15],
            bek: [0.0; 15],
            qk: [[0.0; FDIM]; 15],
            fprev: [0.0; FDIM],
        }
    }
}

impl<const FDIM: usize, const NP: usize> LinearDerivative<FDIM> for CaputoInternal<FDIM, NP> {
    fn init_with_dt_lin(&mut self, dt: f64) -> &mut Self {
        self.c0 = self.caputo.b0 / dt;
        self.k0 = self.caputo.b0 / dt;
        self.dt = dt;
        for k in 0..NP {
            self.e2[k] = self.caputo.tau[k] / (self.caputo.tau[k] + dt);
            self.bek[k] = self.caputo.beta[k] * self.e2[k];
            self.k0 += self.bek[k];
        }
        self.k0 = self.delta * self.k0;
        self.k1 = self.k0 + 1.0;
        // println!(
        //     "Caputo initialized with dt = {}, b0 = {}",
        //     dt, self.caputo.b0
        // );
        self
    }

    fn caputo_derivative_lin(&mut self, fval: &[f64; FDIM], dt: f64) -> [f64; FDIM] {
        if (dt - self.dt).abs() > f64::EPSILON {
            self.init_with_dt_lin(dt);
        }
        let df: [f64; FDIM] = array::from_fn(|i| fval[i] - self.fprev[i]);
        self.fprev = *fval;
        let mut v: [f64; FDIM] = [0.0; FDIM];
        for k in 0..self.qk.len() {
            // Update qk with the new values for the current iteration
            self.qk[k] = array::from_fn(|i| self.e2[k] * self.qk[k][i] + self.bek[k] * df[i]);
            // compute the result of the fractional derivative
            v = array::from_fn(|i| v[i] + self.qk[k][i]);
        }
        v
    }
}
