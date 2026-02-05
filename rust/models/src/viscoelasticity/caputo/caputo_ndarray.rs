use ndarray::{Array, Dimension};

use crate::{
    utils::errors::PyError,
    viscoelasticity::{
        caputo_data::CaputoData,
        derivatives::{NDArrayLinearDerivative, StressHistory},
    },
};

pub struct CaputoInternal<const NP: usize> {
    pub(super) caputo: CaputoData<NP>,
    pub delta: f64,
    pub dt: f64,
    pub(super) c0: f64,
    pub(super) k0: f64,
    pub(super) k1: f64,
    pub(super) e2: [f64; NP],
    pub(super) bek: [f64; NP],
}

impl CaputoInternal<9> {
    pub fn new(alpha: f64, delta: f64, tf: f64) -> Result<Self, PyError> {
        Ok(Self {
            caputo: CaputoData::<9>::new(alpha, tf)?,
            delta: delta,
            dt: 0.0,
            c0: 0.0,
            k0: 0.0,
            k1: 1.0,
            e2: [0.0; 9],
            bek: [0.0; 9],
        })
    }
}

impl CaputoInternal<15> {
    pub fn new(alpha: f64, delta: f64, tf: f64) -> Result<Self, PyError> {
        Ok(Self {
            caputo: CaputoData::<15>::new(alpha, tf)?,
            delta: delta,
            dt: 0.0,
            c0: 0.0,
            k0: 0.0,
            k1: 1.0,
            e2: [0.0; 15],
            bek: [0.0; 15],
        })
    }
}

impl<D: Dimension, const NP: usize> NDArrayLinearDerivative<D, NP> for CaputoInternal<NP> {
    fn init_with_dt_lin(&mut self, dt: f64) {
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
    }

    fn derivative_linear(
        &mut self,
        f: &Array<f64, D>,
        dt: f64,
        store: &mut StressHistory<D, NP>,
    ) -> Array<f64, D> {
        if (dt - self.dt).abs() > f64::EPSILON {
            <CaputoInternal<NP> as NDArrayLinearDerivative<D, NP>>::init_with_dt_lin(self, dt);
        }

        let df: Array<f64, D> = f - &store.fprev;
        store.fprev = f.clone();
        let mut v: Array<f64, D> = Array::zeros(f.raw_dim());
        for k in 0..NP {
            store.qk[k] = self.e2[k] * store.qk[k].clone() + self.bek[k] * &df;
            v = v + &store.qk[k];
        }
        v
    }
}
