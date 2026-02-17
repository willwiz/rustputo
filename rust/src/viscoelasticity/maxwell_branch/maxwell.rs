use ndarray::{Array, Dimension};

use crate::viscoelasticity::derivatives::{NDArrayLinearDerivative, StressHistory};

pub struct MaxwellBranch {
    pub tau: f64,
    dt: f64,
    decay: f64,
}

pub struct MaxwellState<D: Dimension> {
    pub fprev: Array<f64, D>,
    pub store: Array<f64, D>,
}

impl<D: Dimension> NDArrayLinearDerivative<D, 1> for MaxwellBranch {
    fn init_with_dt_lin(&mut self, dt: f64) {
        if (dt - self.dt).abs() < f64::EPSILON {
            return;
        }
        self.dt = dt;
        self.decay = self.tau / (self.tau + dt);
    }

    fn derivative_linear(
        &mut self,
        f: &Array<f64, D>,
        dt: f64,
        store: &mut StressHistory<D, 1>,
    ) -> Array<f64, D> {
        <MaxwellBranch as NDArrayLinearDerivative<D, 1>>::init_with_dt_lin(self, dt);
        let res = self.decay * (&store.qk[0] + f - &store.fprev);
        store.fprev = f.clone();
        store.qk[0] = res.clone();
        res
    }
}
