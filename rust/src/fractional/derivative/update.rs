use super::intermediates::CaputoInternalArray;
use super::utils::interpolate_arr_1d;
use core::array;
pub trait CaputoInitLin<const FDIM: usize> {
    fn init_with_dt_lin(&mut self, dt: f64) -> &Self;
    fn caputo_iter(&mut self, fval: &[f64; FDIM], dt: f64) -> [f64; FDIM];
}

impl<const FDIM: usize, const NP: usize> CaputoInitLin<FDIM> for CaputoInternalArray<FDIM, NP> {
    fn init_with_dt_lin(&mut self, dt: f64) -> &Self {
        let beta0 = interpolate_arr_1d(&self.params.b0, self.alpha);
        self.c0 = beta0 / dt;
        self.k0 = beta0 / dt;
        self.dt = dt;
        for k in 0..NP {
            let beta = interpolate_arr_1d(&self.params.beta[k], self.alpha);
            let tau = interpolate_arr_1d(&self.params.tau[k], self.alpha);
            self.e2[k] = tau / (tau + dt);
            self.bek[k] = beta * self.e2[k];
            self.k0 += self.bek[k];
        }
        self.k0 = self.delta * self.k0;
        self.k1 = self.k0 + 1.0;
        self
    }

    fn caputo_iter(&mut self, fval: &[f64; FDIM], dt: f64) -> [f64; FDIM] {
        if (dt - self.dt).abs() > f64::EPSILON {
            self.init_with_dt_lin(dt);
        }
        let df: [f64; FDIM] = array::from_fn(|i| (fval[i] - self.fprev[i]));
        self.fprev = *fval;
        let mut v: [f64; FDIM] = [0.0; FDIM];
        for k in 0..self.params.n_p {
            // Update qk with the new values for the current iteration
            self.qk[k] = array::from_fn(|i| self.e2[k] * self.qk[k][i] + self.bek[k] * df[i]);
            // compute the result of the fractional derivative
            v = array::from_fn(|i| (v[i] + self.qk[k][i]));
        }
        v
    }
}
