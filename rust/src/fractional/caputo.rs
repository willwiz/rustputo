mod precomputed_data;
use std::{array, f64::consts::PI};

use precomputed_data::caputo_500::{CAPUTO15, CAPUTO9};

use crate::fractional::caputo_store::utils::interpolate_arr_1d;

pub struct CaputoData<const NP: usize> {
    pub b0: f64,
    pub beta: [f64; NP],
    pub tau: [f64; NP],
}

impl CaputoData<9> {
    pub fn new(alpha: f64, tf: f64) -> Self {
        let freq = 2.0 * PI / tf;
        let beta_exp = freq.powf(alpha);
        Self {
            b0: interpolate_arr_1d(&CAPUTO9.b0, alpha) * freq.powf(alpha - 1.0),
            beta: array::from_fn(|i| interpolate_arr_1d(&CAPUTO9.beta[i], alpha) * beta_exp),
            tau: array::from_fn(|i| interpolate_arr_1d(&CAPUTO9.tau[i], alpha) / freq),
        }
    }
}

impl CaputoData<15> {
    pub fn new(alpha: f64, tf: f64) -> Self {
        let freq = 2.0 * PI / tf;
        let beta_exp = freq.powf(alpha);
        Self {
            b0: interpolate_arr_1d(&CAPUTO15.b0, alpha) * freq.powf(alpha - 1.0),
            beta: array::from_fn(|i| interpolate_arr_1d(&CAPUTO15.beta[i], alpha) * beta_exp),
            tau: array::from_fn(|i| interpolate_arr_1d(&CAPUTO15.tau[i], alpha) / freq),
        }
    }
}
