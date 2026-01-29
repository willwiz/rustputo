mod precomputed_data;
use std::f64::consts::PI;

use precomputed_data::caputo_500::{CAPUTO15, CAPUTO9};

use crate::{fractional::utils::interpolate_arr_1d, utils::errors::PyError};

pub struct CaputoData<const NP: usize> {
    pub b0: f64,
    pub beta: [f64; NP],
    pub tau: [f64; NP],
}

impl CaputoData<9> {
    pub fn new(alpha: f64, tf: f64) -> Result<Self, PyError> {
        let freq = 2.0 * PI / tf;
        let beta_exp = freq.powf(alpha);
        let b0 = interpolate_arr_1d(&CAPUTO9.b0, alpha)? * freq.powf(alpha - 1.0);
        let mut beta = [0.0; 9];
        for i in 0..9 {
            beta[i] = interpolate_arr_1d(&CAPUTO9.beta[i], alpha)? * beta_exp;
        }
        let mut tau = [0.0; 9];
        for i in 0..9 {
            tau[i] = interpolate_arr_1d(&CAPUTO9.tau[i], alpha)? / freq;
        }
        Ok(Self { b0, beta, tau })
    }
}

impl CaputoData<15> {
    pub fn new(alpha: f64, tf: f64) -> Result<Self, PyError> {
        let freq = 2.0 * PI / tf;
        let beta_exp = freq.powf(alpha);
        let b0 = interpolate_arr_1d(&CAPUTO15.b0, alpha)? * freq.powf(alpha - 1.0);
        let mut beta = [0.0; 15];
        for i in 0..15 {
            beta[i] = interpolate_arr_1d(&CAPUTO15.beta[i], alpha)? * beta_exp;
        }
        let mut tau = [0.0; 15];
        for i in 0..15 {
            tau[i] = interpolate_arr_1d(&CAPUTO15.tau[i], alpha)? / freq;
        }
        Ok(Self { b0, beta, tau })
    }
}
