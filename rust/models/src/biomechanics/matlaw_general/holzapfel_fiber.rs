use ndarray::Array2;

use crate::{
    biomechanics::model_traits::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, TriaxialPK2Stress, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
    utils::errors::PyError,
};

pub struct HolzapfelFiber {
    pub k: f64,
    pub b: f64,
    pub h: Array2<f64>,
}

impl HolzapfelFiber {
    pub fn new(k: f64, b: f64, h: Array2<f64>) -> Self {
        Self { k, b, h }
    }

    pub fn from_inplane_angle(k: f64, b: f64, theta: f64) -> Self {
        let fiber: [f64; 2] = [theta.cos(), theta.sin()];
        Self {
            k,
            b,
            h: Array2::from_shape_fn((2, 2), |(i, j)| fiber[i] * fiber[j]),
        }
    }

    pub fn from_spherical_angles(k: f64, b: f64, theta: f64, phi: f64) -> Self {
        let fiber: [f64; 3] = [
            theta.cos() * phi.cos(),
            theta.cos() * phi.sin(),
            theta.sin(),
        ];
        Self {
            k,
            b,
            h: Array2::from_shape_fn((3, 3), |(i, j)| fiber[i] * fiber[j]),
        }
    }
}

impl ComputeHyperelasticUniaxialPK2 for HolzapfelFiber {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        let i_f = strain.c - 1.0;
        UniaxialPK2Stress {
            stress: self.k * i_f * (self.b * i_f).powi(2).exp(),
            pressure: 0.0,
        }
    }
}

impl ComputeHyperelasticBiaxialPK2 for HolzapfelFiber {
    fn pk2(&self, strain: &BiaxialDeformation) -> BiaxialPK2Stress {
        let i_f = (&strain.c * &self.h).sum() - 1.0;
        BiaxialPK2Stress {
            stress: self.k * i_f * (self.b * i_f).powi(2).exp() * &self.h,
            pressure: 0.0,
        }
    }
}

impl ComputeHyperelasticTriaxialPK2 for HolzapfelFiber {
    fn pk2(&self, strain: &TriaxialDeformation) -> Result<TriaxialPK2Stress, PyError> {
        let i_f = (&strain.c * &self.h).sum() - 1.0;
        Ok(TriaxialPK2Stress {
            stress: self.k * i_f * (self.b * i_f).powi(2).exp() * &self.h,
        })
    }
}
