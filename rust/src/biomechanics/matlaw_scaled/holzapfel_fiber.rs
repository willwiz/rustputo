use ndarray::Array2;

use crate::{
    biomechanics::model_traits::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, TriaxialPK2Stress, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
};

pub struct HolzapfelFiberScaled {
    pub k: f64,
    pub b: f64,
    pub h: Array2<f64>,
    pub scale: f64,
}

impl HolzapfelFiberScaled {
    pub fn new(k: f64, b: f64, h: Array2<f64>, scale: f64) -> Self {
        Self { k, b, h, scale }
    }

    pub fn from_inplane_angle(k: f64, b: f64, theta: f64, max_strain: Array2<f64>) -> Self {
        let fiber: [f64; 2] = [theta.cos(), theta.sin()];
        let fiber_h = Array2::from_shape_fn((2, 2), |(i, j)| fiber[i] * fiber[j]);
        Self {
            k,
            b,
            h: fiber_h,
            scale: (&fiber_h * &max_strain).sum(),
        }
    }

    pub fn from_spherical_angles(k: f64, b: f64, theta: f64, phi: f64) -> Self {
        let fiber: [f64; 3] = [
            theta.cos() * phi.cos(),
            theta.cos() * phi.sin(),
            theta.sin(),
        ];
        let fiber_h = Array2::from_shape_fn((3, 3), |(i, j)| fiber[i] * fiber[j]);

        Self {
            k,
            b,
            h: fiber_h,
            scale: (&fiber_h * &max_strain).sum(),
        }
    }
}

impl ComputeHyperelasticUniaxialPK2 for HolzapfelFiberScaled {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        let i_f = strain.c - 1.0;
        UniaxialPK2Stress {
            stress: self.k * i_f * (self.b * i_f).powi(2).exp(),
            pressure: 0.0,
        }
    }
}

impl ComputeHyperelasticBiaxialPK2 for HolzapfelFiberScaled {
    fn pk2(&self, strain: &BiaxialDeformation) -> BiaxialPK2Stress {
        let i_f = (&strain.c * &self.h).sum() - 1.0;
        BiaxialPK2Stress {
            stress: self.k * i_f * (self.b * i_f).powi(2).exp() * &self.h,
            pressure: 0.0,
        }
    }
}

impl ComputeHyperelasticTriaxialPK2 for HolzapfelFiberScaled {
    fn pk2(&self, strain: &TriaxialDeformation) -> TriaxialPK2Stress {
        let i_f = (&strain.c * &self.h).sum() - 1.0;
        TriaxialPK2Stress {
            stress: self.k * i_f * (self.b * i_f).powi(2).exp() * &self.h,
        }
    }
}
