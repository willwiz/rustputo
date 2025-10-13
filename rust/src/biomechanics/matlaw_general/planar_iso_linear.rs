use ndarray::Array2;

use crate::{
    biomechanics::modeling::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, TriaxialPK2Stress, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
};

pub struct PlanarIsoLinear {
    pub k: f64,
    pub h: Array2<f64>,
}

impl PlanarIsoLinear {
    pub fn new<const D: usize>(k: f64, h: [f64; D]) -> Self {
        Self {
            k,
            h: 0.5 * (Array2::eye(D) - Array2::from_shape_fn((D, D), |(i, j)| h[i] * h[j])),
        }
    }
}

impl ComputeHyperelasticUniaxialPK2 for PlanarIsoLinear {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        UniaxialPK2Stress {
            stress: self.k * (0.5 * (strain.c + strain.i_n) - 1.0),
            pressure: 0.0,
        }
    }
}

impl ComputeHyperelasticBiaxialPK2 for PlanarIsoLinear {
    fn pk2(&self, strain: &BiaxialDeformation) -> BiaxialPK2Stress {
        BiaxialPK2Stress {
            stress: self.k * (0.5 * (&strain.c * &self.h).sum() - 1.0) * &self.h,
            pressure: 0.0,
        }
    }
}

impl ComputeHyperelasticTriaxialPK2 for PlanarIsoLinear {
    fn pk2(&self, strain: &TriaxialDeformation) -> TriaxialPK2Stress {
        TriaxialPK2Stress {
            stress: self.k * (0.5 * (&strain.c * &self.h).sum() - 1.0) * &self.h,
        }
    }
}
