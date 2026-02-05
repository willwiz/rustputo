use ndarray::Array2;

use crate::{
    biomechanics::model_traits::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, TriaxialPK2Stress, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
    utils::errors::PyError,
};

pub struct NeoHookean {
    pub k: f64,
}

impl NeoHookean {
    pub fn new(k: f64) -> Self {
        Self { k }
    }
}

impl ComputeHyperelasticUniaxialPK2 for NeoHookean {
    fn pk2(&self, _strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        UniaxialPK2Stress {
            stress: self.k,
            pressure: self.k,
        }
    }
}

impl ComputeHyperelasticBiaxialPK2 for NeoHookean {
    fn pk2(&self, _strain: &BiaxialDeformation) -> BiaxialPK2Stress {
        BiaxialPK2Stress {
            stress: self.k * Array2::eye(2),
            pressure: self.k,
        }
    }
}

impl ComputeHyperelasticTriaxialPK2 for NeoHookean {
    fn pk2(&self, strain: &TriaxialDeformation) -> Result<TriaxialPK2Stress, PyError> {
        Ok(TriaxialPK2Stress {
            stress: (self.k * strain.j_23)
                * (Array2::eye(3) - (1.0 / 3.0) * strain.i_1 * &strain.c_inv),
        })
    }
}
