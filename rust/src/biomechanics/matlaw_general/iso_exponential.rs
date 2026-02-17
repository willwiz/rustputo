use crate::{
    biomechanics::model_traits::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, TriaxialPK2Stress, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
    utils::errors::PyError,
};
use ndarray::Array2;

pub struct IsoExponential {
    pub k: f64,
    pub b: f64,
}

impl IsoExponential {
    pub fn new(k: f64, b: f64) -> Self {
        Self { k, b }
    }
}

impl ComputeHyperelasticUniaxialPK2 for IsoExponential {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        let i_1 = strain.c + 2.0 / strain.c.sqrt();
        UniaxialPK2Stress {
            stress: self.k * (self.b * (i_1 - 3.0)).exp(),
            pressure: self.k * (self.b * (i_1 - 3.0)).exp(),
        }
    }
}

impl ComputeHyperelasticBiaxialPK2 for IsoExponential {
    fn pk2(&self, strain: &BiaxialDeformation) -> BiaxialPK2Stress {
        let i: Array2<f64> = Array2::eye(2);
        let i_1 = (&strain.c * &i).sum();
        BiaxialPK2Stress {
            stress: self.k * (self.b * (i_1 - 3.0)).exp() * i,
            pressure: self.k,
        }
    }
}

impl ComputeHyperelasticTriaxialPK2 for IsoExponential {
    fn pk2(&self, strain: &TriaxialDeformation) -> Result<TriaxialPK2Stress, PyError> {
        let i: Array2<f64> = Array2::eye(3);
        let i_1 = (&strain.c * &i).sum();
        Ok(TriaxialPK2Stress {
            stress: (self.k) * (self.b * (i_1 - 3.0)).exp() * i,
        })
    }
}
