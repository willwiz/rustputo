use crate::biomechanics::modeling::{ComputeHyperelasticUniaxialPK2, UniaxialPK2Stress};
use crate::kinematics::deformation::UniaxialDeformation;

pub struct SELinear {
    pub k: f64,
}

impl ComputeHyperelasticUniaxialPK2 for SELinear {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        UniaxialPK2Stress {
            stress: self.k * (0.5 * (strain.c + strain.i_n) - 1.0),
            pressure: 0.0,
        }
    }
}
