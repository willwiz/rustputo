use crate::biomechanics::hyperelasticity::{ComputeUniaxialPK2, UniaxialPK2Stress};
use crate::kinematics::deformation::UniaxialDeformation;

pub struct NeoHookean {
    pub k: f64,
}

impl ComputeUniaxialPK2 for NeoHookean {
    fn pk2(&self, _strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        UniaxialPK2Stress {
            stress: self.k,
            pressure: self.k,
        }
    }
}
