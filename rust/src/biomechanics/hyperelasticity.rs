use crate::kinematics::deformation::UniaxialDeformation;

pub struct UniaxialPK2Stress {
    pub stress: f64,
    pub pressure: f64,
}

pub trait ComputeUniaxialPK2 {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress;
}
