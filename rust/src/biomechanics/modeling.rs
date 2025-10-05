use crate::kinematics::deformation::UniaxialDeformation;

pub struct UniaxialPK2Stress {
    pub stress: f64,
    pub pressure: f64,
}

pub trait ComputeHyperelasticUniaxialPK2 {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress;
}

pub trait ComputeViscoelasticUniaxialPK2 {
    fn pk2(&mut self, strain: &UniaxialDeformation, dt: f64) -> UniaxialPK2Stress;
}
