use crate::kinematics::deformation::UniaxialDeformation;

pub struct UniaxialPK2Stress {
    pub stress: f64,
    pub pressure: f64,
}

pub trait ComputeUniaxialPK2 {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress;
}

pub fn solve_uniaxial_pk2(model: &UniaxialPK2Stress, strain: &UniaxialDeformation) -> f64 {
    return model.stress - model.pressure * strain.i_n * strain.c_inv;
}
