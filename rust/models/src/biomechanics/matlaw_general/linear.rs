use crate::biomechanics::model_traits::{ComputeHyperelasticUniaxialPK2, UniaxialPK2Stress};
use crate::kinematics::deformation::UniaxialDeformation;
use ndarray::Array1;
pub struct PlanarLinear {
    pub k: f64,
    pub n: Array1<f64>,
}

impl PlanarLinear {
    pub fn new(k: f64) -> Self {
        Self {
            k,
            n: ndarray::array![0.0, 0.0, 1.0],
        }
    }
}

impl ComputeHyperelasticUniaxialPK2 for PlanarLinear {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        UniaxialPK2Stress {
            stress: self.k * (0.5 * (strain.c + strain.i_n) - 1.0),
            pressure: 0.0,
        }
    }
}
