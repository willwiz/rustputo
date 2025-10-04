use crate::biomechanics::{
    hyperelasticity::{ComputeUniaxialPK2, UniaxialPK2Stress},
    matlaw_uniaxial::{exponential::HolzapfelUniaxial, linear::SELinear, neohookean::NeoHookean},
};
use crate::kinematics::deformation::UniaxialDeformation;

pub struct AortaUniaxial {
    pub matrix: NeoHookean,
    pub elastin: SELinear,
    pub collagen: HolzapfelUniaxial,
}

impl AortaUniaxial {
    pub fn new(matrix_k: f64, elastin_k: f64, collagen_k: f64, collagen_b: f64) -> Self {
        Self {
            matrix: NeoHookean { k: matrix_k },
            elastin: SELinear { k: elastin_k },
            collagen: HolzapfelUniaxial {
                k: collagen_k,
                b: collagen_b,
            },
        }
    }
}

impl ComputeUniaxialPK2 for AortaUniaxial {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        let matrix_stress = self.matrix.pk2(&strain);
        let elastin_stress = self.elastin.pk2(&strain);
        let collagen_stress = self.collagen.pk2(&strain);

        UniaxialPK2Stress {
            stress: matrix_stress.stress + elastin_stress.stress + collagen_stress.stress,
            pressure: matrix_stress.pressure + elastin_stress.pressure + collagen_stress.pressure,
        }
    }
}
