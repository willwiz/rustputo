use crate::biomechanics::matlaw_uniaxial::linear::SELinear;
use crate::fractional::derivatives::LinearDerivative;
use crate::kinematics::deformation::UniaxialDeformation;
use crate::{
    biomechanics::{
        matlaw_uniaxial::{exponential::HolzapfelUniaxial, neohookean::NeoHookean},
        modeling::{
            ComputeHyperelasticUniaxialPK2, ComputeViscoelasticUniaxialPK2, UniaxialPK2Stress,
        },
    },
    fractional::caputo::caputo_internal::CaputoInternal,
};

pub struct AortaUniaxial {
    pub matrix: NeoHookean,
    pub elastin: SELinear,
    pub collagen: HolzapfelUniaxial,
}

pub struct AortaUniaxialViscoelastic {
    pub elastic: AortaUniaxial,
    pub caputo: CaputoInternal<1, 9>,
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

impl AortaUniaxialViscoelastic {
    pub fn new(
        matrix_k: f64,
        elastin_k: f64,
        collagen_k: f64,
        collagen_b: f64,
        alpha: f64,
        tf: f64,
    ) -> Self {
        Self {
            elastic: AortaUniaxial::new(matrix_k, elastin_k, collagen_k, collagen_b),
            caputo: CaputoInternal::<1, 9>::new(alpha, 0.0, tf),
        }
    }
}

impl ComputeHyperelasticUniaxialPK2 for AortaUniaxial {
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

impl ComputeViscoelasticUniaxialPK2 for AortaUniaxialViscoelastic {
    fn pk2(&mut self, strain: &UniaxialDeformation, dt: f64) -> UniaxialPK2Stress {
        let matrix_stress = self.elastic.matrix.pk2(&strain);
        let elastin_stress = self.elastic.elastin.pk2(&strain);
        let collagen_stress = self.elastic.collagen.pk2(&strain);
        let viscous_stress = self
            .caputo
            .init_with_dt_lin(dt)
            .caputo_derivative_lin(&[collagen_stress.stress], dt);

        UniaxialPK2Stress {
            stress: matrix_stress.stress + elastin_stress.stress + viscous_stress[0],
            pressure: matrix_stress.pressure,
        }
    }
}
