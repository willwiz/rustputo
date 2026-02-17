use crate::biomechanics::matlaw_uniaxial::linear::SELinear;
use crate::kinematics::deformation::UniaxialDeformation;
use crate::utils::errors::PyError;
use crate::viscoelasticity::derivatives::LinearDerivative;
use crate::{
    biomechanics::{
        matlaw_general::{HolzapfelFiber, NeoHookean},
        model_traits::{
            ComputeHyperelasticUniaxialPK2, ComputeViscoelasticUniaxialPK2, UniaxialPK2Stress,
        },
    },
    simulation::FromNumpy,
    viscoelasticity::caputo::caputo_internal::CaputoInternal,
};
use ndarray::ArrayView1;

pub struct AortaUniaxial {
    pub matrix: NeoHookean,
    pub elastin: SELinear,
    pub collagen: HolzapfelFiber,
}

pub struct AortaUniaxialViscoelastic {
    pub elastic: AortaUniaxial,
    pub caputo: CaputoInternal<1, 9>,
}

pub struct AortaParameters {
    matrix_k: f64,
    elastin_k: f64,
    collagen_k: f64,
    collagen_b: f64,
}

pub struct AortaViscoelasticParameters {
    matrix_k: f64,
    elastin_k: f64,
    collagen_k: f64,
    collagen_b: f64,
    alpha: f64,
    tf: f64,
}

impl AortaUniaxial {
    pub fn new(pars: AortaParameters) -> Result<Self, PyError> {
        Ok(Self {
            matrix: NeoHookean { k: pars.matrix_k },
            elastin: SELinear { k: pars.elastin_k },
            collagen: HolzapfelFiber {
                k: pars.collagen_k,
                b: pars.collagen_b,
                h: ndarray::array![[1.0, 0.0], [0.0, 0.0]],
            },
        })
    }
}

impl AortaUniaxialViscoelastic {
    pub fn new(pars: AortaViscoelasticParameters) -> Result<Self, PyError> {
        Ok(Self {
            elastic: AortaUniaxial::new(AortaParameters {
                matrix_k: pars.matrix_k,
                elastin_k: pars.elastin_k,
                collagen_k: pars.collagen_k,
                collagen_b: pars.collagen_b,
            })?,
            caputo: CaputoInternal::<1, 9>::new(pars.alpha, 0.0, pars.tf)?,
        })
    }
}

impl FromNumpy for AortaUniaxial {
    fn from_np(pars: &ArrayView1<f64>, _constants: &ArrayView1<f64>) -> Result<Self, PyError> {
        if pars.len() < 4 {
            return Err(PyError::Shape(format!(
                "Expected 4 parameters for AortaUniaxial model, got {}",
                pars.len()
            )));
        }
        let p = AortaParameters {
            matrix_k: pars[0],
            elastin_k: pars[1],
            collagen_k: pars[2],
            collagen_b: pars[3],
        };
        Self::new(p)
    }
}

impl FromNumpy for AortaUniaxialViscoelastic {
    fn from_np(pars: &ArrayView1<f64>, constants: &ArrayView1<f64>) -> Result<Self, PyError> {
        if pars.len() < 5 {
            return Err(PyError::Shape(format!(
                "Expected 5 parameters for AortaViscoelastic model, got {}",
                pars.len()
            )));
        }
        if constants.len() != 1 {
            return Err(PyError::Shape(format!(
                "Expected 1 constant for AortaViscoelastic model, got {}",
                constants.len()
            )));
        }
        let p = AortaViscoelasticParameters {
            matrix_k: pars[0],
            elastin_k: pars[1],
            collagen_k: pars[2],
            collagen_b: pars[3],
            alpha: pars[4],
            tf: constants[0],
        };
        Self::new(p)
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
            .derivative_linear(&[collagen_stress.stress], dt);

        UniaxialPK2Stress {
            stress: matrix_stress.stress + elastin_stress.stress + viscous_stress[0],
            pressure: matrix_stress.pressure,
        }
    }
}
