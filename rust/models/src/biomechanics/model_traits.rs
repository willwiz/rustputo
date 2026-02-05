use ndarray::Array2;

use crate::{
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
    utils::errors::PyError,
};

pub struct UniaxialPK2Stress {
    pub stress: f64,
    pub pressure: f64,
}

pub struct BiaxialPK2Stress {
    pub stress: Array2<f64>,
    pub pressure: f64,
}

pub struct TriaxialPK2Stress {
    pub stress: Array2<f64>,
}

pub trait ParseParameters {}

pub trait ComputeHyperelasticUniaxialPK2 {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress;
}

pub trait ComputeHyperelasticBiaxialPK2 {
    fn pk2(&self, strain: &BiaxialDeformation) -> BiaxialPK2Stress;
}

pub trait ComputeHyperelasticTriaxialPK2 {
    fn pk2(&self, strain: &TriaxialDeformation) -> Result<TriaxialPK2Stress, PyError>;
}

pub trait ComputeViscoelasticUniaxialPK2 {
    fn pk2(&mut self, strain: &UniaxialDeformation, dt: f64) -> UniaxialPK2Stress;
}

pub trait ComputeViscoelasticBiaxialPK2 {
    fn pk2(&mut self, strain: &BiaxialDeformation, dt: f64) -> BiaxialPK2Stress;
}
pub trait ComputeViscoelasticTriaxialPK2 {
    fn pk2(&mut self, strain: &TriaxialDeformation, dt: f64) -> TriaxialPK2Stress;
}
