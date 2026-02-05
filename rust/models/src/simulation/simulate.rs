use crate::{
    biomechanics::model_traits::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, ComputeViscoelasticBiaxialPK2,
        ComputeViscoelasticTriaxialPK2, ComputeViscoelasticUniaxialPK2, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
    kinematics::Precomputable,
    utils::errors::PyError,
};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

pub fn solve_uniaxial_pk2(model: &UniaxialPK2Stress, strain: &UniaxialDeformation) -> f64 {
    model.stress - model.pressure * strain.i_n * strain.c_inv
}

pub fn solve_biaxial_pk2(model: &BiaxialPK2Stress, strain: &BiaxialDeformation) -> Array2<f64> {
    &model.stress - model.pressure * strain.i_n * &strain.c_inv
}

pub fn simulate_hyperelastic_uniaxial_response<T>(
    tissue: &T,
    strain: &ArrayView1<f64>,
) -> Result<Array1<f64>, PyError>
where
    T: ComputeHyperelasticUniaxialPK2,
{
    let mut stress = Array1::<f64>::zeros(strain.raw_dim());
    let mut kin = UniaxialDeformation::new();
    for (i, &eps) in strain.iter().enumerate() {
        kin.precompute_from(&ArrayView2::from_shape((1, 1), &[eps])?)?;
        let pk2_stress = tissue.pk2(&kin);
        stress[i] = solve_uniaxial_pk2(&pk2_stress, &kin);
    }
    Ok(stress)
}

pub fn simulate_hyperelastic_biaxial_response<T>(
    tissue: &T,
    strain: &ArrayView3<f64>,
) -> Result<Array3<f64>, PyError>
where
    T: ComputeHyperelasticBiaxialPK2,
{
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin: BiaxialDeformation = BiaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps)?;
        let pk2_stress = tissue.pk2(&kin);
        stress
            .index_axis_mut(Axis(0), i)
            .assign(&solve_biaxial_pk2(&pk2_stress, &kin));
    }
    Ok(stress)
}

pub fn simulate_hyperelastic_triaxial_response<T>(
    tissue: &T,
    strain: &ArrayView3<f64>,
) -> Result<Array3<f64>, PyError>
where
    T: ComputeHyperelasticTriaxialPK2,
{
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin = TriaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps)?;
        let pk2_stress = tissue.pk2(&kin)?;
        stress.index_axis_mut(Axis(0), i).assign(&pk2_stress.stress);
    }
    Ok(stress)
}

pub fn simulate_viscoelastic_uniaxial_response<T>(
    tissue: &mut T,
    strain: &ArrayView1<f64>,
    dt: &ArrayView1<f64>,
) -> Result<Array1<f64>, PyError>
where
    T: ComputeViscoelasticUniaxialPK2,
{
    let mut stress = Array1::<f64>::zeros(strain.raw_dim());
    let mut kin = UniaxialDeformation::new();
    for (i, &eps) in strain.iter().skip(1).enumerate() {
        kin.precompute_from(&ArrayView2::from_shape((1, 1), &[eps])?)?;
        let _dt = match dt.get(i) {
            Some(val) => *val,
            None => {
                return Err(PyError::Shape(
                    "ValueError: dt array is shorter than strain array.".to_string(),
                ));
            }
        };
        let pk2_stress = tissue.pk2(&kin, _dt);
        stress[i] = solve_uniaxial_pk2(&pk2_stress, &kin);
    }
    Ok(stress)
}

pub fn simulate_viscoelastic_biaxial_response<T>(
    tissue: &mut T,
    strain: &ArrayView3<f64>,
    dt: &ArrayView1<f64>,
) -> Result<Array3<f64>, PyError>
where
    T: ComputeViscoelasticBiaxialPK2,
{
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin = BiaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps)?;
        let _dt = match dt.get(i) {
            Some(val) => *val,
            None => {
                return Err(PyError::Shape(
                    "ValueError: dt array is shorter than strain array.".to_string(),
                ));
            }
        };
        let pk2_stress = tissue.pk2(&kin, _dt);
        stress
            .index_axis_mut(Axis(0), i)
            .assign(&solve_biaxial_pk2(&pk2_stress, &kin));
    }
    Ok(stress)
}

pub fn simulate_viscoelastic_triaxial_response<T>(
    tissue: &mut T,
    strain: &ArrayView3<f64>,
    dt: &ArrayView1<f64>,
) -> Result<Array3<f64>, PyError>
where
    T: ComputeViscoelasticTriaxialPK2,
{
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin = TriaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps)?;
        let _dt = match dt.get(i) {
            Some(val) => *val,
            None => {
                return Err(PyError::Shape(
                    "ValueError: dt array is shorter than strain array.".to_string(),
                ));
            }
        };
        let pk2_stress = tissue.pk2(&kin, _dt);
        stress.index_axis_mut(Axis(0), i).assign(&pk2_stress.stress);
    }
    Ok(stress)
}
