use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView3, Axis};

use crate::{
    biomechanics::model_traits::{
        BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
        ComputeHyperelasticUniaxialPK2, ComputeViscoelasticBiaxialPK2,
        ComputeViscoelasticTriaxialPK2, ComputeViscoelasticUniaxialPK2, UniaxialPK2Stress,
    },
    kinematics::deformation::{BiaxialDeformation, TriaxialDeformation, UniaxialDeformation},
};

pub fn solve_uniaxial_pk2(model: &UniaxialPK2Stress, strain: &UniaxialDeformation) -> f64 {
    return model.stress - model.pressure * strain.i_n * strain.c_inv;
}

pub fn solve_biaxial_pk2(model: &BiaxialPK2Stress, strain: &BiaxialDeformation) -> Array2<f64> {
    return &model.stress - model.pressure * strain.i_n * &strain.c_inv;
}

// Solving triaxial PK2 requires special consideration for compressibility

pub fn simulate_hyperelastic_uniaxial_response<T: ComputeHyperelasticUniaxialPK2>(
    tissue: &T,
    strain: &ArrayView1<f64>,
) -> Array1<f64> {
    let mut stress = Array1::<f64>::zeros(strain.raw_dim());
    let mut kin = UniaxialDeformation::new();
    for (i, &eps) in strain.iter().enumerate() {
        kin.precompute_from(eps);
        let pk2_stress = tissue.pk2(&kin);
        stress[i] = solve_uniaxial_pk2(&pk2_stress, &kin);
    }
    return stress;
}

pub fn simulate_hyperelastic_biaxial_response<T: ComputeHyperelasticBiaxialPK2>(
    tissue: &T,
    strain: &ArrayView3<f64>,
) -> Array3<f64> {
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin: BiaxialDeformation = BiaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps);
        let pk2_stress = tissue.pk2(&kin);
        stress
            .index_axis_mut(Axis(0), i)
            .assign(&solve_biaxial_pk2(&pk2_stress, &kin));
    }
    return stress;
}

pub fn simulate_hyperelastic_triaxial_response<T: ComputeHyperelasticTriaxialPK2>(
    tissue: &T,
    strain: &ArrayView3<f64>,
) -> Array3<f64> {
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin = TriaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps);
        let pk2_stress = tissue.pk2(&kin);
        stress.index_axis_mut(Axis(0), i).assign(&pk2_stress.stress);
    }
    return stress;
}

pub fn simulate_viscoelastic_uniaxial_response<T: ComputeViscoelasticUniaxialPK2>(
    tissue: &mut T,
    strain: &ArrayView1<f64>,
    dt: &ArrayView1<f64>,
) -> Array1<f64> {
    let mut stress = Array1::<f64>::zeros(strain.raw_dim());
    let mut kin = UniaxialDeformation::new();
    for (i, &eps) in strain.iter().skip(1).enumerate() {
        kin.precompute_from(eps);
        let pk2_stress = tissue.pk2(&kin, *dt.get(i).unwrap());
        stress[i] = solve_uniaxial_pk2(&pk2_stress, &kin);
    }
    return stress;
}

pub fn simulate_viscoelastic_biaxial_response<T: ComputeViscoelasticBiaxialPK2>(
    tissue: &mut T,
    strain: &ArrayView3<f64>,
    dt: &ArrayView1<f64>,
) -> Array3<f64> {
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin = BiaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps);
        let pk2_stress = tissue.pk2(&kin, *dt.get(i).unwrap());
        stress
            .index_axis_mut(Axis(0), i)
            .assign(&solve_biaxial_pk2(&pk2_stress, &kin));
    }
    return stress;
}

pub fn simulate_viscoelastic_triaxial_response<T: ComputeViscoelasticTriaxialPK2>(
    tissue: &mut T,
    strain: &ArrayView3<f64>,
    dt: &ArrayView1<f64>,
) -> Array3<f64> {
    let mut stress = Array3::<f64>::zeros(strain.raw_dim());
    let mut kin = TriaxialDeformation::new();
    for (i, eps) in strain.axis_iter(Axis(0)).enumerate() {
        kin.precompute_from(&eps);
        let pk2_stress = tissue.pk2(&kin, *dt.get(i).unwrap());
        stress.index_axis_mut(Axis(0), i).assign(&pk2_stress.stress);
    }
    return stress;
}
