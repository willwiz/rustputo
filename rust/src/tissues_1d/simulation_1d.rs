use ndarray::{Array1, ArrayView1};

use crate::{
    biomechanics::hyperelasticity::{ComputeUniaxialPK2, UniaxialPK2Stress},
    kinematics::deformation::UniaxialDeformation,
};

pub fn solve_uniaxial_pk2(model: &UniaxialPK2Stress, strain: &UniaxialDeformation) -> f64 {
    return model.stress - model.pressure * strain.i_n * strain.c_inv;
}

pub fn simulate_tissue_response<T: ComputeUniaxialPK2>(
    tissue: T,
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
