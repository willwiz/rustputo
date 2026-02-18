use ndarray::{Array1, ArrayView1, ArrayView2, ArrayView3, Axis};

pub(super) fn solve_hydrostatics_3d(
    stress: &ArrayView2<f64>,
    c_inv: &ArrayView2<f64>,
    surf: &ArrayView1<f64>,
) -> f64 {
    let nc = c_inv.dot(surf);
    let ns = stress.dot(surf);
    (nc.dot(&ns)) / (nc.dot(&nc))
}

pub(super) fn solve_hydrostatics_3d_array(
    stress: &ArrayView3<f64>,
    c_inv: &ArrayView3<f64>,
    surf: &ArrayView1<f64>,
) -> Array1<f64> {
    Array1::from_iter(
        stress
            .axis_iter(Axis(0))
            .zip(c_inv.axis_iter(Axis(0)))
            .map(|(x, y)| solve_hydrostatics_3d(&x, &y, surf)),
    )
}
