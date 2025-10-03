use ndarray::{Array1, ArrayView1, ArrayView2};
use ndarray_linalg::solve::Inverse;
pub fn lgres_mat(x: ArrayView2<'_, f64>, b: ArrayView1<'_, f64>) -> Array1<f64> {
    // This is a placeholder for the actual implementation of the lgres_mat function.
    // For now, it just returns the result of axpy with a = 1.0.
    let x_tx = x.reversed_axes().dot(&x);
    let x_tx_inv = x_tx.inv().unwrap();
    x_tx_inv.dot(&b)
}
