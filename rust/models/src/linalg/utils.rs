use ndarray::{Array2, ArrayView1};

use crate::utils::errors::PyError;

pub fn outer_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Result<Array2<f64>, PyError> {
    let a_col = a.to_shape((a.len(), 1))?;
    let b_row = b.to_shape((1, b.len()))?;
    Ok(a_col.dot(&b_row))
}

pub fn outer_sym_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Result<Array2<f64>, PyError> {
    let op = outer_product(a, b)?;
    Ok(&op + &op.t())
}
