use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

pub fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

// example using a mutable borrow to modify an array in-place
pub fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
    x *= a;
}
