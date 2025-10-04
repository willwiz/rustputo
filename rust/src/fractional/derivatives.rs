pub trait LinearDerivative<const FDIM: usize> {
    fn init_with_dt_lin(&mut self, dt: f64) -> &mut Self;
    fn caputo_derivative_lin(&mut self, f: &[f64; FDIM], dt: f64) -> [f64; FDIM];
}
