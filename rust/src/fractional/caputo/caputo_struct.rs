pub struct CaputoData<const NP: usize>
where
    [(); NP]: Sized,
{
    pub n_p: usize,
    pub b0: [f64; 100],
    pub beta: [[f64; 100]; NP],
    pub tau: [[f64; 100]; NP],
}
