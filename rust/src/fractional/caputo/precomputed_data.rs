pub(super) mod caputo_500;

pub struct CaputoPrecomputedData<const NP: usize> {
    pub b0: [f64; 100],
    pub beta: [[f64; 100]; NP],
    pub tau: [[f64; 100]; NP],
}
