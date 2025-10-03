use super::super::caputo::caputo_struct::CaputoData;

pub struct CaputoInternalArray<const FDIM: usize, const NP: usize> {
    pub alpha: f64,
    pub delta: f64,
    pub dt: f64,
    pub(super) params: &'static CaputoData<NP>,
    pub(super) c0: f64,
    pub(super) k0: f64,
    pub(super) k1: f64,
    pub(super) e2: [f64; NP],
    pub(super) bek: [f64; NP],
    pub(super) fprev: [f64; FDIM],
    pub(super) qk: [[f64; FDIM]; NP],
}

impl<const FDIM: usize> CaputoInternalArray<FDIM, 9> {
    pub fn new(alpha: f64, delta: f64) -> Self {
        let c0 = 0.0;
        let k0 = 0.0;
        let k1 = 1.0;
        let e2 = [0.0; 9];
        let bek = [0.0; 9];
        let qk = [[0.0; FDIM]; 9];
        let fprev = [0.0; FDIM];

        Self {
            alpha,
            delta,
            dt: 0.0,
            params: CaputoData::<9>::new(),
            c0,
            k0,
            k1,
            e2,
            bek,
            fprev,
            qk,
        }
    }
}

impl<const FDIM: usize> CaputoInternalArray<FDIM, 15> {
    pub fn new(alpha: f64, delta: f64) -> Self {
        let c0 = 0.0;
        let k0 = 0.0;
        let k1 = 1.0;
        let e2 = [0.0; 15];
        let bek = [0.0; 15];
        let qk = [[0.0; FDIM]; 15];
        let fprev = [0.0; FDIM];

        Self {
            alpha,
            delta,
            dt: 0.0,
            params: CaputoData::<15>::new(),
            c0,
            k0,
            k1,
            e2,
            bek,
            fprev,
            qk,
        }
    }
}
