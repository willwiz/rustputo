use crate::biomechanics::model_traits::{
    BiaxialPK2Stress, ComputeHyperelasticBiaxialPK2, ComputeHyperelasticTriaxialPK2,
    ComputeHyperelasticUniaxialPK2, TriaxialPK2Stress, UniaxialPK2Stress,
};
use crate::kinematics::deformation::{
    BiaxialDeformation, TriaxialDeformation, UniaxialDeformation,
};
use crate::utils::errors::PyError;
use ndarray::Array2;
pub struct DoubleE {
    pub b_iso: f64,
    pub b_shear: f64,
    pub k_ff: f64,
    pub k_ss: f64,
    pub k_nn: f64,
    pub k_fs: f64,
    pub k_fn: f64,
    pub k_sn: f64,
    pub mxm: ndarray::Array2<f64>,
    pub sxs: ndarray::Array2<f64>,
    pub nxn: ndarray::Array2<f64>,
    pub mxs: ndarray::Array2<f64>,
    pub mxn: ndarray::Array2<f64>,
    pub sxn: ndarray::Array2<f64>,
}

impl DoubleE {
    pub fn new(
        b_iso: f64,
        b_shear: f64,
        k_ff: f64,
        k_ss: f64,
        k_nn: f64,
        k_fs: f64,
        k_fn: f64,
        k_sn: f64,
        h: Array2<f64>,
    ) -> Self {
        let sym = |v: Array2<f64>| &v + &v.t();
        let mxm = h
            .row(0)
            .to_shape((h.ncols(), 1))
            .unwrap()
            .dot(&h.row(0).to_shape((1, h.ncols())).unwrap());
        let sxs = h
            .row(1)
            .to_shape((h.ncols(), 1))
            .unwrap()
            .dot(&h.row(1).to_shape((1, h.ncols())).unwrap());
        let nxn = h
            .row(2)
            .to_shape((h.ncols(), 1))
            .unwrap()
            .dot(&h.row(2).to_shape((1, h.ncols())).unwrap());
        let mxs = sym(h
            .row(0)
            .to_shape((h.ncols(), 1))
            .unwrap()
            .dot(&h.row(1).to_shape((1, h.ncols())).unwrap()));
        let mxn = sym(h
            .row(0)
            .to_shape((h.ncols(), 1))
            .unwrap()
            .dot(&h.row(2).to_shape((1, h.ncols())).unwrap()));
        let sxn = sym(h
            .row(1)
            .to_shape((h.ncols(), 1))
            .unwrap()
            .dot(&h.row(2).to_shape((1, h.ncols())).unwrap()));
        Self {
            b_iso,
            b_shear,
            k_ff,
            k_ss,
            k_nn,
            k_fs,
            k_fn,
            k_sn,
            mxm,
            sxs,
            nxn,
            mxs,
            mxn,
            sxn,
        }
    }
}

impl ComputeHyperelasticUniaxialPK2 for DoubleE {
    fn pk2(&self, strain: &UniaxialDeformation) -> UniaxialPK2Stress {
        let l_s = 1.0 / strain.c.sqrt();
        let i_1 = strain.c + 2.0 * l_s;
        let w_1 = (self.b_iso * (i_1 - 3.0)).exp();
        UniaxialPK2Stress {
            stress: self.k_ff * (w_1 * strain.c - 1.0),
            pressure: 0.5 * (self.k_ss + self.k_nn) * (w_1 * l_s - 1.0),
        }
    }
}

impl ComputeHyperelasticBiaxialPK2 for DoubleE {
    fn pk2(&self, strain: &BiaxialDeformation) -> BiaxialPK2Stress {
        let i_f = (&strain.c * &self.mxm).sum();
        let i_s = (&strain.c * &self.sxs).sum();
        let i_n = 1.0 / strain.det;
        let i_fs = (&strain.c * &self.mxs).sum();
        let w_1 = (self.b_iso * (i_f + i_s + i_n - 3.0)).exp();
        let w_2 = (self.b_shear * i_fs).exp();
        let stress = self.k_ff * (w_1 * i_f - 1.0) * &self.mxm
            + self.k_ss * (w_1 * i_s - 1.0) * &self.sxs
            + self.k_fs * (w_2 * i_fs) * &self.mxs;
        BiaxialPK2Stress {
            stress: stress,
            pressure: self.k_nn * (w_1 * i_n - 1.0),
        }
    }
}

impl ComputeHyperelasticTriaxialPK2 for DoubleE {
    fn pk2(&self, strain: &TriaxialDeformation) -> Result<TriaxialPK2Stress, PyError> {
        let i_f = (&strain.c * &self.mxm).sum();
        let i_s = (&strain.c * &self.sxs).sum();
        let i_n = (&strain.c * &self.nxn).sum();
        let i_fs = (&strain.c * &self.mxs).sum();
        let i_fn = (&strain.c * &self.mxn).sum();
        let i_sn = (&strain.c * &self.sxn).sum();
        let w_1 = (self.b_iso * (i_f + i_s + i_n - 3.0)).exp();
        let w_2 = (self.b_shear * (i_fs + i_fn + i_sn)).exp();
        let stress = self.k_ff * (w_1 * i_f - 1.0) * &self.mxm
            + self.k_ss * (w_1 * i_s - 1.0) * &self.sxs
            + self.k_nn * (w_1 * i_n - 1.0) * &self.nxn
            + self.k_fs * (w_2 * i_fs) * &self.mxs
            + self.k_fn * (w_2 * i_fn) * &self.mxn
            + self.k_sn * (w_2 * i_sn) * &self.sxn;
        Ok(TriaxialPK2Stress { stress })
    }
}
