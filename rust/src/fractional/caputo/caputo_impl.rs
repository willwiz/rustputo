use super::caputo_struct::CaputoData;
use super::precomputed_data::caputo_500::{CAPUTO15, CAPUTO9};

pub trait CaputoValue {}
impl CaputoValue for CaputoData<9> {}
impl CaputoValue for CaputoData<15> {}

pub const IMPLEMENTED_CAPUTO_VALUE: [i32; 2] = [9, 15];

impl CaputoData<9>
where
    Self: CaputoValue,
{
    pub fn new() -> &'static Self {
        &CAPUTO9
    }
}

impl CaputoData<15>
where
    Self: CaputoValue,
{
    pub fn new() -> &'static Self
    where
        Self: CaputoValue,
    {
        &CAPUTO15
    }
}
