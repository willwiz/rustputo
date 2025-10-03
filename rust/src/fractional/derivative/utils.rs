pub fn interpolate_arr_1d(arr: &[f64; 100], x: f64) -> f64 {
    if x < 0.0 || x > 1.0 {
        panic!("x must be in the range (0, max_val), unable to proceed with interpolation.");
    }
    let percentile = (x / 100.0) as usize;
    let t = x % 100.0;
    if percentile == 0 {
        return arr[0] * t;
    }
    arr[0] * (1.0 - t) + arr[1] * t
}
