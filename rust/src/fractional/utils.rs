pub fn interpolate_arr_1d(arr: &[f64; 100], x: f64) -> f64 {
    if x < 0.0 || x > 1.0 {
        panic!("x must be in the range (0, max_val), unable to proceed with interpolation.");
    }

    let percentile = x * 100.0;
    let index = percentile.floor() as usize;
    let t = percentile % 1.0;
    if index == 0 {
        return arr[0] * t;
    }
    arr[index - 1] * (1.0 - t) + arr[index] * t
}
