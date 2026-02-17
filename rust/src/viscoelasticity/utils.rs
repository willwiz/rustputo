use crate::utils::errors::PyError;

pub fn interpolate_arr_1d(arr: &[f64; 100], x: f64) -> Result<f64, PyError> {
    if x < 0.0 || x > 1.0 {
        return Err(PyError::Shape(
            "x must be in the range (0, max_val), unable to proceed with interpolation."
                .to_string(),
        ));
    }

    let percentile = x * 100.0;
    let index = percentile.floor() as usize;
    let t = percentile % 1.0;
    if index == 0 {
        return Ok(arr[0] * t);
    }
    Ok(arr[index - 1] * (1.0 - t) + arr[index] * t)
}
