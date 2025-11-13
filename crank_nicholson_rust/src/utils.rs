use ndarray::{Array1};

pub fn safe_exp(x: f64) -> f64 {
    f64::exp(x.clamp(-100.0, 100.0))
}

pub fn vtrap(x: f64, y: f64) -> f64 {
    let z = x / y;
    if z.abs() < 1e-6 {
        y * (1.0 - z / 2.0)
    } else {
        x / (safe_exp(z) - 1.0)
    }
}

pub fn i_inj_uacm2(t: f64, t_start_inj: f64, t_stop_inj: f64, nx: usize, inj_uacm2: f64, idx_inj: usize) -> Array1<f64> {
    let mut arr = Array1::<f64>::zeros(nx);
    if t_start_inj <= t && t <= t_stop_inj {
        arr[idx_inj] = inj_uacm2;
    }
    arr
}