use ndarray::{Array1, Zip};
use crate::utils::{vtrap,safe_exp};


pub fn alpha_m(V: f64) -> f64 { vtrap(2.5 - 0.1 * (V + 70.0), 1.0) }
pub fn beta_m(V: f64) -> f64 { 4.0 * safe_exp(-(V + 70.0) / 18.0) }
pub fn alpha_h(V: f64) -> f64 { 0.07 * safe_exp(-(V + 70.0) / 20.0) }
pub fn beta_h(V: f64) -> f64 { 1.0 / (safe_exp(3.0 - 0.1 * (V + 70.0)) + 1.0) }
pub fn alpha_n(V: f64) -> f64 { 0.1 * vtrap(1.0 - 0.1 * (V + 70.0), 1.0) }
pub fn beta_n(V: f64) -> f64 { 0.125 * safe_exp(-(V + 70.0) / 80.0) }

pub fn i_ion(V: f64, m: f64, h: f64, n: f64) -> f64 {
    let gnabar = 0.12;
    let gkbar = 0.036;
    let gl = 0.0003;
    let ena = 45.0;
    let ek = -82.0;
    let el = -59.4;

    let gna = gnabar * m.powi(3) * h;
    let gk = gkbar * n.powi(4);

    gna * (V - ena) * 1e3 + gk * (V - ek) * 1e3 + gl * (V - el) * 1e3
}

pub fn half_step_gate_q10(
    g_prev: f64,
    alpha_fn: fn(f64) -> f64,
    beta_fn: fn(f64) -> f64,
    V: f64,
    dt: f64,
) -> f64 {
    let celsius: f64 = 37.0;
    let q10: f64 = 2.24659524757f64.powf((celsius - 6.3) / 10.0);
    let alpha = q10 * alpha_fn(V);
    let beta = q10 * beta_fn(V);

    let denom = 1.0 / dt + 0.5 * (alpha + beta);
    let term1 = alpha / denom;
    let term2 = ((1.0 / dt - 0.5 * (alpha + beta)) / denom) * g_prev;
    term1 + term2
}

pub fn update_gate_halfstep(
    g_prev: &Array1<f64>,
    alpha_fn: fn(f64) -> f64,
    beta_fn: fn(f64) -> f64,
    v: &Array1<f64>,
    dt: f64,
) -> Array1<f64> {
    const CELSIUS: f64 = 37.0;
    const Q10_BASE: f64 = 2.24659524757;
    let q10 = Q10_BASE.powf((CELSIUS - 6.3) / 10.0);

    let mut g_new = Array1::<f64>::zeros(g_prev.len());
    Zip::from(&mut g_new)
        .and(g_prev)
        .and(v)
        .for_each(|g_out, &g_old, &v_i| {
            let alpha = q10 * alpha_fn(v_i);
            let beta = q10 * beta_fn(v_i);
            let denom = (1.0 / dt) + 0.5 * (alpha + beta);
            let term1 = alpha / denom;
            let term2 = ((1.0 / dt) - 0.5 * (alpha + beta)) / denom * g_old;
            *g_out = term1 + term2;
        });
    g_new
}

pub fn half_step_gates(
    dt: f64,
    v: &Array1<f64>,
    m: &mut Array1<f64>,
    h: &mut Array1<f64>,
    n: &mut Array1<f64>,
) {
    if dt <= 0.0 {
        return;
    }

    *m = update_gate_halfstep(m, alpha_m, beta_m, v, dt);
    *h = update_gate_halfstep(h, alpha_h, beta_h, v, dt);
    *n = update_gate_halfstep(n, alpha_n, beta_n, v, dt);
}
