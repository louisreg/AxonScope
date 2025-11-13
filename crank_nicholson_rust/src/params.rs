pub struct SimParams {
    // Main parameters
    pub l: f64,         // µm
    pub d: f64,         // µm
    pub cm: f64,        // µF/cm²
    pub ra: f64,        // ohm*cm
    pub dt: f64,        // ms
    pub tsim: f64,      // ms
    pub vinit: f64,     // mV
    pub t_start: f64,   // injection start time ms
    pub duration: f64,  // injection duration ms
    pub amplitude: f64, // nA

    // Derived parameters
    pub dx: f64,
    pub dx_cm: f64,
    pub a_cm: f64,
    pub cm_f: f64,
    pub ra_cm: f64,
    pub diff: f64,
    pub alpha: f64,
    pub inj_amp: f64,
    pub idx_inj: usize,
    pub nsteps: usize,
}

impl SimParams {
    pub fn new(nx: usize) -> Self {
        let l = 1000.0;
        let d = 0.5;
        let cm = 1.0;
        let ra = 100.0;
        let dt = 0.001;
        let tsim = 10.0;
        let vinit = -70.0;
        let t_start = 1.0;
        let duration = 1.0;
        let amplitude = 2.0;
        let idx_inj = nx / 2;

        let dx = l / (nx - 1) as f64;
        let dx_cm = dx * 1e-4;
        let a_cm = (d / 2.0) * 1e-4;

        let cm_f = 2.0 * std::f64::consts::PI * a_cm * cm * 1e-6; // [F/cm]
        let ra_cm = ra / (std::f64::consts::PI * a_cm * a_cm);    // [ohm*cm]
        let diff = 1.0 / (ra_cm * cm_f) / 1000.0;                 // [cm²/ms]
        let alpha = diff * dt / (2.0 * dx_cm * dx_cm);
        let inj_amp = (amplitude * 1e-3) / (2.0 * std::f64::consts::PI * a_cm * dx_cm); // µA/cm²

        let nsteps = (tsim / dt) as usize;

        SimParams {
            l, d, cm, ra, dt, tsim, vinit, t_start, duration, amplitude,
            dx, dx_cm, a_cm, cm_f, ra_cm, diff, alpha, inj_amp, idx_inj, nsteps,
        }
    }
}
