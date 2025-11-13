use ndarray::{Array1, Array2};
use crate::params::SimParams;
use crate::gates::{alpha_m, beta_m,
                    alpha_h, beta_h,
                    alpha_n, beta_n,half_step_gates,i_ion};
use crate::utils::{i_inj_uacm2};
use ndarray_linalg::{Solve, SolveTridiagonal, Tridiagonal};
use rayon::prelude::*; // optional parallelism
use ndarray_linalg::layout::MatrixLayout;
use ndarray_linalg::{FactorizeTridiagonalInto};
use ndarray_linalg::SolveTridiagonalInplace;
use ndarray::Axis;
use ndarray::s;

pub fn hines_basic(nx: usize, params: &SimParams) -> (Array1<f64>, Array2<f64>) {
    let mut V = Array1::<f64>::from_elem(nx, params.vinit);
    let mut m = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_m(v) / (alpha_m(v) + beta_m(v))));
    let mut h = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_h(v) / (alpha_h(v) + beta_h(v))));
    let mut n = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_n(v) / (alpha_n(v) + beta_n(v))));

    let mut V_all = Array2::<f64>::zeros((params.nsteps, nx));

    // Matrice tridiagonale
    let mut A = Array2::<f64>::zeros((nx, nx));
    for i in 1..nx-1 {
        A[[i, i-1]] = -params.alpha;
        A[[i, i]] = 1.0 + 2.0 * params.alpha;
        A[[i, i+1]] = -params.alpha;
    }
    A[[0,0]] = 1.0;
    A[[nx-1,nx-1]] = 1.0;

    for t_idx in 0..params.nsteps {
        let t = t_idx as f64 * params.dt;
        let t_mid = t + params.dt / 2.0;

        let Iinj = i_inj_uacm2(t_mid, params.t_start, params.t_start + params.duration, nx, params.inj_amp, params.idx_inj);

        let mut rhs = V.clone();
        for i in 1..nx-1 {
            rhs[i] += params.dt / (2.0 * params.cm) * (Iinj[i] - i_ion(V[i], m[i], h[i], n[i]));
        }

        rhs[0] = params.vinit;
        rhs[nx-1] = params.vinit;

        half_step_gates(params.dt, &V, &mut m, &mut h, &mut n);

        let V_half = A.solve(&rhs).expect("Erreur solve V_half");
        let mut V_new = &V_half * 2.0 - &V;
        V_new[0] = params.vinit;
        V_new[nx-1] = params.vinit;

        V_all.row_mut(t_idx).assign(&V_new);
        V = V_new;
    }

    (Array1::linspace(0.0, params.tsim, params.nsteps), V_all)
}

pub fn hines_tridiag(nx: usize, params: &SimParams) -> (Array1<f64>, Array2<f64>) {
    let mut V = Array1::<f64>::from_elem(nx, params.vinit);
    let mut m = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_m(v) / (alpha_m(v) + beta_m(v))));
    let mut h = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_h(v) / (alpha_h(v) + beta_h(v))));
    let mut n = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_n(v) / (alpha_n(v) + beta_n(v))));

    let mut V_all = Array2::<f64>::zeros((params.nsteps, nx));

    // --- Build tridiagonal coefficients ---
    let mut dl = vec![-params.alpha; nx - 1];
    let mut d  = vec![1.0 + 2.0 * params.alpha; nx];
    let mut du = vec![-params.alpha; nx - 1];

    // Boundary conditions
    d[0] = 1.0;
    d[nx - 1] = 1.0;
    du[0] = 0.0;
    dl[nx - 2] = 0.0;

    // The MatrixLayout field is required but not actually used.
    let layout = MatrixLayout::C { row: nx as i32, lda: nx as i32 };

    let A = Tridiagonal {
        l: layout,
        dl,
        d,
        du,
    };

    // --- Time integration loop ---
    for t_idx in 0..params.nsteps {
        let t = t_idx as f64 * params.dt;
        let t_mid = t + params.dt / 2.0;

        let Iinj = i_inj_uacm2(
            t_mid,
            params.t_start,
            params.t_start + params.duration,
            nx,
            params.inj_amp,
            params.idx_inj,
        );

        let mut rhs = V.clone();
        for i in 1..nx - 1 {
            rhs[i] += params.dt / (2.0 * params.cm)
                * (Iinj[i] - i_ion(V[i], m[i], h[i], n[i]));
        }

        rhs[0] = params.vinit;
        rhs[nx - 1] = params.vinit;

        half_step_gates(params.dt, &V, &mut m, &mut h, &mut n);

        // Solve tridiagonal system
        let V_half = A
            .solve_tridiagonal(&rhs)
            .expect("Failed to solve tridiagonal system");

        let mut V_new = &V_half * 2.0 - &V;
        V_new[0] = params.vinit;
        V_new[nx - 1] = params.vinit;

        V_all.row_mut(t_idx).assign(&V_new);
        V = V_new;
    }

    (Array1::linspace(0.0, params.tsim, params.nsteps), V_all)
}




pub fn hines_tridiag_fast(nx: usize, params: &SimParams) -> (Array1<f64>, Array2<f64>) {
    // --- Initialization ---
    let mut V = Array1::<f64>::from_elem(nx, params.vinit);
    let mut m = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_m(v) / (alpha_m(v) + beta_m(v))));
    let mut h = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_h(v) / (alpha_h(v) + beta_h(v))));
    let mut n = Array1::<f64>::from_iter(V.iter().map(|&v| alpha_n(v) / (alpha_n(v) + beta_n(v))));

    let mut V_all = Array2::<f64>::zeros((params.nsteps, nx));

    // --- Tridiagonal coefficients (constant) ---
    let mut dl = vec![-params.alpha; nx - 1];
    let mut d  = vec![1.0 + 2.0 * params.alpha; nx];
    let mut du = vec![-params.alpha; nx - 1];
    d[0] = 1.0;
    d[nx - 1] = 1.0;
    du[0] = 0.0;
    dl[nx - 2] = 0.0;

    let layout = MatrixLayout::C { row: nx as i32, lda: nx as i32 };
    let A = ndarray_linalg::Tridiagonal { l: layout, dl, d, du };
    let lu = A.factorize_tridiagonal_into().expect("LU factorization failed");

    // --- Preallocate buffers ---
    let mut rhs = Array1::<f64>::zeros(nx);
    let mut rhs2d = rhs.clone().insert_axis(Axis(1));
    let mut Iinj = Array1::<f64>::zeros(nx);

    // --- Main loop ---
    for t_idx in 0..params.nsteps {
        let t = t_idx as f64 * params.dt;
        let t_mid = t + params.dt * 0.5;

        // Compute current injection
        Iinj.assign(&i_inj_uacm2(
            t_mid,
            params.t_start,
            params.t_start + params.duration,
            nx,
            params.inj_amp,
            params.idx_inj,
        ));

        // Build RHS (in-place)
        rhs.assign(&V);
        for i in 1..nx - 1 {
            rhs[i] += params.dt / (2.0 * params.cm)
                * (Iinj[i] - i_ion(V[i], m[i], h[i], n[i]));
        }
        rhs[0] = params.vinit;
        rhs[nx - 1] = params.vinit;

        half_step_gates(params.dt, &V, &mut m, &mut h, &mut n);

        // Solve in-place
        //lu.solve_tridiagonal_inplace(&mut rhs).unwrap();

        rhs2d.column_mut(0).assign(&rhs);
        lu.solve_tridiagonal_inplace(&mut rhs2d).unwrap();
        rhs.assign(&rhs2d.column(0));

        // Flatten back to 1D
        rhs.assign(&rhs2d.column(0));

        // Compute new potential (reusing rhs buffer)
        for i in 0..nx {
            rhs[i] = 2.0 * rhs[i] - V[i];
        }
        rhs[0] = params.vinit;
        rhs[nx - 1] = params.vinit;

        V_all.row_mut(t_idx).assign(&rhs);
        V.assign(&rhs);
    }

    (Array1::linspace(0.0, params.tsim, params.nsteps), V_all)
}



// assume SimParams, alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n,
// i_ion, half_step_gates, i_inj_uacm2 are available in scope

#[inline(always)]
fn init_gate(v: f64, alpha_fn: fn(f64)->f64, beta_fn: fn(f64)->f64) -> f64 {
    let a = alpha_fn(v);
    let b = beta_fn(v);
    a / (a + b)
}

pub fn hines_tridiag_rayon(nx: usize, params: &SimParams) -> (Array1<f64>, Array2<f64>) {
    // --- init arrays (one-time allocations) ---
    let mut V = Array1::<f64>::from_elem(nx, params.vinit);

    // initialize gates with equilibrium values (fast scalar init)
    let mut m = Array1::<f64>::zeros(nx);
    let mut h = Array1::<f64>::zeros(nx);
    let mut n = Array1::<f64>::zeros(nx);
    {
        // fill via slices for speed
        let v_slice = V.as_slice().unwrap();
        let m_slice = m.as_slice_mut().unwrap();
        let h_slice = h.as_slice_mut().unwrap();
        let n_slice = n.as_slice_mut().unwrap();
        for i in 0..nx {
            m_slice[i] = init_gate(v_slice[i], alpha_m, beta_m);
            h_slice[i] = init_gate(v_slice[i], alpha_h, beta_h);
            n_slice[i] = init_gate(v_slice[i], alpha_n, beta_n);
        }
    }

    // store outputs (you can store only every nth frame if memory matters)
    let mut V_all = Array2::<f64>::zeros((params.nsteps, nx));

    // --- tridiagonal coefficients (constant) ---
    let mut dl = vec![-params.alpha; nx - 1];
    let mut d  = vec![1.0 + 2.0 * params.alpha; nx];
    let mut du = vec![-params.alpha; nx - 1];

    // boundary conditions
    d[0] = 1.0;
    d[nx - 1] = 1.0;
    du[0] = 0.0;
    dl[nx - 2] = 0.0;

    // Layout required by Tridiagonal struct
    let layout = MatrixLayout::C { row: nx as i32, lda: nx as i32 };
    let A = ndarray_linalg::Tridiagonal { l: layout, dl, d, du };

    // factorize once
    let lu = A.factorize_tridiagonal_into().expect("LU factorization failed");

    // --- preallocate buffers reused each timestep ---
    let mut rhs = Array1::<f64>::zeros(nx);       // 1D RHS
    let mut rhs2d = Array2::<f64>::zeros((nx, 1)); // (nx,1) for in-place solver
    let mut Iinj = Array1::<f64>::zeros(nx);

    // small aliases for speed (avoid repeated field lookups in loop)
    let dt = params.dt;
    let cm = params.cm;
    let vinit = params.vinit;
    let t_start = params.t_start;
    let t_stop = params.t_start + params.duration;
    let inj_amp_val = params.inj_amp;
    let idx_inj = params.idx_inj;
    let nsteps = params.nsteps;

    // main loop
    for t_idx in 0..nsteps {
        let t = t_idx as f64 * dt;
        let t_mid = t + 0.5 * dt;

        // compute injection (cheap)
        // copy into preallocated Iinj
        {
            let inj_slice = Iinj.as_slice_mut().unwrap();
            // zero once
            unsafe { std::ptr::write_bytes(inj_slice.as_mut_ptr(), 0, inj_slice.len()); }
            if t_mid >= t_start && t_mid <= t_stop {
                inj_slice[idx_inj] = inj_amp_val;
            }
        }

        {
            // --- Mutably borrow rhs for parallel update ---
            let rhs_slice = rhs.as_slice_mut().unwrap();
            let v_s = V.as_slice().unwrap();
            let m_s = m.as_slice().unwrap();
            let h_s = h.as_slice().unwrap();
            let n_s = n.as_slice().unwrap();
            let inj_s = Iinj.as_slice().unwrap();

            rhs_slice[0] = params.vinit;
            rhs_slice[nx - 1] = params.vinit;

            if nx >= 200 {
                rhs_slice[1..nx - 1]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(offset, val)| {
                        let i = offset + 1;
                        let iion = i_ion(v_s[i], m_s[i], h_s[i], n_s[i]);
                        *val = v_s[i] + params.dt / (2.0 * params.cm) * (inj_s[i] - iion);
                    });
            } else {
                for i in 1..nx - 1 {
                    let iion = i_ion(v_s[i], m_s[i], h_s[i], n_s[i]);
                    rhs_slice[i] =
                        v_s[i] + params.dt / (2.0 * params.cm) * (inj_s[i] - iion);
                }
            }
        } // <-- rhs_slice mutable borrow ends here, compiler happy

        // You can now safely mutate rhs again:
        rhs[0] = params.vinit;
        rhs[nx - 1] = params.vinit;

        // advance gates half-step (in place)
        half_step_gates(dt, &V, &mut m, &mut h, &mut n);

        // solve tridiagonal in-place: need rhs as (nx,1) Array2
        rhs2d.column_mut(0).assign(&rhs);
        lu.solve_tridiagonal_inplace(&mut rhs2d).expect("solve failed");
        // extract column back into rhs slice (no allocate)
        rhs.assign(&rhs2d.column(0));

        // V_new = 2*V_half - V
        {
            let v_slice = V.as_slice().unwrap();
            let rhs_slice = rhs.as_slice_mut().unwrap(); // now contains V_half
            for i in 0..nx {
                rhs_slice[i] = 2.0 * rhs_slice[i] - v_slice[i];
            }
            rhs_slice[0] = vinit;
            rhs_slice[nx-1] = vinit;
        }

        // store & swap: write row into V_all and copy rhs into V
        V_all.row_mut(t_idx).assign(&rhs.view());
        V.assign(&rhs);
    }

    (Array1::linspace(0.0, params.tsim, nsteps), V_all)
}

