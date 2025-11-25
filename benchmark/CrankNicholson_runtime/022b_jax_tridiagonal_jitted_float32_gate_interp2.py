import jax
import jax.numpy as jnp
from jax import lax
import settings as s
import solver_helper_jax_f32 as sh
import matplotlib.pyplot as plt
import utils as u
import time

jax.config.update("jax_enable_x64", False)  # float32

dtype = jnp.float32
# --------------------------
# Precomputed lookup for gate kinetics
# --------------------------
# ---------- Make lookup tables for minf and tau ----------
# Vrange chosen to cover typical HH voltages
Vmin, Vmax, Ntab = -120.0, 80.0, 1024

def make_table_minf_tau(alpha_fn, beta_fn, Vmin=Vmin, Vmax=Vmax, Ntab=Ntab):
    Vtab = jnp.linspace(Vmin, Vmax, Ntab, dtype=dtype)
    am = alpha_fn(Vtab)
    bm = beta_fn(Vtab)
    ah = alpha_fn(Vtab)   # placeholder; call specific funcs below
    # But we'll call specific functions for each gate below
    return Vtab

# Build minf/tau tables for each gate (use your sh.alpha_*/beta_* functions)
Vtab = jnp.linspace(Vmin, Vmax, Ntab, dtype=dtype)
# m
am = sh.alpha_m(Vtab); bm = sh.beta_m(Vtab)
sum_m = jnp.maximum(am + bm, dtype(1e-12))
minf_tab = am / sum_m
tau_m_tab = 1.0 / ( (2.24659524757 ** ((37.0 - 6.3)/10.0)) * sum_m )  # q10 factor included
# h
ah = sh.alpha_h(Vtab); bh = sh.beta_h(Vtab)
sum_h = jnp.maximum(ah + bh, dtype(1e-12))
hinf_tab = ah / sum_h
tau_h_tab = 1.0 / ( (2.24659524757 ** ((37.0 - 6.3)/10.0)) * sum_h )
# n
an = sh.alpha_n(Vtab); bn = sh.beta_n(Vtab)
sum_n = jnp.maximum(an + bn, dtype(1e-12))
ninf_tab = an / sum_n
tau_n_tab = 1.0 / ( (2.24659524757 ** ((37.0 - 6.3)/10.0)) * sum_n )

dV_tab = (Vmax - Vmin) / (Ntab - 1)

def interp1d_fast(V, Vtab, ftab, dV):
    """Linear interpolation between table points (JAX-friendly)."""
    idx_f = (V - Vtab[0]) / dV
    idx0 = jnp.clip(jnp.floor(idx_f).astype(jnp.int32), 0, ftab.size - 2)
    idx1 = idx0 + 1
    frac = idx_f - idx0
    f0 = ftab[idx0]
    f1 = ftab[idx1]
    return f0 + frac * (f1 - f0)

# fast interp reuse interp1d_fast defined earlier in your code
def interp_tab(V, tab):
    return interp1d_fast(V, Vtab, tab, dV_tab)

# exact update using minf/tau
def update_gate_exact(g_prev, minf_fun, tau_tab_fun, V, dt):
    # minf_fun: function V -> minf (via interp)
    # tau_tab_fun: function V -> tau (via interp)
    minf = minf_fun(V)
    tau = tau_tab_fun(V)
    # avoid tiny tau
    tau = jnp.maximum(tau, dtype(1e-12))
    # exact integration factor
    expm = jnp.exp(-dt / tau)
    return minf - (minf - g_prev) * expm

def half_step_gates_minf_tau(dt_ms, V_mV, Nx, m, h, n):
    m_new = update_gate_exact(m, lambda V: interp_tab(V, minf_tab), lambda V: interp_tab(V, tau_m_tab), V_mV, dt_ms)
    h_new = update_gate_exact(h, lambda V: interp_tab(V, hinf_tab), lambda V: interp_tab(V, tau_h_tab), V_mV, dt_ms)
    n_new = update_gate_exact(n, lambda V: interp_tab(V, ninf_tab), lambda V: interp_tab(V, tau_n_tab), V_mV, dt_ms)
    return m_new, h_new, n_new

# --------------------------
# Crank-Nicholson using tridiagonal_solve (optimized)
# --------------------------
def run_jax_tridiagonal_scan_optimized(Nx, Nt, alpha, inj_uA_per_cm2):
    x_axon = jnp.linspace(0.0, s.L, Nx, dtype=jnp.float32)
    idx_inj = jnp.argmin(jnp.abs(x_axon - s.position))

    # Tridiagonal vectors
    dl = -alpha * jnp.ones(Nx, dtype=jnp.float32).at[0].set(0.0)
    d  = (1 + 2*alpha) * jnp.ones(Nx, dtype=jnp.float32)
    du = -alpha * jnp.ones(Nx, dtype=jnp.float32).at[-1].set(0.0)

    # Voltage and gating variables
    V = jnp.full((Nx,), s.Vinit, dtype=jnp.float32)
    minf, mtau, hinf, htau, ninf, ntau = sh.rates(V)
    m_RA, h_RA, n_RA = minf, hinf, ninf

    # Preallocate output
    V_all = jnp.zeros((Nt, Nx), dtype=jnp.float32)

    # --------------------------
    def step(carry, n):
        V, m_RA, h_RA, n_RA = carry
        t_mid = n * s.dt

        # Currents
        Iinj = sh.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj)
        Iion_curr = sh.Iion(V, Nx, m_RA, n_RA, h_RA)

        rhs = V + (s.dt / (2.0*s.Cm)) * (Iinj - Iion_curr)

        # Update gating variables
        m_RA, h_RA, n_RA = half_step_gates_minf_tau(s.dt, V, Nx, m_RA, h_RA, n_RA)

        # Apply boundary conditions
        rhs = rhs.at[0].set(s.Vinit).at[-1].set(s.Vinit)

        # Solve tridiagonal system
        V_half = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]

        # Crank-Nicholson update
        V_new = 2.0 * V_half - V
        V_new = jnp.clip(V_new, -500.0, 500.0)
        V_new = V_new.at[0].set(s.Vinit).at[-1].set(s.Vinit)

        return (V_new, m_RA, h_RA, n_RA), V_new  # carry without V_all, output is V_new

    # --------------------------
    # Scan over time
    (V_final, m_final, h_final, n_final), V_all = lax.scan(step, (V, m_RA, h_RA, n_RA), jnp.arange(Nt))
    t_vec = jnp.arange(Nt, dtype=jnp.float32) * s.dt
    return t_vec, V_all  # V_all already stacked

# --------------------------
# Benchmark
# --------------------------
t_v = []
res_list = []

for Nx in s.Nx_v:
    dx = s.L / (Nx - 1)
    dx_cm = dx * 1e-4
    dx2 = dx_cm**2
    alpha = s.D * (s.dt / 2.0) / dx2
    inj_uA_per_cm2 = s.amplitude * 1e-3 / (2.0 * jnp.pi * s.a_cm * dx_cm)
    Nt = int(jnp.ceil(s.tsim / s.dt))

    run_fn_jit = jax.jit(lambda Nx=Nx, Nt=Nt, alpha=alpha, inj_uA_per_cm2=inj_uA_per_cm2:
                         run_jax_tridiagonal_scan_optimized(Nx, Nt, alpha, inj_uA_per_cm2))

    # Compile first
    res_compiled = run_fn_jit()
    res_compiled[1].block_until_ready()
    start_time = time.perf_counter()
    res = run_fn_jit()
    res[1].block_until_ready()
    end_time = time.perf_counter()

    print(f"Nx={Nx}: Execution time (tridiagonal float32 ultra-optimized) = {end_time - start_time:.4f} s")
    t_v.append(end_time - start_time)
    res_list.append(res)

# --------------------------
# Save benchmark results
# --------------------------
df = u.res_to_df(s.Nx_v, t_v, label="jax_tridiagonal_jit_ultra_f32_lookup2")
u.append_to_csv(df)

# --------------------------
# Plot example
# --------------------------
t_vec, V_all = res_list[-1]
x_axon = jnp.linspace(0.0, s.L, s.Nx_v[-1], dtype=jnp.float32)
x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [jnp.argmin(jnp.abs(x_axon - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(t_vec, V_all[:, idx], label=f'x={xp:.3f} cm')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
plt.show()
