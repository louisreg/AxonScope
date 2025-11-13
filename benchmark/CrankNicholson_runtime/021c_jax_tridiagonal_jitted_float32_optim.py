import jax
import jax.numpy as jnp
from jax import lax
import settings as s
import solver_helper_jax_f32 as sh
import matplotlib.pyplot as plt
import utils as u
import time

jax.config.update("jax_enable_x64", False)  # float32

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
        m_RA, h_RA, n_RA = sh.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)

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
df = u.res_to_df(s.Nx_v, t_v, label="jax_tridiagonal_jit_ultra_f32")
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
