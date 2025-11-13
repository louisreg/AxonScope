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
# Crank-Nicholson using tridiagonal_solve
# --------------------------
def run_jax_tridiagonal_scan(Nx, Nt, alpha, inj_uA_per_cm2):
    x_axon = jnp.linspace(0.0, s.L, Nx, dtype=jnp.float32)
    idx_inj = jnp.argmin(jnp.abs(x_axon - s.position))

    # Tridiagonal vectors
    dl = -alpha * jnp.ones(Nx, dtype=jnp.float32)
    dl = dl.at[0].set(0.0)
    d  = (1 + 2*alpha) * jnp.ones(Nx, dtype=jnp.float32)
    du = -alpha * jnp.ones(Nx, dtype=jnp.float32)
    du = du.at[-1].set(0.0)

    # Voltage arrays
    V = jnp.full((Nx,), s.Vinit, dtype=jnp.float32)
    V_all = jnp.zeros((Nt, Nx), dtype=jnp.float32)

    # Initialize gating variables
    minf, mtau, hinf, htau, ninf, ntau = sh.rates(V.astype(jnp.float32))
    m_RA, h_RA, n_RA = minf, hinf, ninf

    # --------------------------
    def step(carry, n):
        V, m_RA, h_RA, n_RA, V_all = carry
        t_mid = n * s.dt

        Iinj = sh.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj).astype(jnp.float32)
        Iion_curr = sh.Iion(V, Nx, m_RA, n_RA, h_RA).astype(jnp.float32)

        rhs = V + (s.dt / (2.0*s.Cm)) * (Iinj - Iion_curr)

        # Update gating variables
        m_RA, h_RA, n_RA = sh.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)
        m_RA = m_RA.astype(jnp.float32)
        h_RA = h_RA.astype(jnp.float32)
        n_RA = n_RA.astype(jnp.float32)

        # Apply boundary conditions in RHS only
        rhs = rhs.at[0].set(s.Vinit)
        rhs = rhs.at[-1].set(s.Vinit)

        # Solve tridiagonal system (ensure RHS is 2D)
        rhs_2d = rhs[:, None]  # shape (Nx, 1)
        V_half = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs_2d)
        V_half = V_half[:, 0]  # flatten back to 1D

        # Crank-Nicholson update
        V_new = 2.0 * V_half - V
        V_new = jnp.clip(V_new, -500.0, 500.0)
        V_new = V_new.at[0].set(s.Vinit)
        V_new = V_new.at[-1].set(s.Vinit)

        # Store
        V_all = V_all.at[n, :].set(V_new)
        return (V_new, m_RA, h_RA, n_RA, V_all), None

    # --------------------------
    # Scan over time steps
    (V, m_RA, h_RA, n_RA, V_all), _ = lax.scan(step, (V, m_RA, h_RA, n_RA, V_all), jnp.arange(Nt))
    t_vec = jnp.arange(Nt, dtype=jnp.float32) * s.dt
    return t_vec, V_all

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
                         run_jax_tridiagonal_scan(Nx, Nt, alpha, inj_uA_per_cm2))

    # Compile first
    res_compiled = run_fn_jit()
    res_compiled[1].block_until_ready()
    start_time = time.perf_counter()
    res = run_fn_jit()
    res[1].block_until_ready()
    end_time = time.perf_counter()

    print(f"Nx={Nx}: Execution time (tridiagonal float32) = {end_time - start_time:.4f} s")
    t_v.append(end_time - start_time)
    res_list.append(res)

# Save benchmark results
df = u.res_to_df(s.Nx_v, t_v, label="jax_tridiagonal_jit_optim_f32")
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
