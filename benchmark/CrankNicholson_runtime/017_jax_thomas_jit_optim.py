import jax
import jax.numpy as jnp
from jax import lax
import settings as s
import solver_helper_jax as sh
import matplotlib.pyplot as plt
import utils as u
import time

jax.config.update("jax_enable_x64", True)

# --------------------------
# Thomas algorithm (tridiagonal solver)
# --------------------------
def thomas_solve(a, b, c, d):
    n = b.shape[0]

    def fwd_step(carry, i):
        c_prev, d_prev = carry
        denom = jnp.where(i > 0, b[i] - a[i]*c_prev[i-1], b[0])
        c_new = c_prev.at[i].set(jnp.where(i < n-1, c[i]/denom, 0.0))
        d_new = d_prev.at[i].set(jnp.where(i > 0, (d[i] - a[i]*d_prev[i-1])/denom, d[0]/b[0]))
        return (c_new, d_new), None

    c0 = jnp.zeros_like(c)
    d0 = jnp.zeros_like(d)
    (c_, d_), _ = lax.scan(fwd_step, (c0, d0), jnp.arange(n))

    # Backward pass
    def scan_bwd(carry, i):
        x_i = d_[i] - c_[i]*carry
        return x_i, x_i

    _, x_all = lax.scan(scan_bwd, d_[-1], jnp.arange(n-2, -1, -1))
    x = jnp.zeros_like(d).at[-1].set(d_[-1]).at[:-1].set(x_all[::-1])
    return x

# --------------------------
# JAX Crank-Nicholson simulation using Thomas
# --------------------------
def run_jax_thomas(Nx, Nt, alpha, inj_uA_per_cm2):
    x_axon = jnp.linspace(0.0, s.L, Nx)
    idx_inj = jnp.argmin(jnp.abs(x_axon - s.position))

    # Tridiagonal vectors computed once
    a = -alpha * jnp.ones(Nx, dtype=jnp.float64)
    b = (1 + 2*alpha) * jnp.ones(Nx, dtype=jnp.float64)
    c = -alpha * jnp.ones(Nx, dtype=jnp.float64)
    b = b.at[0].set(1.0).at[-1].set(1.0)
    a = a.at[0].set(0.0)
    c = c.at[-1].set(0.0)

    V = jnp.full((Nx,), s.Vinit, dtype=jnp.float64)
    V_all = jnp.zeros((Nt, Nx), dtype=jnp.float64)

    # Initialize gating variables
    minf, mtau, hinf, htau, ninf, ntau = sh.rates(jnp.full((Nx,), s.Vinit))
    m_RA, h_RA, n_RA = minf, hinf, ninf

    def step(carry, n):
        V, m_RA, h_RA, n_RA, V_all = carry
        t_mid = n * s.dt

        Iinj = sh.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj)
        Iion_curr = sh.Iion(V, Nx, m_RA, n_RA, h_RA)

        rhs = V + (s.dt / (2.0*s.Cm))*(Iinj - Iion_curr)

        # Update gates
        m_RA, h_RA, n_RA = sh.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)

        # Boundary conditions
        rhs = rhs.at[0].set(s.Vinit)
        rhs = rhs.at[-1].set(s.Vinit)

        # Solve tridiagonal system
        V_half = thomas_solve(a, b, c, rhs)
        V_new = 2.0*V_half - V
        V_new = jnp.clip(V_new, -500.0, 500.0)
        V_new = V_new.at[0].set(s.Vinit)
        V_new = V_new.at[-1].set(s.Vinit)

        V_all = V_all.at[n,:].set(V_new)
        return (V_new, m_RA, h_RA, n_RA, V_all), None

    # Use lax.scan over Nt steps
    (V, m_RA, h_RA, n_RA, V_all), _ = lax.scan(step, (V, m_RA, h_RA, n_RA, V_all), jnp.arange(Nt))
    t_vec = jnp.arange(Nt)*s.dt
    return t_vec, V_all

# --------------------------
# Benchmark for all Nx values
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
                         run_jax_thomas(Nx, Nt, alpha, inj_uA_per_cm2))
    
    start = time.perf_counter()
    res = run_fn_jit()
    res[1].block_until_ready()  
    end = time.perf_counter()
    
    print(f"Nx={Nx}: Execution time (Thomas jitted, scan) = {end-start:.4f} s")
    t_v.append(end-start)
    res_list.append(res)

# Save benchmark results
df = u.res_to_df(s.Nx_v, t_v, label="jax_thomas_scan_jit")
u.append_to_csv(df)

# --------------------------
# Example plot for last Nx
# --------------------------
t_vec, V_all = res_list[-1]
x_axon = jnp.linspace(0.0, s.L, s.Nx_v[-1])
x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [jnp.argmin(jnp.abs(x_axon - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(t_vec, V_all[:, idx], label=f'x={xp:.3f} cm')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
plt.show()
