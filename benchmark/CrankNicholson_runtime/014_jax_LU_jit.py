import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import settings as s
import utils as u
from axonscope.benchmark import minibench
import solver_helper_jax as sh
import time
from jax.scipy.linalg import lu_factor, lu_solve

# --------------------------
# Enable 64-bit precision
# --------------------------
jax.config.update("jax_enable_x64", True)

# --------------------------
# Run JAX simulation for given Nx with LU factorization
# --------------------------
def run_jax_lu(Nx, Nt, alpha, inj_uA_per_cm2):
    x_axon = jnp.linspace(0.0, s.L, Nx)
    idx_inj = jnp.argmin(jnp.abs(x_axon - s.position))

    V = jnp.full((Nx,), s.Vinit, dtype=jnp.float64)
    V_all = jnp.zeros((Nt, Nx), dtype=jnp.float64)

    # Construct tridiagonal matrix
    A = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    A = A.at[0,0].set(1.0)
    A = A.at[-1,-1].set(1.0)
    for i in range(1, Nx-1):
        A = A.at[i,i-1].set(-alpha)
        A = A.at[i,i].set(1.0 + 2.0*alpha)
        A = A.at[i,i+1].set(-alpha)

    # LU factorization
    lu, piv = lu_factor(A)

    # Initialize gating variables
    minf, mtau, hinf, htau, ninf, ntau = sh.rates(jnp.full((Nx,), s.Vinit))
    m_RA, h_RA, n_RA = minf, hinf, ninf

    def step(n, carry):
        V, m_RA, h_RA, n_RA, V_all = carry
        t_mid = n * s.dt

        Iinj = sh.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj)
        Iion_curr = sh.Iion(V, Nx, m_RA, n_RA, h_RA)

        rhs = V + (s.dt / (2.0 * s.Cm)) * (Iinj - Iion_curr)

        # Update gates
        m_RA, h_RA, n_RA = sh.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)

        # Boundary conditions
        rhs = rhs.at[0].set(s.Vinit)
        rhs = rhs.at[-1].set(s.Vinit)

        # Solve using precomputed LU
        V_half = lu_solve((lu, piv), rhs)
        V_new = 2.0 * V_half - V
        V_new = jnp.clip(V_new, -500.0, 500.0)
        V_new = V_new.at[0].set(s.Vinit)
        V_new = V_new.at[-1].set(s.Vinit)

        V_all = V_all.at[n,:].set(V_new)
        return V_new, m_RA, h_RA, n_RA, V_all

    V, m_RA, h_RA, n_RA, V_all = lax.fori_loop(0, Nt, step, (V, m_RA, h_RA, n_RA, V_all))
    t_vec = jnp.arange(Nt) * s.dt
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
                         run_jax_lu(Nx, Nt, alpha, inj_uA_per_cm2))
    
    start = time.time()
    res = run_fn_jit()
    res[1].block_until_ready()
    end = time.time()
    print(f"Nx={Nx}: Execution time (JIT+LU) = {end-start:.4f} s")
    t_v.append(end-start)
    res_list.append(res)

# Save benchmark results
df = u.res_to_df(s.Nx_v, t_v, label="jax_lu_jit")
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
