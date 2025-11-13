import numpy as np
import matplotlib.pyplot as plt
import settings as s
import solver_helper as sh
import utils as u
from axonscope.benchmark import minibench

def thomas_solver_vectorized(a, b, c, d):
    """
    Vectorized Thomas algorithm for a single RHS vector d.
    a = lower diagonal (length N-1)
    b = main diagonal (length N)
    c = upper diagonal (length N-1)
    d = RHS vector (length N)
    Returns solution vector x.
    """
    N = len(d)
    # Copy to avoid modifying inputs
    ac, bc, cc, dc = a.copy(), b.copy(), c.copy(), d.copy()
    
    # Forward sweep
    cc[0] /= bc[0]
    dc[0] /= bc[0]
    for i in range(1, N-1):
        tmp = bc[i] - ac[i-1]*cc[i-1]
        cc[i] /= tmp
        dc[i] = (dc[i] - ac[i-1]*dc[i-1]) / tmp
    dc[-1] = (dc[-1] - ac[-1]*dc[-2]) / (bc[-1] - ac[-1]*cc[-2])
    
    # Backward substitution
    x = np.zeros(N)
    x[-1] = dc[-1]
    for i in reversed(range(N-1)):
        x[i] = dc[i] - cc[i]*x[i+1]
    return x

def run(Nx):
    dx = s.L / (Nx - 1)
    dx_cm = dx * 1e-4
    dx2 = dx_cm ** 2
    x_axon = np.linspace(0, s.L, Nx)
    alpha = s.D * (s.dt / 2.0) / dx2
    idx_inj = np.argmin(np.abs(x_axon - s.position))

    inj_uA_per_cm2 = s.amplitude * 1e-3 / (2.0 * np.pi * s.a_cm * dx_cm)
    Nt = int(np.ceil(s.tsim / s.dt))

    V = np.ones(Nx) * s.Vinit
    V_all = np.zeros((Nt, Nx))
    t_vec = np.arange(Nt) * s.dt

    # --------------------------
    # Tridiagonal coefficients
    a = -alpha * np.ones(Nx-1)
    b = (1.0 + 2.0*alpha) * np.ones(Nx)
    c = -alpha * np.ones(Nx-1)
    b[0] = b[-1] = 1.0
    c[0] = 0.0
    a[-1] = 0.0

    # --------------------------
    # Gating variables (vectorized)
    m_RA, h_RA, n_RA = sh.rates(np.ones(Nx) * s.Vinit)[:3]
    
    # --------------------------
    # Time loop (inherently sequential)
    for n in range(Nt):
        t_mid = t_vec[n]

        Iinj = sh.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj)

        rhs = V + (s.dt / (2.0 * s.Cm)) * (Iinj - sh.Iion(V, Nx, m_RA, n_RA, h_RA))

        # Update gating variables
        m_RA, h_RA, n_RA = sh.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)

        # Dirichlet BCs
        rhs[0] = rhs[-1] = s.Vinit

        # Solve tridiagonal system with vectorized Thomas
        V_half = thomas_solver_vectorized(a, b, c, rhs)

        # Hines extrapolation (explicit)
        V = np.clip(2*V_half - V, -500, 500)
        V[0] = V[-1] = s.Vinit

        V_all[n, :] = V

    return t_vec, V_all

# --------------------------
# Benchmark
# --------------------------
#s.Nx_v = [21,51]
t_v = []
for Nx in s.Nx_v:
    res, t = minibench(run, Nx, n_iter=s.n_iter)
    t_v.append(t)

df = u.res_to_df(s.Nx_v, t_v, label="numpy_thomas_vectorized")
u.append_to_csv(df)

t_vec, V_all = res

x_axon = np.linspace(0, s.L, Nx)
x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [np.argmin(np.abs(x_axon - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(t_vec, V_all[:, idx], label=f'x={xp:.3f} cm')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
plt.show()
