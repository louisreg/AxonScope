import numpy as np
import matplotlib.pyplot as plt
import settings as s
import solver_helper as sh
import utils as u
from axonscope.benchmark import minibench

def run(Nx,):

    dx = s.L / (Nx - 1)
    dx2 = (dx * 1e-4) ** 2
    dx_cm = dx * 1e-4
    x_axon = np.linspace(0, s.L, Nx)
    alpha = s.D * (s.dt / 2.0) / dx2  # diffusion coefficient
    idx_inj = np.argmin(np.abs(x_axon - s.position)) 



    inj_uA_per_cm2 = s.amplitude * 1e-3 / (2.0 * np.pi * s.a_cm * dx_cm)
    Nt = int(np.ceil(s.tsim / s.dt))

    V = np.ones(Nx) * s.Vinit
    V_all = np.zeros((Nt, Nx))
    t_vec = np.zeros(Nt)

    # Construct the tridiagonal matrix A (time-independent)
    A = np.zeros((Nx, Nx))
    for i in range(1, Nx-1):
        A[i, i-1] = -alpha
        A[i, i]   = 1.0 + 2.0*alpha
        A[i, i+1] = -alpha
    A[0, 0]   = 1.0
    A[-1, -1] = 1.0

    m_RA = np.zeros(Nx, dtype=float)
    h_RA = np.zeros(Nx, dtype=float)
    n_RA = np.zeros(Nx, dtype=float)

    V0 = np.ones(Nx) * float(s.Vinit)
    minf, mtau, hinf, htau, ninf, ntau = sh.rates(V0)

    m_RA[:] = minf
    h_RA[:] = hinf
    n_RA[:] = ninf

    for n in range(Nt):
        t_mid = n*s.dt + s.dt/2.0
        t_vec[n] = n*s.dt

        # -----------------------------
        # Hines equation:
        # (A) V^{1/2} = V^n + (dt/2Cm)[ I_inj(t+dt/2) - I_HH(V^{1/2}, t+dt/2) ]
        # -----------------------------

        # Right-hand side: V^n + (dt/2Cm) * (I_inj - I_HH)
        rhs = np.array(V, copy=True)
        Iinj = sh.Iinj_uAcm2(t_mid,s.t_start_inj,s.t_stop_inj,Nx, inj_uA_per_cm2, idx_inj)# [µA/cm²]
        
        rhs += (s.dt / (2.0 * s.Cm)) * (Iinj - sh.Iion(V, Nx, m_RA, n_RA, h_RA))

        m_RA, h_RA, n_RA = sh.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)

        # Boundary conditions
        rhs[0]  = s.Vinit
        rhs[-1] = s.Vinit


        V_half = np.linalg.solve(A, rhs)

        # Explicit update (Hines: extrapolation)
        V_new = 2.0 * V_half - V
        V_new[0]  = s.Vinit
        V_new[-1] = s.Vinit

        V_all[n, :] = V_new
        V = V_new
    return(t_vec, V_all)



t_v = []

for Nx in s.Nx_v:
    res, t = minibench(run,Nx,n_iter=s.n_iter)
    t_v.append(t)

df = u.res_to_df(s.Nx_v, t_v, label = "standalone")

u.append_to_csv(df)

t_vec = res[0]
V_all = res[1]

x_axon = np.linspace(0, s.L, Nx)
# --------------------------
# PLOT
# --------------------------
x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [np.argmin(np.abs(x_axon - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(t_vec, V_all[:,idx], label=f'x={xp:.3f} cm')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
plt.show()
