import torch
import numpy as np
import matplotlib.pyplot as plt
import settings as s
import solver_helper as sh
import utils as u
from axonscope.benchmark import minibench

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_torch_LU(Nx):
    dx = s.L / (Nx - 1)
    dx_cm = dx * 1e-4
    dx2 = dx_cm**2
    x_axon = torch.linspace(0, s.L, Nx, device=device, dtype=torch.float64)
    alpha = s.D * (s.dt / 2.0) / dx2
    idx_inj = torch.argmin(torch.abs(x_axon - s.position))

    inj_uA_per_cm2 = s.amplitude * 1e-3 / (2.0 * np.pi * s.a_cm * dx_cm)
    Nt = int(np.ceil(s.tsim / s.dt))

    # Voltage vectors
    V = torch.full((Nx,), s.Vinit, device=device, dtype=torch.float64)
    V_all = torch.zeros((Nt, Nx), device=device, dtype=torch.float64)
    t_vec = torch.arange(Nt, device=device, dtype=torch.float64) * s.dt

    # Build tridiagonal matrix A
    A = torch.zeros((Nx, Nx), device=device, dtype=torch.float64)
    for i in range(1, Nx - 1):
        A[i, i - 1] = -alpha
        A[i, i] = 1.0 + 2.0 * alpha
        A[i, i + 1] = -alpha
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    # Pre-factorize A once using LU decomposition
    lu, pivots = torch.linalg.lu_factor(A)

    # Initialize gating variables
    minf, mtau, hinf, htau, ninf, ntau = sh.rates(np.ones(Nx) * s.Vinit)
    m_RA = torch.tensor(minf, device=device, dtype=torch.float64)
    h_RA = torch.tensor(hinf, device=device, dtype=torch.float64)
    n_RA = torch.tensor(ninf, device=device, dtype=torch.float64)

    # Time loop
    for n in range(Nt):
        t_mid = t_vec[n].item()

        # Injected current
        Iinj = torch.tensor(
            sh.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj.item()),
            device=device,
            dtype=torch.float64
        )

        # Compute ionic currents
        Iion_curr = torch.tensor(
            sh.Iion(V.cpu().numpy(), Nx, m_RA.cpu().numpy(), n_RA.cpu().numpy(), h_RA.cpu().numpy()),
            device=device,
            dtype=torch.float64
        )

        # Right-hand side
        rhs = V + (s.dt / (2.0 * s.Cm)) * (Iinj - Iion_curr)

        # Update gating variables
        m_RA_np, h_RA_np, n_RA_np = sh.half_step_gates(
            s.dt, V.cpu().numpy(), Nx, m_RA.cpu().numpy(), h_RA.cpu().numpy(), n_RA.cpu().numpy()
        )
        m_RA = torch.tensor(m_RA_np, device=device, dtype=torch.float64)
        h_RA = torch.tensor(h_RA_np, device=device, dtype=torch.float64)
        n_RA = torch.tensor(n_RA_np, device=device, dtype=torch.float64)

        # Boundary conditions
        rhs[0] = s.Vinit
        rhs[-1] = s.Vinit

        # Solve using LU decomposition
        V_half = torch.linalg.lu_solve(lu, pivots, rhs.unsqueeze(1)).squeeze(1)

        # Hines extrapolation
        V_new = 2.0 * V_half - V
        V_new = torch.clamp(V_new, -500.0, 500.0)
        V_new[0] = s.Vinit
        V_new[-1] = s.Vinit

        V_all[n, :] = V_new
        V = V_new

    return t_vec.cpu().numpy(), V_all.cpu().numpy()

# --------------------------
# Benchmark
# --------------------------
t_v = []
for Nx in s.Nx_v:
    res, t = minibench(run_torch_LU, Nx, n_iter=s.n_iter)
    t_v.append(t)

df = u.res_to_df(s.Nx_v, t_v, label="torch_LU")
u.append_to_csv(df)

t_vec, V_all = res

# Plot results
x_axon = np.linspace(0, s.L, Nx)
x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [np.argmin(np.abs(x_axon - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(t_vec, V_all[:, idx], label=f"x={xp:.3f} cm")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
plt.show()
