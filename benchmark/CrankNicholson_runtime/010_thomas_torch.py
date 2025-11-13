import torch
import numpy as np
import matplotlib.pyplot as plt
import settings as s
import solver_helper_torch as sh_torch  # <-- your torch version of solver_helper
import utils as u
from axonscope.benchmark import minibench

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

--> doesn't work, pb continue aux limites surement

# ===============================================================
# Thomas algorithm (tridiagonal solver, pure Torch)
# ===============================================================
@torch.compile  # JIT for speed
def thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system Ax = d where:
      a = sub-diagonal (len N-1)
      b = main diagonal (len N)
      c = super-diagonal (len N-1)
      d = RHS
    Pure Torch implementation (no Python loops visible to Torch).
    """
    n = b.shape[0]
    cp = torch.empty_like(c)
    dp = torch.empty_like(d)
    bp = b.clone()

    cp[0] = c[0] / bp[0]
    dp[0] = d[0] / bp[0]

    for i in range(1, n - 1):
        denom = bp[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (bp[-1] - a[-1] * cp[-2])

    x = torch.empty_like(d)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------
@torch.compile  # JIT for speed
def run_torch_thomas(Nx):
    dx = s.L / (Nx - 1)
    dx_cm = dx * 1e-4
    dx2 = dx_cm**2
    x_axon = torch.linspace(0, s.L, Nx, device=device)
    alpha = s.D * (s.dt / 2.0) / dx2
    idx_inj = torch.argmin(torch.abs(x_axon - s.position))

    inj_uA_per_cm2 = s.amplitude * 1e-3 / (2.0 * np.pi * s.a_cm * dx_cm)
    Nt = int(np.ceil(s.tsim / s.dt))

    # Voltage
    V = torch.full((Nx,), s.Vinit, device=device, dtype=torch.float64)
    V_all = torch.zeros((Nt, Nx), device=device, dtype=torch.float64)
    t_vec = torch.arange(Nt, device=device) * s.dt

    # Tridiagonal coefficients
    a = torch.full((Nx - 1,), -alpha, device=device, dtype=torch.float64)
    b = torch.full((Nx,), 1.0 + 2.0 * alpha, device=device, dtype=torch.float64)
    c = torch.full((Nx - 1,), -alpha, device=device, dtype=torch.float64)
    b[0] = b[-1] = 1.0

    # Gating variables
    minf, mtau, hinf, htau, ninf, ntau = sh_torch.rates(torch.full((Nx,), s.Vinit, device=device))
    m_RA, h_RA, n_RA = minf.clone(), hinf.clone(), ninf.clone()

    for n in range(Nt):
        t_mid = t_vec[n].item()

        # External current
        Iinj = sh_torch.Iinj_uAcm2(t_mid, s.t_start_inj, s.t_stop_inj, Nx, inj_uA_per_cm2, idx_inj)

        # Ionic current
        Iion_curr = sh_torch.Iion(V, Nx, m_RA, n_RA, h_RA)

        # Right-hand side
        rhs = V + (s.dt / (2.0 * s.Cm)) * (Iinj - Iion_curr)

        # Update gates
        m_RA, h_RA, n_RA = sh_torch.half_step_gates(s.dt, V, Nx, m_RA, h_RA, n_RA)

        # Boundary conditions
        rhs[0] = s.Vinit
        rhs[-1] = s.Vinit

        # Solve A * V_half = rhs using Thomas algorithm
        V_half = thomas_solve(a, b, c, rhs)

        # Hines extrapolation
        V_new = 2.0 * V_half - V
        V_new = torch.clamp(V_new, -500.0, 500.0)
        V_new[0] = s.Vinit
        V_new[-1] = s.Vinit

        V_all[n, :] = V_new
        V = V_new

    return t_vec.cpu().numpy(), V_all.cpu().numpy()

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
t_v = []
for Nx in s.Nx_v:
    res, t = minibench(run_torch_thomas, Nx, n_iter=s.n_iter)
    t_v.append(t)

df = u.res_to_df(s.Nx_v, t_v, label="torch_thomas")
u.append_to_csv(df)

# ------------------------------------------------------------
# Plot results
# ------------------------------------------------------------
t_vec, V_all = res
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
