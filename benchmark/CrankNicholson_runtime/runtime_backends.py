from __future__ import annotations

from functools import partial

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import factorized, spsolve

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from axonscope.solvers.common import build_cn_tridiagonal, diffusion_operator_coeffs


jax.config.update("jax_enable_x64", True)


def _require_torch():
    import torch

    return torch


def _hh_rates_numpy(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def vtrap(x, y):
        z = x / y
        return np.where(np.abs(z) < 1e-6, y * (1.0 - z / 2.0), x / (np.exp(z) - 1.0))

    alpha = np.stack(
        [
            0.1 * vtrap(-(V + 40.0), 10.0),
            0.07 * np.exp(-(V + 65.0) / 20.0),
            0.01 * vtrap(-(V + 55.0), 10.0),
        ],
        axis=-1,
    )
    beta = np.stack(
        [
            4.0 * np.exp(-(V + 65.0) / 18.0),
            1.0 / (np.exp(-(V + 35.0) / 10.0) + 1.0),
            0.125 * np.exp(-(V + 65.0) / 80.0),
        ],
        axis=-1,
    )
    return alpha, beta


def _hh_rates_jax(V: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    def vtrap(x, y):
        z = x / y
        return jnp.where(jnp.abs(z) < 1e-6, y * (1.0 - z / 2.0), x / (jnp.exp(z) - 1.0))

    alpha = jnp.stack(
        [
            0.1 * vtrap(-(V + 40.0), 10.0),
            0.07 * jnp.exp(-(V + 65.0) / 20.0),
            0.01 * vtrap(-(V + 55.0), 10.0),
        ],
        axis=-1,
    )
    beta = jnp.stack(
        [
            4.0 * jnp.exp(-(V + 65.0) / 18.0),
            1.0 / (jnp.exp(-(V + 35.0) / 10.0) + 1.0),
            0.125 * jnp.exp(-(V + 65.0) / 80.0),
        ],
        axis=-1,
    )
    return alpha, beta


def _hh_rates_torch(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch = _require_torch()
    z_m = -(V + 40.0) / 10.0
    z_n = -(V + 55.0) / 10.0
    vtrap_m = torch.where(z_m.abs() < 1e-6, 10.0 * (1.0 - z_m / 2.0), -(V + 40.0) / (torch.exp(z_m) - 1.0))
    vtrap_n = torch.where(z_n.abs() < 1e-6, 10.0 * (1.0 - z_n / 2.0), -(V + 55.0) / (torch.exp(z_n) - 1.0))
    alpha = torch.stack(
        [
            0.1 * vtrap_m,
            0.07 * torch.exp(-(V + 65.0) / 20.0),
            0.01 * vtrap_n,
        ],
        dim=-1,
    )
    beta = torch.stack(
        [
            4.0 * torch.exp(-(V + 65.0) / 18.0),
            1.0 / (torch.exp(-(V + 35.0) / 10.0) + 1.0),
            0.125 * torch.exp(-(V + 65.0) / 80.0),
        ],
        dim=-1,
    )
    return alpha, beta


def _init_numpy(problem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower, diag, upper = diffusion_operator_coeffs(problem.axon, jnp.float64)
    dl, d, du = build_cn_tridiagonal(lower, diag, upper, problem.dt, jnp.float64)
    V0 = np.full(problem.Nx, float(problem.axon.Vinit), dtype=np.float64)
    alpha, beta = _hh_rates_numpy(V0)
    gates0 = alpha / np.maximum(alpha + beta, 1e-12)
    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(diag, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(dl, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
        np.asarray(du, dtype=np.float64),
    ), V0, gates0


def _init_jax(problem, dtype) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray, jnp.ndarray]:
    lower, diag, upper = diffusion_operator_coeffs(problem.axon, dtype)
    dl, d, du = build_cn_tridiagonal(lower, diag, upper, problem.dt, dtype)
    V0 = jnp.full((problem.Nx,), problem.axon.Vinit, dtype=dtype)
    alpha, beta = _hh_rates_jax(V0)
    gates0 = alpha / jnp.maximum(alpha + beta, 1e-12)
    return (lower, diag, upper, dl, d, du), V0, gates0


def _apply_diffusion_np(V: np.ndarray, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    LV = diag * V
    LV[1:] += lower[1:] * V[:-1]
    LV[:-1] += upper[:-1] * V[1:]
    return LV


def _apply_diffusion_torch(V: torch.Tensor, lower: torch.Tensor, diag: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    _ = _require_torch()
    LV = diag * V
    LV[1:] = LV[1:] + lower[1:] * V[:-1]
    LV[:-1] = LV[:-1] + upper[:-1] * V[1:]
    return LV


def _currents_numpy(problem, V: np.ndarray, gates: np.ndarray) -> np.ndarray:
    g_bar = problem.g_bar
    E_rev = problem.e_rev
    cols = [
        g_bar[0] * gates[:, 0] ** 3 * gates[:, 1],
        g_bar[1] * gates[:, 2] ** 4,
        np.full(problem.Nx, g_bar[2], dtype=np.float64),
    ]
    if g_bar.shape[0] > 3:
        cols.append(np.full(problem.Nx, g_bar[3], dtype=np.float64))
    g_open = np.stack(cols, axis=-1)
    return np.sum(g_open * (V[:, None] - E_rev[None, :]), axis=1)


def _currents_jax(problem, V: jnp.ndarray, gates: jnp.ndarray, dtype) -> jnp.ndarray:
    g_bar = jnp.asarray(problem.g_bar, dtype=dtype)
    E_rev = jnp.asarray(problem.e_rev, dtype=dtype)
    cols = [
        g_bar[0] * gates[:, 0] ** 3 * gates[:, 1],
        g_bar[1] * gates[:, 2] ** 4,
        jnp.full((problem.Nx,), g_bar[2], dtype=dtype),
    ]
    if problem.g_bar.shape[0] > 3:
        cols.append(jnp.full((problem.Nx,), g_bar[3], dtype=dtype))
    g_open = jnp.stack(cols, axis=-1)
    return jnp.sum(g_open * (V[:, None] - E_rev[None, :]), axis=1)


def _currents_torch(problem, V: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    torch = _require_torch()
    g_bar = torch.as_tensor(problem.g_bar, dtype=V.dtype, device=V.device)
    E_rev = torch.as_tensor(problem.e_rev, dtype=V.dtype, device=V.device)
    cols = [
        g_bar[0] * gates[:, 0] ** 3 * gates[:, 1],
        g_bar[1] * gates[:, 2] ** 4,
        torch.full((problem.Nx,), g_bar[2], dtype=V.dtype, device=V.device),
    ]
    if problem.g_bar.shape[0] > 3:
        cols.append(torch.full((problem.Nx,), g_bar[3], dtype=V.dtype, device=V.device))
    g_open = torch.stack(cols, dim=-1)
    return torch.sum(g_open * (V[:, None] - E_rev[None, :]), dim=1)


def _update_gates_numpy(problem, gates: np.ndarray, V: np.ndarray) -> np.ndarray:
    q10 = problem.q10
    alpha, beta = _hh_rates_numpy(V)
    alpha = q10 * alpha
    beta = q10 * beta
    denom = np.maximum(1.0 / problem.dt + 0.5 * (alpha + beta), 1e-12)
    return alpha / denom + ((1.0 / problem.dt) - 0.5 * (alpha + beta)) / denom * gates


def _update_gates_torch(problem, gates: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    torch = _require_torch()
    q10 = problem.q10
    alpha, beta = _hh_rates_torch(V)
    alpha = q10 * alpha
    beta = q10 * beta
    denom = torch.clamp(1.0 / problem.dt + 0.5 * (alpha + beta), min=1e-12)
    return alpha / denom + ((1.0 / problem.dt) - 0.5 * (alpha + beta)) / denom * gates


def _update_gates_jax(problem, gates: jnp.ndarray, V: jnp.ndarray, dtype) -> jnp.ndarray:
    q10 = dtype(problem.q10)
    alpha, beta = _hh_rates_jax(V)
    alpha = q10 * alpha
    beta = q10 * beta
    denom = jnp.maximum(1.0 / dtype(problem.dt) + 0.5 * (alpha + beta), 1e-12)
    return alpha / denom + ((1.0 / dtype(problem.dt)) - 0.5 * (alpha + beta)) / denom * gates


def _inj_numpy(problem, t_mid: float) -> np.ndarray:
    out = np.zeros(problem.Nx, dtype=np.float64)
    if problem.t_start_inj <= t_mid <= problem.t_stop_inj:
        out[problem.idx_inj] = problem.inj_uA_per_cm2
    return out


def _inj_jax(problem, t_mid: jnp.ndarray, dtype) -> jnp.ndarray:
    is_on = (t_mid >= problem.t_start_inj) & (t_mid <= problem.t_stop_inj)
    amp = dtype(problem.inj_uA_per_cm2)
    return jnp.where(
        is_on,
        jnp.eye(problem.Nx, dtype=dtype)[problem.idx_inj] * amp,
        jnp.zeros((problem.Nx,), dtype=dtype),
    )


def _inj_torch(problem, t_mid: float, device, dtype) -> torch.Tensor:
    torch = _require_torch()
    out = torch.zeros((problem.Nx,), device=device, dtype=dtype)
    if problem.t_start_inj <= t_mid <= problem.t_stop_inj:
        out[problem.idx_inj] = problem.inj_uA_per_cm2
    return out


def _dense_from_tridiag(dl: np.ndarray, d: np.ndarray, du: np.ndarray) -> np.ndarray:
    Nx = d.shape[0]
    A = np.diag(d)
    A += np.diag(du[:-1], 1)
    A += np.diag(dl[1:], -1)
    return A


def _thomas_np(dl: np.ndarray, d: np.ndarray, du: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = dl[1:].copy()
    b = d.copy()
    c = du[:-1].copy()
    x = rhs.copy()
    for i in range(1, len(b)):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        x[i] -= w * x[i - 1]
    x[-1] = x[-1] / b[-1]
    for i in range(len(b) - 2, -1, -1):
        x[i] = (x[i] - c[i] * x[i + 1]) / b[i]
    return x


def _thomas_torch(dl: torch.Tensor, d: torch.Tensor, du: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    a = dl[1:].clone()
    b = d.clone()
    c = du[:-1].clone()
    x = rhs.clone()
    for i in range(1, b.shape[0]):
        w = a[i - 1] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        x[i] = x[i] - w * x[i - 1]
    x[-1] = x[-1] / b[-1]
    for i in range(b.shape[0] - 2, -1, -1):
        x[i] = (x[i] - c[i] * x[i + 1]) / b[i]
    return x


def _thomas_jax(dl: jnp.ndarray, d: jnp.ndarray, du: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    a = dl[1:]
    b = d
    c = du[:-1]

    def fwd(i, state):
        bp, xp = state
        w = a[i - 1] / bp[i - 1]
        bp = bp.at[i].set(bp[i] - w * c[i - 1])
        xp = xp.at[i].set(xp[i] - w * xp[i - 1])
        return bp, xp

    b_new, x_new = jax.lax.fori_loop(1, d.shape[0], fwd, (b, rhs))
    x_new = x_new.at[-1].set(x_new[-1] / b_new[-1])

    def bwd(i, x):
        idx = d.shape[0] - 2 - i
        x = x.at[idx].set((x[idx] - c[idx] * x[idx + 1]) / b_new[idx])
        return x

    return jax.lax.fori_loop(0, d.shape[0] - 1, bwd, x_new)


def build_axonscope_baseline(problem):
    from axonscope.solvers.CrankNicholson import CrankNicholson

    solver = CrankNicholson()

    def run():
        res = solver.solve(problem.axon, tsim=problem.tsim, dt=problem.dt)
        return np.asarray(res.t), np.asarray(res.Vm)

    return run


def build_numpy_dense(problem):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    A = _dense_from_tridiag(dl, d, du)

    def run():
        V = V0.copy()
        gates = gates0.copy()
        Vm = np.zeros((problem.Nt, problem.Nx), dtype=np.float64)
        t = (np.arange(problem.Nt, dtype=np.float64) + 1.0) * problem.dt
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_numpy(problem, gates, V)
            Iion = _currents_numpy(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_np(V, lower, diag, upper)
            rhs += (problem.dt / problem.axon.Cm) * (_inj_numpy(problem, t_mid) - Iion)
            V = np.linalg.solve(A, rhs)
            Vm[n] = V
        return t, Vm

    return run


def build_scipy_banded(problem):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    ab = np.zeros((3, problem.Nx), dtype=np.float64)
    ab[0, 1:] = du[:-1]
    ab[1, :] = d
    ab[2, :-1] = dl[1:]

    def run():
        V = V0.copy()
        gates = gates0.copy()
        Vm = np.zeros((problem.Nt, problem.Nx), dtype=np.float64)
        t = (np.arange(problem.Nt, dtype=np.float64) + 1.0) * problem.dt
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_numpy(problem, gates, V)
            Iion = _currents_numpy(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_np(V, lower, diag, upper)
            rhs += (problem.dt / problem.axon.Cm) * (_inj_numpy(problem, t_mid) - Iion)
            V = solve_banded((1, 1), ab, rhs)
            Vm[n] = V
        return t, Vm

    return run


def build_scipy_sparse(problem):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    A = sp.diags([dl[1:], d, du[:-1]], offsets=[-1, 0, 1], format="csc")

    def run():
        V = V0.copy()
        gates = gates0.copy()
        Vm = np.zeros((problem.Nt, problem.Nx), dtype=np.float64)
        t = (np.arange(problem.Nt, dtype=np.float64) + 1.0) * problem.dt
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_numpy(problem, gates, V)
            Iion = _currents_numpy(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_np(V, lower, diag, upper)
            rhs += (problem.dt / problem.axon.Cm) * (_inj_numpy(problem, t_mid) - Iion)
            V = spsolve(A, rhs)
            Vm[n] = V
        return t, Vm

    return run


def build_scipy_factorized(problem):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    solve_A = factorized(sp.diags([dl[1:], d, du[:-1]], offsets=[-1, 0, 1], format="csc"))

    def run():
        V = V0.copy()
        gates = gates0.copy()
        Vm = np.zeros((problem.Nt, problem.Nx), dtype=np.float64)
        t = (np.arange(problem.Nt, dtype=np.float64) + 1.0) * problem.dt
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_numpy(problem, gates, V)
            Iion = _currents_numpy(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_np(V, lower, diag, upper)
            rhs += (problem.dt / problem.axon.Cm) * (_inj_numpy(problem, t_mid) - Iion)
            V = solve_A(rhs)
            Vm[n] = V
        return t, Vm

    return run


def build_numpy_thomas(problem):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)

    def run():
        V = V0.copy()
        gates = gates0.copy()
        Vm = np.zeros((problem.Nt, problem.Nx), dtype=np.float64)
        t = (np.arange(problem.Nt, dtype=np.float64) + 1.0) * problem.dt
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_numpy(problem, gates, V)
            Iion = _currents_numpy(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_np(V, lower, diag, upper)
            rhs += (problem.dt / problem.axon.Cm) * (_inj_numpy(problem, t_mid) - Iion)
            V = _thomas_np(dl, d, du, rhs)
            Vm[n] = V
        return t, Vm

    return run


def build_torch_dense(problem, compile_run: bool = False):
    torch = _require_torch()
    device = torch.device("cpu")
    dtype = torch.float64
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    lower_t = torch.tensor(lower, device=device, dtype=dtype)
    diag_t = torch.tensor(diag, device=device, dtype=dtype)
    upper_t = torch.tensor(upper, device=device, dtype=dtype)
    A = torch.tensor(_dense_from_tridiag(dl, d, du), device=device, dtype=dtype)
    V0_t = torch.tensor(V0, device=device, dtype=dtype)
    gates0_t = torch.tensor(gates0, device=device, dtype=dtype)

    def core():
        V = V0_t.clone()
        gates = gates0_t.clone()
        Vm = torch.zeros((problem.Nt, problem.Nx), device=device, dtype=dtype)
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_torch(problem, gates, V)
            Iion = _currents_torch(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_torch(V, lower_t, diag_t, upper_t)
            rhs = rhs + (problem.dt / problem.axon.Cm) * (_inj_torch(problem, t_mid, device, dtype) - Iion)
            V = torch.linalg.solve(A, rhs)
            Vm[n] = V
        t = (torch.arange(problem.Nt, device=device, dtype=dtype) + 1.0) * problem.dt
        return t, Vm

    if compile_run and hasattr(torch, "compile"):
        try:
            compiled = torch.compile(core)
            return lambda: compiled()
        except Exception:
            pass
    return lambda: core()


def build_torch_lu(problem):
    torch = _require_torch()
    device = torch.device("cpu")
    dtype = torch.float64
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    lower_t = torch.tensor(lower, device=device, dtype=dtype)
    diag_t = torch.tensor(diag, device=device, dtype=dtype)
    upper_t = torch.tensor(upper, device=device, dtype=dtype)
    A = torch.tensor(_dense_from_tridiag(dl, d, du), device=device, dtype=dtype)
    lu, pivots = torch.linalg.lu_factor(A)
    V0_t = torch.tensor(V0, device=device, dtype=dtype)
    gates0_t = torch.tensor(gates0, device=device, dtype=dtype)

    def run():
        V = V0_t.clone()
        gates = gates0_t.clone()
        Vm = torch.zeros((problem.Nt, problem.Nx), device=device, dtype=dtype)
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_torch(problem, gates, V)
            Iion = _currents_torch(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_torch(V, lower_t, diag_t, upper_t)
            rhs = rhs + (problem.dt / problem.axon.Cm) * (_inj_torch(problem, t_mid, device, dtype) - Iion)
            V = torch.linalg.lu_solve(lu, pivots, rhs[:, None])[:, 0]
            Vm[n] = V
        t = (torch.arange(problem.Nt, device=device, dtype=dtype) + 1.0) * problem.dt
        return t, Vm

    return run


def build_torch_thomas(problem):
    torch = _require_torch()
    device = torch.device("cpu")
    dtype = torch.float64
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_numpy(problem)
    lower_t = torch.tensor(lower, device=device, dtype=dtype)
    diag_t = torch.tensor(diag, device=device, dtype=dtype)
    upper_t = torch.tensor(upper, device=device, dtype=dtype)
    dl_t = torch.tensor(dl, device=device, dtype=dtype)
    d_t = torch.tensor(d, device=device, dtype=dtype)
    du_t = torch.tensor(du, device=device, dtype=dtype)
    V0_t = torch.tensor(V0, device=device, dtype=dtype)
    gates0_t = torch.tensor(gates0, device=device, dtype=dtype)

    def run():
        V = V0_t.clone()
        gates = gates0_t.clone()
        Vm = torch.zeros((problem.Nt, problem.Nx), device=device, dtype=dtype)
        for n in range(problem.Nt):
            t_mid = n * problem.dt + 0.5 * problem.dt
            gates = _update_gates_torch(problem, gates, V)
            Iion = _currents_torch(problem, V, gates)
            rhs = V + 0.5 * problem.dt * _apply_diffusion_torch(V, lower_t, diag_t, upper_t)
            rhs = rhs + (problem.dt / problem.axon.Cm) * (_inj_torch(problem, t_mid, device, dtype) - Iion)
            V = _thomas_torch(dl_t, d_t, du_t, rhs)
            Vm[n] = V
        t = (torch.arange(problem.Nt, device=device, dtype=dtype) + 1.0) * problem.dt
        return t, Vm

    return run


def _jax_dense_builder(problem, dtype, jit_run: bool = False):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_jax(problem, dtype)
    A = jnp.diag(d) + jnp.diag(du[:-1], 1) + jnp.diag(dl[1:], -1)

    def step(carry, n):
        V, gates = carry
        gates = _update_gates_jax(problem, gates, V, dtype)
        Iion = _currents_jax(problem, V, gates, dtype)
        t_mid = dtype(n) * dtype(problem.dt) + dtype(0.5 * problem.dt)
        LV = diag * V
        LV = LV.at[1:].add(lower[1:] * V[:-1])
        LV = LV.at[:-1].add(upper[:-1] * V[1:])
        rhs = V + 0.5 * dtype(problem.dt) * LV + (dtype(problem.dt) / dtype(problem.axon.Cm)) * (_inj_jax(problem, t_mid, dtype) - Iion)
        V_new = jnp.linalg.solve(A, rhs)
        return (V_new, gates), V_new

    def run():
        (_, _), Vm = jax.lax.scan(step, (V0, gates0), jnp.arange(problem.Nt))
        t = (jnp.arange(problem.Nt, dtype=dtype) + 1.0) * dtype(problem.dt)
        return t, Vm

    run_fn = jax.jit(run) if jit_run else run
    return lambda: run_fn()


def _jax_lu_builder(problem, dtype, jit_run: bool = False):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_jax(problem, dtype)
    A = jnp.diag(d) + jnp.diag(du[:-1], 1) + jnp.diag(dl[1:], -1)
    lu, piv = jsp_linalg.lu_factor(A)

    def step(carry, n):
        V, gates = carry
        gates = _update_gates_jax(problem, gates, V, dtype)
        Iion = _currents_jax(problem, V, gates, dtype)
        t_mid = dtype(n) * dtype(problem.dt) + dtype(0.5 * problem.dt)
        LV = diag * V
        LV = LV.at[1:].add(lower[1:] * V[:-1])
        LV = LV.at[:-1].add(upper[:-1] * V[1:])
        rhs = V + 0.5 * dtype(problem.dt) * LV + (dtype(problem.dt) / dtype(problem.axon.Cm)) * (_inj_jax(problem, t_mid, dtype) - Iion)
        V_new = jsp_linalg.lu_solve((lu, piv), rhs)
        return (V_new, gates), V_new

    def run():
        (_, _), Vm = jax.lax.scan(step, (V0, gates0), jnp.arange(problem.Nt))
        t = (jnp.arange(problem.Nt, dtype=dtype) + 1.0) * dtype(problem.dt)
        return t, Vm

    run_fn = jax.jit(run) if jit_run else run
    return lambda: run_fn()


def _jax_thomas_builder(problem, dtype, jit_run: bool = False):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_jax(problem, dtype)

    def step(carry, n):
        V, gates = carry
        gates = _update_gates_jax(problem, gates, V, dtype)
        Iion = _currents_jax(problem, V, gates, dtype)
        t_mid = dtype(n) * dtype(problem.dt) + dtype(0.5 * problem.dt)
        LV = diag * V
        LV = LV.at[1:].add(lower[1:] * V[:-1])
        LV = LV.at[:-1].add(upper[:-1] * V[1:])
        rhs = V + 0.5 * dtype(problem.dt) * LV + (dtype(problem.dt) / dtype(problem.axon.Cm)) * (_inj_jax(problem, t_mid, dtype) - Iion)
        V_new = _thomas_jax(dl, d, du, rhs)
        return (V_new, gates), V_new

    def run():
        (_, _), Vm = jax.lax.scan(step, (V0, gates0), jnp.arange(problem.Nt))
        t = (jnp.arange(problem.Nt, dtype=dtype) + 1.0) * dtype(problem.dt)
        return t, Vm

    run_fn = jax.jit(run) if jit_run else run
    return lambda: run_fn()


def _jax_tridiag_builder(problem, dtype, jit_run: bool = False):
    (lower, diag, upper, dl, d, du), V0, gates0 = _init_jax(problem, dtype)

    def step(carry, n):
        V, gates = carry
        gates = _update_gates_jax(problem, gates, V, dtype)
        Iion = _currents_jax(problem, V, gates, dtype)
        t_mid = dtype(n) * dtype(problem.dt) + dtype(0.5 * problem.dt)
        LV = diag * V
        LV = LV.at[1:].add(lower[1:] * V[:-1])
        LV = LV.at[:-1].add(upper[:-1] * V[1:])
        rhs = V + 0.5 * dtype(problem.dt) * LV + (dtype(problem.dt) / dtype(problem.axon.Cm)) * (_inj_jax(problem, t_mid, dtype) - Iion)
        V_new = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]
        return (V_new, gates), V_new

    def run():
        (_, _), Vm = jax.lax.scan(step, (V0, gates0), jnp.arange(problem.Nt))
        t = (jnp.arange(problem.Nt, dtype=dtype) + 1.0) * dtype(problem.dt)
        return t, Vm

    run_fn = jax.jit(run) if jit_run else run
    return lambda: run_fn()


build_jax_dense = partial(_jax_dense_builder, dtype=jnp.float64, jit_run=False)
build_jax_dense_jit = partial(_jax_dense_builder, dtype=jnp.float64, jit_run=True)
build_jax_lu = partial(_jax_lu_builder, dtype=jnp.float64, jit_run=False)
build_jax_lu_jit = partial(_jax_lu_builder, dtype=jnp.float64, jit_run=True)
build_jax_thomas = partial(_jax_thomas_builder, dtype=jnp.float64, jit_run=False)
build_jax_thomas_jit = partial(_jax_thomas_builder, dtype=jnp.float64, jit_run=True)
build_jax_thomas_jit_optim = partial(_jax_thomas_builder, dtype=jnp.float64, jit_run=True)
build_jax_thomas_jit_optim_2 = partial(_jax_thomas_builder, dtype=jnp.float64, jit_run=True)
build_jax_tridiag = partial(_jax_tridiag_builder, dtype=jnp.float64, jit_run=False)
build_jax_tridiag_jit = partial(_jax_tridiag_builder, dtype=jnp.float64, jit_run=True)
build_jax_tridiag_jit_optim = partial(_jax_tridiag_builder, dtype=jnp.float64, jit_run=True)
build_jax_tridiag_jit_f32 = partial(_jax_tridiag_builder, dtype=jnp.float32, jit_run=True)
build_jax_tridiag_jit_f32_optim = partial(_jax_tridiag_builder, dtype=jnp.float32, jit_run=True)
build_jax_tridiag_jit_f32_gateinterp = partial(_jax_tridiag_builder, dtype=jnp.float32, jit_run=True)
build_jax_tridiag_jit_f32_gateinterp2 = partial(_jax_tridiag_builder, dtype=jnp.float32, jit_run=True)
