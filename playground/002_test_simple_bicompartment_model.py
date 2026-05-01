# test_multidomain_membrane.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


Array = jnp.ndarray


# ============================================================
# 1. Simple membrane models
# ============================================================

class PassiveMembrane:
    n_gates = 0

    def __init__(self, gl=0.3, el=-54.3):
        self.gl = gl
        self.el = el

    def init_gates(self, V: Array) -> Array:
        return jnp.zeros((V.shape[0], 0))

    def step_ion(self, V: Array, gates: Array, dt: float):
        g_l = self.gl * jnp.ones_like(V)
        Iion = g_l * (V - self.el)
        Gtot = g_l
        return gates, Iion, Gtot


class HHMembrane:
    n_gates = 3

    def __init__(
        self,
        gnabar=120.0,
        gkbar=36.0,
        gl=0.3,
        ena=50.0,
        ek=-77.0,
        el=-54.3,
        celsius=6.3,
    ):
        self.gnabar = gnabar
        self.gkbar = gkbar
        self.gl = gl
        self.ena = ena
        self.ek = ek
        self.el = el
        self.q10 = 3.0 ** ((celsius - 6.3) / 10.0)

    @staticmethod
    def vtrap(x, y):
        z = x / y
        return jnp.where(
            jnp.abs(z) < 1e-7,
            y * (1.0 - z / 2.0),
            x / (jnp.exp(z) - 1.0),
        )

    def rates(self, V: Array):
        alpha_m = 0.1 * self.vtrap(-(V + 40.0), 10.0)
        beta_m = 4.0 * jnp.exp(-(V + 65.0) / 18.0)

        alpha_h = 0.07 * jnp.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (jnp.exp(-(V + 35.0) / 10.0) + 1.0)

        alpha_n = 0.01 * self.vtrap(-(V + 55.0), 10.0)
        beta_n = 0.125 * jnp.exp(-(V + 65.0) / 80.0)

        alpha = jnp.stack([alpha_m, alpha_h, alpha_n], axis=1)
        beta = jnp.stack([beta_m, beta_h, beta_n], axis=1)
        return alpha, beta

    def init_gates(self, V: Array) -> Array:
        alpha, beta = self.rates(V)
        return alpha / jnp.maximum(alpha + beta, 1e-12)

    def step_ion(self, V: Array, gates: Array, dt: float):
        alpha, beta = self.rates(V)
        alpha = self.q10 * alpha
        beta = self.q10 * beta

        inv_dt = 1.0 / dt
        ab = alpha + beta
        denom = jnp.maximum(inv_dt + 0.5 * ab, 1e-12)

        gates_new = alpha / denom + ((inv_dt - 0.5 * ab) / denom) * gates

        m = gates_new[:, 0]
        h = gates_new[:, 1]
        n = gates_new[:, 2]

        g_na = self.gnabar * m**3 * h
        g_k = self.gkbar * n**4
        g_l = self.gl * jnp.ones_like(V)

        Iion = (
            g_na * (V - self.ena)
            + g_k * (V - self.ek)
            + g_l * (V - self.el)
        )

        Gtot = g_na + g_k + g_l

        return gates_new, Iion, Gtot


# ============================================================
# 2. Multi-domain membrane wrapper
# ============================================================

@dataclass(frozen=True)
class Domain:
    name: str
    indices: Array
    model: object


class MultiDomainMembrane:
    def __init__(self, domains: Tuple[Domain, ...]):
        self.domains = domains

    def init_gates(self, V: Array):
        gates = []
        for d in self.domains:
            gates.append(d.model.init_gates(V[d.indices]))
        return tuple(gates)

    def step_ion(self, V: Array, gates_tuple: tuple, dt: float):
        Iion = jnp.zeros_like(V)
        Gtot = jnp.zeros_like(V)

        gates_new = []

        for d, gates_d in zip(self.domains, gates_tuple):
            idx = d.indices
            Vd = V[idx]

            gates_d_new, I_d, G_d = d.model.step_ion(Vd, gates_d, dt)

            Iion = Iion.at[idx].set(I_d)
            Gtot = Gtot.at[idx].set(G_d)
            gates_new.append(gates_d_new)

        return tuple(gates_new), Iion, Gtot


# ============================================================
# 3. Diffusion operator
# ============================================================

def diffusion_coeffs(Nx: int, D: float, dx: float):
    lower = jnp.zeros((Nx,))
    diag = jnp.zeros((Nx,))
    upper = jnp.zeros((Nx,))

    c = D / dx**2

    lower = lower.at[1:].set(c)
    upper = upper.at[:-1].set(c)
    diag = diag.at[:].set(-2.0 * c)

    # sealed-end Neumann-like boundaries
    diag = diag.at[0].set(-2.0 * c)
    upper = upper.at[0].set(2.0 * c)

    lower = lower.at[-1].set(2.0 * c)
    diag = diag.at[-1].set(-2.0 * c)

    return lower, diag, upper


def apply_diffusion(V: Array, lower: Array, diag: Array, upper: Array):
    out = diag * V
    out = out.at[1:].add(lower[1:] * V[:-1])
    out = out.at[:-1].add(upper[:-1] * V[1:])
    return out


# ============================================================
# 4. Simple explicit test solver
# ============================================================

def run_simulation():
    Nx = 201
    L = 1000.0
    dx = L / (Nx - 1)

    dt = 0.001
    tsim = 10.0
    Nt = int(tsim / dt)

    Vinit = -65.0
    V0 = jnp.full((Nx,), Vinit)

    x = jnp.linspace(0.0, L, Nx)

    # Domains:
    # passive everywhere except active HH patch around center
    center = L / 2.0
    active_mask = jnp.abs(x - center) < 80.0

    active_idx = jnp.where(active_mask, size=Nx, fill_value=-1)[0]
    active_idx = active_idx[active_idx >= 0]

    passive_idx = jnp.where(~active_mask, size=Nx, fill_value=-1)[0]
    passive_idx = passive_idx[passive_idx >= 0]

    membrane = MultiDomainMembrane(
        domains=(
            Domain("passive", passive_idx, PassiveMembrane()),
            Domain("active_hh", active_idx, HHMembrane()),
        )
    )

    gates0 = membrane.init_gates(V0)

    lower, diag, upper = diffusion_coeffs(Nx=Nx, D=0.05, dx=dx)

    def Iinj(t):
        amp = 20.0
        return jnp.where((t >= 1.0) & (t <= 1.5), amp, 0.0)

    stim_idx = int(Nx // 2)

    def step(carry, n):
        V, gates = carry
        t = n * dt

        gates_new, Iion, _ = membrane.step_ion(V, gates, dt)

        LV = apply_diffusion(V, lower, diag, upper)

        I = jnp.zeros_like(V)
        I = I.at[stim_idx].set(Iinj(t))

        dVdt = LV + I - Iion

        V_new = V + dt * dVdt

        return (V_new, gates_new), V_new

    (_, _), V_all = jax.lax.scan(step, (V0, gates0), jnp.arange(Nt))

    t = jnp.arange(Nt) * dt

    return x, t, V_all, active_idx, passive_idx


# ============================================================
# 5. Run
# ============================================================

if __name__ == "__main__":
    x, t, V_all, active_idx, passive_idx = run_simulation()

    plt.figure(figsize=(10, 4))
    plt.imshow(
        V_all.T,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(x[0]), float(x[-1])],
        cmap="viridis",
    )
    plt.colorbar(label="Vm [mV]")
    plt.xlabel("Time [ms]")
    plt.ylabel("Position [µm]")
    plt.title("Multi-domain membrane test: passive + active HH patch")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    for pos in [250, 400, 500, 600, 750]:
        idx = int(jnp.argmin(jnp.abs(x - pos)))
        plt.plot(t, V_all[:, idx], label=f"x={pos} µm")

    plt.xlabel("Time [ms]")
    plt.ylabel("Vm [mV]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()