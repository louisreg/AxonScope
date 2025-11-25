from __future__ import annotations

import jax.numpy as jnp
from typing import List, Callable
import matplotlib.pyplot as plt
import numpy as np
import time

from axonscope.channel_models.base_channel_model import IonChannelModelBase
from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM
from axonscope.math_functions import vtrap_jax as vtrap
from axonscope.settings import dtype
from axonscope.icm_compute import Gating
from axonscope.axons import GenericAxon, HodgkinHuxley

from axonscope.solvers import CrankNicholson

Array1D = jnp.ndarray
Array2D = jnp.ndarray
GFunc = Callable[[Array2D, Array1D], Array2D]
RateFunc = Callable[[Array1D], Array2D]


class CompositeIonChannelModel(IonChannelModelBase):


    def __init__(self, models: List[IonChannelModelBase]):
        super().__init__()
        self.models: List[IonChannelModelBase] = models  # must be static for JIT --> realy??

        # sizes per submodel (Python ints)
        sizes = [int(m.init_gates(jnp.array([0.0])).shape[-1]) for m in models]
        self.sizes: List[int] = sizes

        # cumulative boundaries for slicing gates
        cum = [0]
        for s in sizes:
            cum.append(cum[-1] + s)
        self.cum_sizes: List[int] = cum  # length = len(models) + 1


    @property
    def g_bar(self):
        # concatenation of channel maximal conductances (vector)
        return jnp.concatenate([m.g_bar for m in self.models])

    @property
    def E_rev(self):
        # concatenation of reversal potentials (vector)
        return jnp.concatenate([m.E_rev for m in self.models])

    # -------------------------
    # init_gates
    # -------------------------
    def init_gates(self, V0_mV):
        """
        Return initial gating variables for each submodel concatenated.
        Output shape: (batch, total_gates)
        """
        outs = []
        for m in self.models:
            g = m.init_gates(V0_mV)            # shape maybe (batch, n_i) or (n_i,)
            g = jnp.atleast_2d(g)          # ensure (batch, n_i)
            outs.append(g)
        if len(outs) == 0:
            return jnp.zeros((V0_mV.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


    def alpha_funcs(self, V):
        """
        Concatenate alpha(V) from each submodel.
        Output shape: (batch, total_gates)
        """
        outs = []
        for m in self.models:
            a = m.alpha_funcs(V)   # expected (batch, n_i) or (n_i,)
            a = jnp.atleast_2d(a)
            outs.append(a)
        if len(outs) == 0:
            return jnp.zeros((V.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


    def beta_funcs(self, V):
        """
        Concatenate beta(V) from each submodel.
        Output shape: (batch, total_gates)
        """
        outs = []
        for m in self.models:
            b = m.beta_funcs(V)
            b = jnp.atleast_2d(b)
            outs.append(b)
        if len(outs) == 0:
            return jnp.zeros((V.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)


    def g_funcs(self, gates, g_bar_unused=None):
        """
        Compute conductances from gating variables using each submodel's g_funcs.
        gates: (batch, total_gates) or (total_gates,)
        Returns concatenated g parts: shape (batch, n_channels_out)
        Note: assumes each submodel.g_funcs returns shape (batch, k_i) (commonly k_i=1).
        """
        outs = []
        for i, m in enumerate(self.models):
            i0 = self.cum_sizes[i]
            i1 = self.cum_sizes[i + 1]
            sub = gates[..., i0:i1]           # (batch, n_i)
            g_part = m.g_funcs(sub, m.g_bar)  # (batch, k_i)
            g_part = jnp.atleast_2d(g_part)
            outs.append(g_part)
        if len(outs) == 0:
            return jnp.zeros((gates.shape[0], 0))
        return jnp.concatenate(outs, axis=-1)

# =========================================================
# Leak
# =========================================================
class HH_leak_MM(IonChannelModelBase):
    def __init__(self, gl=0.0003, el=-54.3):
        super().__init__()
        self.gl = dtype(gl)
        self.el = dtype(el)

    @staticmethod
    def alpha_funcs(V: Array1D) -> Array2D:
        return jnp.zeros((V.shape[0], 0), dtype=dtype)

    @staticmethod
    def beta_funcs(V: Array1D) -> Array2D:
        return jnp.zeros((V.shape[0], 0), dtype=dtype)

    @staticmethod
    def g_funcs(gates: jnp.ndarray, g_bar: jnp.ndarray) -> jnp.ndarray:
        N = gates.shape[0]
        g_out = jnp.full((N,), g_bar[0], dtype=gates.dtype)
        return g_out[:, None]  # force (N,1)

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        g_inf, _ = Gating.rates(V0_mV, 1.0, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> Array1D:
        return jnp.array([self.gl], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> Array1D:
        return jnp.array([self.el], dtype=dtype)

# =========================================================
# Sodium
# =========================================================
class HH_na_MM(IonChannelModelBase):
    def __init__(self, gnabar=0.12, ena=50.0):
        super().__init__()
        self.gnabar = dtype(gnabar)
        self.ena = dtype(ena)

    @staticmethod
    def alpha_funcs(V: Array1D) -> Array2D:
        m = 0.1 * vtrap(-(V + 40.0), 10.0)
        h = 0.07 * jnp.exp(-(V + 65.0)/20.0)
        return jnp.stack([m, h], axis=-1)

    @staticmethod
    def beta_funcs(V: Array1D) -> Array2D:
        m = 4.0 * jnp.exp(-(V + 65.0)/18.0)
        h = 1.0 / (jnp.exp(-(V + 35.0)/10.0) + 1.0)
        return jnp.stack([m, h], axis=-1)

    @staticmethod
    def g_funcs(gates: Array2D, g_bar: Array1D) -> Array2D:
        m = gates[:, 0] 
        h = gates[:, 1] 
        g_na = g_bar[0] * m**3 * h
        return g_na[:, None]

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        g_inf, _ = Gating.rates(V0_mV, 1.0, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> Array1D:
        return jnp.array([self.gnabar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> Array1D:
        return jnp.array([self.ena], dtype=dtype)

# =========================================================
# Potassium
# =========================================================
class HH_k_MM(IonChannelModelBase):
    def __init__(self, gkbar=0.036, ek=-77.0):
        super().__init__()
        self.gkbar = dtype(gkbar)
        self.ek = dtype(ek)

    @staticmethod
    def alpha_funcs(V: Array1D) -> Array2D:
        #V = jnp.atleast_1d(V)
        n = 0.01 * vtrap(-(V + 55.0), 10.0)
        return n[:, None]

    @staticmethod
    def beta_funcs(V: Array1D) -> Array2D:
        #V = jnp.atleast_1d(V)
        n = 0.125 * jnp.exp(-(V + 65.0)/80.0)
        return n[:, None]

    @staticmethod
    def g_funcs(gates: Array2D, g_bar: Array1D) -> Array2D:
        #gates = jnp.atleast_2d(gates)
        #n = gates[:, 0] if gates.shape[1] > 0 else 0.0
        g_k = g_bar[0] * gates[:, 0]**4
        return g_k[:, None]

    def init_gates(self, V0_mV: Array1D) -> Array2D:
        g_inf, _ = Gating.rates(V0_mV, 1.0, self.alpha_funcs, self.beta_funcs)
        return g_inf

    @property
    def g_bar(self) -> Array1D:
        return jnp.array([self.gkbar], dtype=dtype) * 1e3

    @property
    def E_rev(self) -> Array1D:
        return jnp.array([self.ek], dtype=dtype)



# === Instantiate models ===
mono = HodgkinHuxleyICM()
comp = CompositeIonChannelModel([
    HH_k_MM(),
    HH_na_MM(),
    
    HH_leak_MM(),
])

#comp = CompositeIonChannelModel([HH_na_MM()])

dt = 0.001
tsim = 10
Nx = 11
L = 1_000
d = 0.5

comp_axon = GenericAxon(ion_channel=comp,L=L, d=d, Nx=Nx, Temp=6.3)
mono_axon = GenericAxon(ion_channel=mono,L=L, d=d, Nx=Nx, Temp=6.3)
#HH = mono(L=L, d=d)


V = jnp.linspace(-80, 50, 5)  # petit nombre pour debug

# --- alpha ---

a_mono = mono.alpha_funcs(V)
a_comp = comp.alpha_funcs(V)


"""# m,h,n (ignorer leak)

a_comp_reordered = jnp.concatenate([
    a_comp[:, 1:3],  # m,h from HH_na_MM (indices 1 et 2 dans comp alpha)
    a_comp[:, 0:1],  # n from HH_k_MM
], axis=-1)
print("Alpha close:", np.allclose(a_mono, a_comp_reordered))

# --- beta ---

b_mono = mono.beta_funcs(V)
b_comp = comp.beta_funcs(V)
b_comp_reordered = jnp.concatenate([
    b_comp[:, 1:3],  # m,h from HH_na_MM (indices 1 et 2 dans comp alpha)
    b_comp[:, 0:1],  # n from HH_k_MM
], axis=-1)
print("Beta close:", np.allclose(b_mono, b_comp_reordered))

# --- init_gates ---

g_mono = mono.init_gates(V)
g_comp = comp.init_gates(V)
g_comp_reordered = jnp.concatenate([
g_comp[:, 1:3],  # m,h
g_comp[:, 0:1],  # n
], axis=-1)
print("Gates close:", np.allclose(g_mono, g_comp_reordered))

# --- g_funcs ---

gbar_mono = mono.g_bar
gbar_comp = comp.g_bar
g_comp_vals = comp.g_funcs(g_comp, gbar_comp)

# réordonner les conductances Na,K,L

g_comp_vals_reordered = jnp.concatenate([
g_comp_vals[:, 2:3],  # Na
g_comp_vals[:, 0:1],  # K
g_comp_vals[:, 1:2],  # Leak
], axis=-1)
g_mono_vals = mono.g_funcs(g_mono, gbar_mono)
print("g_funcs close:", np.allclose(g_mono_vals, g_comp_vals_reordered))


# --- init_gates ---
gates_mono = mono.init_gates(V)
gates_comp = comp.init_gates(V)

# Réordonner les gates dans comp pour correspondre à l'ordre m,h,n du mono
gates_comp_reordered = jnp.concatenate([
    gates_comp[:, 1:3],  # m,h de HH_na_MM
    gates_comp[:, 0:1],  # n de HH_k_MM
], axis=-1)

print("init_gates close:", np.allclose(gates_mono, gates_comp_reordered))
print("Gates mono:\n", gates_mono)
print("Gates comp_reordered:\n", gates_comp_reordered)

# --- g_bar ---
gbar_mono = mono.g_bar
gbar_comp = comp.g_bar
# Réordonner comp pour l'ordre Na,K,L du mono
gbar_comp_reordered = jnp.concatenate([
    gbar_comp[2:],  # Na
    gbar_comp[0:1], # K
    gbar_comp[1:2], # Leak
])

print("g_bar close:", np.allclose(gbar_mono, gbar_comp_reordered))
print("g_bar mono:", gbar_mono)
print("g_bar comp_reordered:", gbar_comp_reordered)

# --- E_rev ---
Erev_mono = mono.E_rev
Erev_comp = comp.E_rev
# Réordonner comp pour l'ordre Na,K,L du mono
Erev_comp_reordered = jnp.concatenate([
    Erev_comp[2:],  # Na
    Erev_comp[0:1], # K
    Erev_comp[1:2], # Leak
])

print("E_rev close:", np.allclose(Erev_mono, Erev_comp_reordered))
print("E_rev mono:", Erev_mono)
print("E_rev comp_reordered:", Erev_comp_reordered)

#exit()"""

# --- Axons ---
axons = {
    "comp_axon": comp_axon,
    "mono_axon": mono_axon,
}

benchmarks = {}
# --- Inject current ---
t_start = 1.0
duration = 1.0
amplitude = 5
for axon in axons.values():
    axon.insert_I_Clamp(position=L/2, t_start=t_start, duration=duration, amplitude=amplitude)

# --- Solve with CN and Euler ---
results = {}
for name, axon in axons.items():
    solver_cn = CrankNicholson()
    t0 = time.perf_counter()
    res_cn = solver_cn.solve(axon, tsim=tsim, dt=dt)
    t1 = time.perf_counter()
    results[name] = {"CN": res_cn}
    benchmarks[name] = t1 - t0


res_mono = results["comp_axon"]["CN"]
res_comp = results["mono_axon"]["CN"]
Vm_mono = res_mono.Vm
Vm_comp = res_comp.Vm
extent = [0, tsim, 0, L]
# --- Relative error ---
eps = 1e-9
Vm_relerr = Vm_comp - Vm_mono

print(np.max(np.abs(Vm_relerr)))

print("Mono vs Comp close:", np.allclose(Vm_mono, Vm_comp))

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Mono ---
axs[0].imshow(
    Vm_mono.T,
    aspect='auto',
    extent=extent,
    origin='lower',
    cmap='viridis'
)
axs[0].set_title("Vm Mono")

# --- Composite ---
axs[1].imshow(
    Vm_comp.T,
    aspect='auto',
    extent=extent,
    origin='lower',
    cmap='viridis'
)
axs[1].set_title("Vm Composite")

# --- DIFFERENCE ---
im = axs[2].imshow(
    Vm_relerr.T,
    aspect='auto',
    extent=extent,
    origin='lower',
    cmap='bwr',          # diverging colormap
    vmin=-np.max(np.abs(Vm_relerr)),
    vmax=np.max(np.abs(Vm_relerr)),
)
axs[2].set_title("Difference (comp - mono)")

fig.colorbar(im, ax=axs[2], shrink=0.7)

plt.tight_layout()
plt.show()

#axs[-1, 0].set_xlabel('Time [ms]')
#axs[-1, 1].set_xlabel('Time [ms]')
#plt.show()
#exit()
# --- Print benchmark results ---
print("\n=== BENCHMARK Mono vs Comp ===")
for name, duration in benchmarks.items():
    print(f"{name:12s} : {duration:.4f} seconds")

# --- Positions to track ---
x_positions = [L/4, L/3, L/2, 2*L/3, 3*L/4]

# --- Create figure: stacked vertically ---
fig, ax = plt.subplots(figsize=(7, 6))

for i, (name, res_dict) in enumerate(results.items()):
    res_cn = res_dict["CN"]
    indices = [np.argmin(np.abs(res_cn.Vm.shape[1]*0 + res_cn.Vm.shape[1]*0 + np.linspace(0,L,Nx) - xp)) for xp in x_positions]

    # Left: Vm vs time at different positions
    for idx, xp in zip(indices, x_positions):
        if name == "mono_axon":
            ax.plot(res_cn.t, res_cn.Vm[:, idx],"--", label=f'x={xp:.1f}µm')
        else:
            ax.plot(res_cn.t, res_cn.Vm[:, idx], label=f'x={xp:.1f}µm')

ax.set_ylabel('Vm [mV]')
ax.set_xlabel('Time [ms]')
ax.legend(fontsize=8)
ax.grid(True)


fig.tight_layout()
##filename = save_dir + "/compare_three_axons_CN_vs_Euler.png"
#fig.savefig(filename)
plt.show()
#plt.close(fig)



