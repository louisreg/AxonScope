#NOTE: THIS NEEDS PYTHON 3.12 -> nrv_env

import nrv
import settings as s
import matplotlib.pyplot as plt
import numpy as np
import settings as s
import solver_helper as sh
import utils as u
from axonscope.benchmark import minibench

def run(Nx):
    model = "Rattay_Aberham"  # Rattay_Aberham if not precised
    axon_u = nrv.unmyelinated(0, 0, s.d, s.L, model=model, Nsec=Nx)
    axon_u.insert_I_Clamp(0.5, s.t_start, s.duration, s.amplitude)
    results = axon_u(t_sim=s.tsim, dt = s.dt)

    return(results)

t_v = []

for Nx in s.Nx_v:
    res, t = minibench(run,Nx,n_iter=s.n_iter)
    t_v.append(t)

df = u.res_to_df(s.Nx_v, t_v, label = "nrv_neuron")

u.append_to_csv(df)

# --------------------------
# PLOT
# --------------------------
x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [np.argmin(np.abs(res["x"] - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(res["t"], res["V_mem"][idx], label=f'x={xp:.3f} cm')
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend()
plt.show()
