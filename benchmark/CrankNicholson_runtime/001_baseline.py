from axonscope.axons import RattayAberham
from axonscope.solvers import CrankNicholson
from axonscope.benchmark import minibench
import numpy as np
import matplotlib.pyplot as plt

import utils as u
import settings as s

t_v = []

for Nx in s.Nx_v:
    axon = RattayAberham(L=s.L, d=s.d, Nx=Nx)
    axon.insert_I_Clamp(position=s.position, t_start=s.t_start, duration=s.duration, amplitude=s.amplitude)
    solver = CrankNicholson()
    res, t = minibench(solver.solve,axon, s.tsim, s.dt,n_iter=s.n_iter)
    t_v.append(t)

df = u.res_to_df(s.Nx_v, t_v, label = "baseline")

u.append_to_csv(df)


x_positions = [s.L/4, s.L/3, s.L/2, 2*s.L/3, 3*s.L/4]
indices = [np.argmin(np.abs(axon.x - xp)) for xp in x_positions]

fig, ax = plt.subplots(1, figsize=(5,5))
for idx, xp in zip(indices, x_positions):
    ax.plot(res.t, res.Vm[:, idx], label=f'x = {xp:.1f} µm')

plt.show()