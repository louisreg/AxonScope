import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["JAX_DISABLE_TF_COMPAT"] = "1"

import jax
import time
from axonscope.solvers import CrankNicholson
from axonscope.axons import RattayAberham

# --- axon parameters
L = 1000    #in µm
d = 0.5       # diameter in µm
Nx = 1001
axon = RattayAberham(L=L, d=d, Nx=Nx, celsius=37)

#Inject current 1ms pulse 
t_start = 1.0
duration = 1.0
amplitude = 2
axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

# --- solver setup
solver = CrankNicholson()
tsim = 10.0         # total simulation time [ms]
dt = 0.001          # time step [ms]

solver.solve(axon, tsim=1.0, dt=0.01)  # warm-up

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    t0 = time.perf_counter()
    res = solver.solve(axon, tsim, dt)
    res.Vm.block_until_ready()  
    t1 = time.perf_counter()

    jax.profiler.stop_trace()
    print(f"Execution time: {t1 - t0:.4f} s")
    print("Trace saved to /tmp/jax-trace")