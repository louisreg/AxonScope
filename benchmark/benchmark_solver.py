
from axonscope.axons import HodgkinHuxley
from axonscope.solvers import Euler
from axonscope.benchmark import Benchmark
import timeit

# --- axon parameters
L = 1000    #in µm
d = 0.5       # diameter in µm
Nx = 51
axon = HodgkinHuxley(L=L, d=d, Nx=Nx, celsius=6.3)

#Inject current 1ms pulse 
t_start = 1.0
duration = 1.0
amplitude = 2
axon.insert_I_Clamp(position=L / 2, t_start=t_start, duration=duration, amplitude=amplitude)

# --- solver setup
solver = Euler()
tsim = 10.0         # total simulation time [ms]
dt = 0.001          # time step [ms]

bench = Benchmark()
bench.enable(level = 1)
solver.solve(axon, tsim=tsim, dt=dt)




