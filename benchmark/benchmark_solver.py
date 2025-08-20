
from axonscope.axons import HodgkinHuxley
from axonscope.solvers import Euler
from axonscope.benchmark import Benchmark

L = 1_000        # µm
d = 5            # µm
Nx = 101
Cm = 10.0        # µF/cm^2
Gl = 1e-4        # S/cm^2
EL = -70.0       # mV
rho = 100.0      # ohm*cm
I_inj_nA = 10    # nA
t_start = 2      # ms
t_on = 1         # ms
Tsim = 10        # ms

# --- axon parameters
L = 1000    #in µm
d = 0.5       # diameter in µm
Nx = 101
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
res = solver.solve(axon, tsim=tsim, dt=dt)

