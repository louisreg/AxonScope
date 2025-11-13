import numpy as np

Nx_v = [11,21,51,101,201,501,1001,2001,5001]
Nx_v = [11,21,51,101,201,501,1001]
#Nx_v = [11,21,51,101,201,501]
n_iter = 1

# --- axon parameters
L = 1000    #in µm
d = 0.5       # diameter in µm


#Inject current 1ms pulse 
position=L / 2
t_start = 1.0
duration = 1.0
amplitude = 2

# --- solver setup
tsim = 10.0         # total simulation time [ms]
dt = 0.001          # time step [ms]


Vinit = -70.0



Cm=1.0
Ra=100.0
a = d / 2
a_cm = a * 1e-4
cm = 2.0 * np.pi * a_cm * Cm * 1e-6     # [F/cm]
ra = Ra / (np.pi * a_cm**2)   
D = (1.0 / (ra * cm)) / 1000.0  

t_start_inj = t_start 
t_stop_inj = t_start + duration 
