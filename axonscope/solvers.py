import numpy as np 
from abc import ABC, abstractmethod
from axonscope.axons import Axon

class Solver(ABC):
    @abstractmethod
    def solve(self, axon: Axon, tsim, dt=None):
        pass


class Euler(Solver): 
    def __init__(self): 
        pass 
    
    def solve(self, axon, tsim, dt): 

        if dt is None:  # stability margin
            dt_diff = (axon.ra * axon.cm * axon.dx_cm**2) / 2.0 
            dt_leak = 2.0 * axon.rm * axon.cm 
            dt = 0.4 * min(dt_diff, dt_leak)
            dt *= 1e3  # in ms
        Nt = int(np.ceil(tsim / dt)) 

        V = np.ones(axon.Nx) * axon.Vinit  # [mV]
        V_all = np.zeros((Nt, axon.Nx)) 
        t_vec = np.zeros(Nt) 
        t = 0.0 
        for n in range(Nt): 
            t_vec[n] = t 
            V = self.euler_step(axon, V, dt, t) 
            t += dt 
            V_all[n, :] = V 
        return V_all, t_vec 
    
    def euler_step(self, axon, V, dt, t): 
        # second derivative in space
        d2vdx2 = np.zeros_like(V) 
        d2vdx2[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / axon.dx_cm**2 

        # total membrane current per unit area [µA/cm²]
        Idiff = axon.D * d2vdx2 * axon.Cm      # from diffusion term
        axon.step_gates(dt, V)
        Iion = axon.Iion(V=V)          # ionic current
        Iinj_uAcm2 = axon.Iinj_uAcm2(t)
        dVdt = (Idiff - Iion + Iinj_uAcm2) / axon.Cm  # [mV/ms]
        
        V_new = V + dt * dVdt

        # boundary conditions
        V_new[0] = axon.Vinit
        V_new[-1] = axon.Vinit 
        return V_new

