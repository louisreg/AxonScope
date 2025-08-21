import numpy as np 
from abc import ABC, abstractmethod
from numpy.typing import NDArray

from axonscope.axons import Axon
from axonscope.simresult import SimResult
from axonscope.benchmark import Benchmark

bench = Benchmark()

class Solver(ABC):
    @abstractmethod
    def solve(self, axon: Axon, tsim, dt) -> SimResult:
        pass


class Euler(Solver): 
    def __init__(self): 
        pass 
    
    @bench.benchmark(level=1)  
    def solve(self, axon: Axon, tsim, dt) -> SimResult: 

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
        return SimResult(axon, V_all, t_vec)
    
    @bench.benchmark(level=2)  
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


class CrankNicholson(Solver):
    """
    Unoptimised implementation of the Hines (1984) scheme.
    """

    def __init__(self):
        pass

    @bench.benchmark(level=1)
    def solve(self, axon:Axon, tsim, dt) -> SimResult:
        Nx = axon.Nx
        Nt = int(np.ceil(tsim / dt))

        V = np.ones(Nx) * axon.Vinit
        V_all = np.zeros((Nt, Nx))
        t_vec = np.zeros(Nt)

        dx2 = axon.dx_cm ** 2
        alpha = axon.D * (dt / 2.0) / dx2  # diffusion coefficient

        # Construct the tridiagonal matrix A (time-independent)
        A = np.zeros((Nx, Nx))
        for i in range(1, Nx-1):
            A[i, i-1] = -alpha
            A[i, i]   = 1.0 + 2.0*alpha
            A[i, i+1] = -alpha
        A[0, 0]   = 1.0
        A[-1, -1] = 1.0

        for n in range(Nt):
            t_mid = n*dt + dt/2.0
            t_vec[n] = n*dt

            # -----------------------------
            # Hines equation:
            # (A) V^{1/2} = V^n + (dt/2Cm)[ I_inj(t+dt/2) - I_HH(V^{1/2}, t+dt/2) ]
            # -----------------------------

            # Right-hand side: V^n + (dt/2Cm) * (I_inj - I_HH)
            rhs = np.array(V, copy=True)
            Iinj = axon.Iinj_uAcm2(t_mid)  # [µA/cm²]
            
            Iion = axon.Iion(V)
            rhs += (dt / (2.0 * axon.Cm)) * (Iinj - Iion)

            axon.half_step_gates(dt, V)

            # Boundary conditions
            rhs[0]  = axon.Vinit
            rhs[-1] = axon.Vinit


            V_half = np.linalg.solve(A, rhs)

            # Explicit update (Hines: extrapolation)
            V_new = 2.0 * V_half - V
            V_new[0]  = axon.Vinit
            V_new[-1] = axon.Vinit

            V_all[n, :] = V_new
            V = V_new

        return SimResult(axon, V_all, t_vec)