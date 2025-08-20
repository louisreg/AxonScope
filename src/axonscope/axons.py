import numpy as np
from abc import ABC, abstractmethod

from axonscope.math_functions import vtrap
from axonscope.benchmark import Benchmark

bench = Benchmark()

# -----------------------
# Abstract classes
# -----------------------
class Axon(ABC):
    def __init__(self, L, d, Nx=101, Cm=1.0, Ra=100.0, Vinit=-70.0):
        self.L = L                  
        self.Nx = Nx
        self.dx = L / (Nx - 1)      
        self.d = d                  
        self.a = d / 2              
        self.Cm = Cm                 # [µF/cm²]
        self.Ra = Ra                 # [Ω·cm]
        self.Vinit = Vinit           # [mV]

        self.a_cm = self.a * 1e-4
        self.L_cm = self.L * 1e-4
        self.dx_cm = self.dx * 1e-4
        self.x = np.linspace(0, L, Nx)

        # Derived quantities
        self.cm = 2.0 * np.pi * self.a_cm * Cm * 1e-6     # [F/cm]
        self.ra = Ra / (np.pi * self.a_cm**2)             # [Ω/cm]
        self.D = (1.0 / (self.ra * self.cm)) / 1000.0     # [cm²/ms]

        self.stim = None
        self.idx_inj = None
        self.t_start_inj = None
        self.t_stop_inj = None
        self.inj_uA_per_cm2 = None

    @abstractmethod
    def Iion(self, V):
        pass

    @abstractmethod
    def step_gates(self, dt_ms, V_mV):
        pass

    def insert_I_Clamp(self, position, t_start, duration, amplitude): 
        """amplitude in µA/cm² directly"""
        self.idx_inj = np.argmin(np.abs(self.x - position)) 
        self.inj_uA_per_cm2 = amplitude * 1e-3 / (2.0 * np.pi * self.a_cm * self.dx_cm)
        self.t_start_inj = t_start 
        self.t_stop_inj = t_start + duration 

        self.stim = True 
        
    def Iinj_uAcm2(self, t):
        """Return array of injected current density [µA/cm²]"""
        if self.stim:
            if self.t_start_inj <= t <= self.t_stop_inj:
                arr = np.zeros(self.Nx)
                arr[self.idx_inj] = self.inj_uA_per_cm2
                return arr
            else:
                return np.zeros(self.Nx)
        return np.zeros(self.Nx)

class Passive(Axon): 
    def __init__(self, L, d, Nx=101, Rm=1e4, Cm=1.0, Ra=100.0, EL=-70.0, Vinit=-70.0): 
        super().__init__(L=L, d=d, Nx=Nx, Cm=Cm, Ra=Ra, Vinit=Vinit)
         
        self.Rm = Rm                 # [Ω·cm²]
        self.EL = EL                 # [mV]

        # Derived quantities
        self.rm = Rm / (2 * np.pi * self.a_cm)            # [Ω·cm]
        self.k = (1.0 / (self.rm * self.cm)) / 1000.0     # [1/ms]

    @bench.benchmark(level=2)        
    def Iion(self, V):
        """
        Passive leak ionic current [µA/cm²]
        Ohm's law: I = g_leak * (V - E_L)
        g_leak = 1 / Rm   in S/cm²
        """
        g_leak = 1.0 / self.Rm  # [S/cm²]
        return g_leak * (V - self.EL) * 1e3  # mV→V and in µA/cm²
    
    @bench.benchmark(level=2)  
    def step_gates(self, dt_ms, V_mV):
        """
        not needed here
        """
        pass
    

class RattayAberham(Axon):
    """
    HH model 'RattayAberham' (from hh.mod).
    - V in mV
    - Iion(...) returns ionic current density in µA/cm²
    - Gating variables updated by explicit call to step_gates(dt_ms, V)
    """

    def __init__(
        self,
        L,
        d,
        Nx=101,
        Cm=1.0,
        Ra=100.0,
        Vinit=-70.0,
        gnabar=0.12,   # S/cm^2
        gkbar=0.036,   # S/cm^2
        gl=0.0003,     # S/cm^2
        el=-59.4,      # mV
        ena=45.0,      # mV
        ek=-82.0,      # mV
        celsius=37.0,  # degC
    ):
        super().__init__(L=L, d=d, Nx=Nx, Cm=Cm, Ra=Ra, Vinit=Vinit)

        # channel parameters
        self.gnabar = float(gnabar)
        self.gkbar = float(gkbar)
        self.gl = float(gl)
        self.el = float(el)
        self.ena = float(ena)
        self.ek = float(ek)
        self.celsius = float(celsius)

        # gating variables (per compartment)
        self.m = np.zeros(self.Nx, dtype=float)
        self.h = np.zeros(self.Nx, dtype=float)
        self.n = np.zeros(self.Nx, dtype=float)

        # initialize gating variables at Vinit equilibrium
        V0 = np.ones(self.Nx) * float(self.Vinit)
        minf, mtau, hinf, htau, ninf, ntau = self._rates(V0)
        self.m[:] = minf
        self.h[:] = hinf
        self.n[:] = ninf

    @bench.benchmark(level=2)  
    def _rates(self, V_mV):
        v = np.asarray(V_mV, dtype=float)
        q10 = np.power(2.24659524757, (self.celsius - 6.3) / 10.0)

        # m
        alpha_m = vtrap(2.5 - 0.1 * (v + 70.0), 1.0)
        beta_m = 4.0 * np.exp(-(v + 70.0) / 18.0)
        sum_m = np.maximum(alpha_m + beta_m, 1e-12)
        mtau = 1.0 / (q10 * sum_m)
        minf = alpha_m / sum_m

        # h
        alpha_h = 0.07 * np.exp(-(v + 70.0) / 20.0)
        beta_h = 1.0 / (np.exp(3.0 - 0.1 * (v + 70.0)) + 1.0)
        sum_h = np.maximum(alpha_h + beta_h, 1e-12)
        htau = 1.0 / (q10 * sum_h)
        hinf = alpha_h / sum_h

        # n
        alpha_n = 0.1 * vtrap(1.0 - 0.1 * (v + 70.0), 1.0)
        beta_n = 0.125 * np.exp(-(v + 70.0) / 80.0)
        sum_n = np.maximum(alpha_n + beta_n, 1e-12)
        ntau = 1.0 / (q10 * sum_n)
        ninf = alpha_n / sum_n

        return minf, mtau, hinf, htau, ninf, ntau

    # ---- gating integration (cnexp style) ----
    @bench.benchmark(level=2)  
    def step_gates(self, dt_ms, V_mV):
        """Advance gating variables with time step dt_ms (ms) and voltages V_mV (mV)."""
        if dt_ms <= 0.0:
            return

        V = np.asarray(V_mV, dtype=float)
        if V.shape != (self.Nx,):
            V = np.full(self.Nx, float(V.item()))

        minf, mtau, hinf, htau, ninf, ntau = self._rates(V)

        mtau = np.maximum(mtau, 1e-12)
        htau = np.maximum(htau, 1e-12)
        ntau = np.maximum(ntau, 1e-12)

        self.m = minf - (minf - self.m) * np.exp(-dt_ms / mtau)
        self.h = hinf - (hinf - self.h) * np.exp(-dt_ms / htau)
        self.n = ninf - (ninf - self.n) * np.exp(-dt_ms / ntau)

    # ---- ionic currents ----
    @bench.benchmark(level=2)  
    def Iion(self, V):
        """Return ionic current density [µA/cm²] at time t (ms) for V (mV)."""
        V_arr = np.asarray(V, dtype=float)
        if V_arr.shape != (self.Nx,):
            V_arr = np.full(self.Nx, float(V_arr.item()))

        gna = self.gnabar * (self.m ** 3) * self.h   # S/cm^2
        gk = self.gkbar * (self.n ** 4)              # S/cm^2
        gl = self.gl                                 # S/cm^2

        ina = gna * (V_arr - self.ena) * 1e3  # µA/cm^2
        ik = gk * (V_arr - self.ek) * 1e3
        il = gl * (V_arr - self.el) * 1e3

        return ina + ik + il


class HodgkinHuxley(Axon):
    """
    Hodgkin-Huxley squid model (from NEURON's hh.mod).
    - V in mV
    - Iion(t, V) returns ionic current density in µA/cm²
    - step_gates(dt_ms, V_mV) updates m,h,n using cnexp (exact) updates
    """

    def __init__(
        self,
        L,
        d,
        Nx=101,
        Cm=1.0,
        Ra=200.0,
        Vinit=-67.5,
        gnabar=0.12,   # S/cm^2
        gkbar=0.036,   # S/cm^2
        gl=0.0003,     # S/cm^2
        el=-54.3,      # mV
        ena=50.0,      # mV 
        ek=-77.0,      # mV
        celsius=6.3,   # degC (squid default)
    ):
        super().__init__(L=L, d=d, Nx=Nx, Cm=Cm, Ra=Ra, Vinit=Vinit)

        # channel parameters
        self.gnabar = float(gnabar)
        self.gkbar = float(gkbar)
        self.gl = float(gl)
        self.el = float(el)
        self.ena = float(ena)
        self.ek = float(ek)
        self.celsius = float(celsius)

        # gating variables
        self.m = np.zeros(self.Nx, dtype=float)
        self.h = np.zeros(self.Nx, dtype=float)
        self.n = np.zeros(self.Nx, dtype=float)

        # init gates at steady state for Vinit
        V0 = np.ones(self.Nx) * float(self.Vinit)
        minf, mtau, hinf, htau, ninf, ntau = self._rates(V0)
        self.m[:] = minf
        self.h[:] = hinf
        self.n[:] = ninf

    @bench.benchmark(level=2)  
    def _rates(self, V_mV):
        """
        Compute minf, mtau, hinf, htau, ninf, ntau for V (mV).
        Following hh.mod: q10 = 3^((celsius - 6.3)/10)
        mtau, htau, ntau returned in ms.
        """
        v = np.asarray(V_mV, dtype=float)
        q10 = np.power(3.0, (self.celsius - 6.3) / 10.0)

        # m
        alpha_m = 0.1 * vtrap(-(v + 40.0), 10.0)
        beta_m = 4.0 * np.exp(-(v + 65.0) / 18.0)
        sum_m = np.maximum(alpha_m + beta_m, 1e-12)
        mtau = 1.0 / (q10 * sum_m)
        minf = alpha_m / sum_m

        # h
        alpha_h = 0.07 * np.exp(-(v + 65.0) / 20.0)
        beta_h = 1.0 / (np.exp(-(v + 35.0) / 10.0) + 1.0)
        sum_h = np.maximum(alpha_h + beta_h, 1e-12)
        htau = 1.0 / (q10 * sum_h)
        hinf = alpha_h / sum_h

        # n
        alpha_n = 0.01 * vtrap(-(v + 55.0), 10.0)
        beta_n = 0.125 * np.exp(-(v + 65.0) / 80.0)
        sum_n = np.maximum(alpha_n + beta_n, 1e-12)
        ntau = 1.0 / (q10 * sum_n)
        ninf = alpha_n / sum_n

        return minf, mtau, hinf, htau, ninf, ntau

    # -------- gating update (cnexp) --------
    @bench.benchmark(level=2)  
    def step_gates(self, dt_ms, V_mV):
        """
        Advance gating variables with time step dt_ms (ms) and voltages V_mV (mV).
        Uses exact CNEXP update: x <- x_inf - (x_inf - x)*exp(-dt/tau)
        """
        if dt_ms <= 0.0:
            return

        V = np.asarray(V_mV, dtype=float)
        if V.shape != (self.Nx,):
            V = np.full(self.Nx, float(V.item()))

        minf, mtau, hinf, htau, ninf, ntau = self._rates(V)

        # clamp taus
        mtau = np.maximum(mtau, 1e-12)
        htau = np.maximum(htau, 1e-12)
        ntau = np.maximum(ntau, 1e-12)

        self.m = minf - (minf - self.m) * np.exp(-dt_ms / mtau)
        self.h = hinf - (hinf - self.h) * np.exp(-dt_ms / htau)
        self.n = ninf - (ninf - self.n) * np.exp(-dt_ms / ntau)

    # -------- ionic currents (no gate update) --------
    @bench.benchmark(level=2)  
    def Iion(self, V):
        """
        Return ionic current density [µA/cm²] for voltage array V (mV).
        t is kept for API compatibility but not used here.
        """
        V_arr = np.asarray(V, dtype=float)
        if V_arr.shape != (self.Nx,):
            V_arr = np.full(self.Nx, float(V_arr.item()))

        # conductances (S/cm^2)
        gna = self.gnabar * (self.m ** 3) * self.h
        gk = self.gkbar * (self.n ** 4)
        gl = self.gl

        # currents in µA/cm^2: I(µA/cm2) = g(S/cm2) * (V_mV - E_mV) * 1e3
        ina = gna * (V_arr - self.ena) * 1e3
        ik = gk * (V_arr - self.ek) * 1e3
        il = gl * (V_arr - self.el) * 1e3

        return ina + ik + il
