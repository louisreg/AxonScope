
from __future__ import annotations
from abc import ABC, abstractmethod
from axonscope.axons.base import AxonBase
from axonscope.simresult import SimResult

# -----------------------------------------------------------------------------
# Abstract solver base class
# -----------------------------------------------------------------------------
class Solver(ABC):
    """
    Abstract base class for temporal solvers of the cable equation.

    Concrete solver classes must implement `solve(axon, tsim, dt)` and return a
    :class:`SimResult` object containing the voltage traces and time vector.

    The cable equation solved is (per compartment):
        dV/dt = D * d^2V/dx^2 + (I_inj(t) - I_ion(V, gates)) / C_m

    where:
    - V is membrane voltage in mV
    - C_m is membrane capacitance (µF/cm²)
    - D = a / (2 * Ra * C_m) is the axial diffusion coefficient [cm²/ms]
      stored on the axon description
    - I_ion is ionic current density (µA/cm²)
    - I_inj is external injected current density (µA/cm²)
    """

    @abstractmethod
    def solve(self, axon: AxonBase, tsim: float, dt: float) -> SimResult:
        """
        Run simulation for a given axon.

        Parameters
        ----------
        axon : AxonBase
            Axon object providing geometry, ion channel model and stimulus.
        tsim : float
            Total simulation time (ms).
        dt : float
            Time step (ms).

        Returns
        -------
        SimResult
            Simulation result containing V_all (Nt × Nx) and t_vec (Nt).
        """
        raise NotImplementedError
