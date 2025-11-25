import pytest
import jax.numpy as jnp
import numpy as np

from axonscope.channel_models.hodgkin_huxley import HodgkinHuxleyICM, HHNaICM, HHKICM, HHLeakICM
from axonscope.channel_models.base_channel_model import CompositeICM
from axonscope.solvers import CrankNicholson
from axonscope.axons import GenericAxon
from axonscope.settings import dtype


def test_axon_composite_vs_mono_hodgkin_huxley():
    """
    End-to-end test:
    Run a full axon simulation with a CompositeICM(Na,K,L)
    and compare to the standard HodgkinHuxleyICM axon.

    Vm(t,x) must match within 0.1 mV absolute tolerance.
    """

    # -------------------------------------------------------
    # Build ICMs
    # -------------------------------------------------------
    mono_icm = HodgkinHuxleyICM()
    comp_icm = CompositeICM([HHNaICM(), HHKICM(), HHLeakICM()])

    # -------------------------------------------------------
    # Axon geometry
    # -------------------------------------------------------

    dt = 0.001
    tsim = 10
    Nx = 11
    L = 1_000
    d = 0.5

    ax_mono = GenericAxon(ion_channel=mono_icm,L=L, d=d, Nx=Nx, Temp=6.3)
    ax_comp = GenericAxon(ion_channel=comp_icm,L=L, d=d, Nx=Nx, Temp=6.3)

    # Conductivity etc are HH-like defaults
    solver = CrankNicholson()

    t_start = 1.0
    duration = 1.0
    amplitude = 5

    ax_mono.insert_I_Clamp(position=L/2, t_start=t_start, duration=duration, amplitude=amplitude)
    ax_comp.insert_I_Clamp(position=L/2, t_start=t_start, duration=duration, amplitude=amplitude)

    # -------------------------------------------------------
    # Run both simulations
    # -------------------------------------------------------
    res_mono = solver.solve(ax_mono, tsim, dt)
    res_comp = solver.solve(ax_comp, tsim, dt)

    Vm_mono = np.array(res_mono.Vm)
    Vm_comp = np.array(res_comp.Vm)

    assert Vm_mono.shape == Vm_comp.shape, \
        f"Vm shapes differ: {Vm_mono.shape} vs {Vm_comp.shape}"

    # -------------------------------------------------------
    # Check absolute difference
    # -------------------------------------------------------
    abs_err = np.abs(Vm_mono - Vm_comp)
    max_err = abs_err.max()

    if max_err > 0.001:
        rel_err = np.max(abs_err / (np.abs(Vm_mono) + 1e-9))
        idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
        raise AssertionError(
            f"Axon Vm differ by more than 0.01 mV\n"
            f"Max error = {max_err:.4f} mV at index {idx}\n"
            f"Relative error = {rel_err:.4e}\n"
        )

    # If OK:
    assert max_err <= 0.001
