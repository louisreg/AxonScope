"""
Shared fixtures for the unit test suite.

No NRV dependency. All tests here must run in < 30 s total on CPU.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from axonscope.axons.unmyelinated import HodgkinHuxley, RattayAberham
from axonscope.solvers.CrankNicholson import CrankNicholson


@pytest.fixture(scope="session")
def small_hh_axon():
    """Minimal HH axon (L=500 µm, Nx=21) with a central I-clamp."""
    axon = HodgkinHuxley(L=500.0, d=0.5, Nx=21, celsius=6.3)
    axon.insert_I_Clamp(position=250.0, t_start=1.0, duration=1.0, amplitude=2.0)
    return axon


@pytest.fixture(scope="session")
def small_hh_result(small_hh_axon):
    """Pre-computed CrankNicholson result for the small HH axon."""
    return CrankNicholson().solve(small_hh_axon, tsim=10.0, dt=0.01)
