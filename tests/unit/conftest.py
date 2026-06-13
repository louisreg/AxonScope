"""
Shared fixtures for the unit test suite.

No NRV dependency. All tests here must run in < 30 s total on CPU.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from axonscope import AxonSimulation, degC, um
from axonscope.axons.unmyelinated import HodgkinHuxley, RattayAberham
from axonscope.solvers.crank_nicholson import CrankNicholson
from axonscope.stimulation import Stimulus


@pytest.fixture(scope="session")
def small_hh_axon():
    """Minimal HH axon with a central I-clamp."""
    axon = AxonSimulation(
        HodgkinHuxley(length=500.0 * um, diameter=0.5 * um, compartments=21, celsius=6.3 * degC)
    )
    axon.add_current_clamp(position_um=250.0,
        current=Stimulus.pulse(start=1.0, duration=1.0, amplitude=2.0),
    )
    return axon


@pytest.fixture(scope="session")
def small_hh_result(small_hh_axon):
    """Pre-computed CrankNicholson result for the small HH axon."""
    return CrankNicholson().solve(small_hh_axon, tsim=10.0, dt=0.01)
