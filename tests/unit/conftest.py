"""
Shared fixtures for the unit test suite.

No NRV dependency. All tests here must run in < 30 s total on CPU.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from axonscope import AxonInstance, AxonSimulation, degC, ms, um
from axonscope.axons.unmyelinated import HodgkinHuxley, RattayAberham
from axonscope.stimulation import Stimulus


@pytest.fixture(scope="session")
def small_hh_axon():
    """Minimal HH axon with a central I-clamp."""
    axon = AxonInstance(
        HodgkinHuxley(length=500.0 * um, diameter=0.5 * um, compartments=21, celsius=6.3 * degC)
    )
    axon.add_current_clamp(position=250.0 * um,
        current=Stimulus.pulse(start=1.0 * ms, duration=1.0 * ms, amplitude=2.0),
    )
    return axon


@pytest.fixture(scope="session")
def small_hh_result(small_hh_axon):
    """Pre-computed public simulation result for the small HH axon."""
    return AxonSimulation(small_hh_axon, duration=10.0, dt=0.01).run().single
