import numpy as np
import pytest

import axonscope as axs
from axonscope.analytical import PointSourceElectrode
from axonscope.stimulation import Stimulus
from axonscope.stimulation import (
    IntracellularContext,
    IntracellularCurrentClamp,
)
from axonscope.runtime.jax.stimulation_runtime import compile_stimulus


# ==========================================================
# BASIC CONSTRUCTION
# ==========================================================

def test_constant_stimulus():
    stim = Stimulus.constant(5.0)

    assert np.isclose(stim.evaluate([0.0])[0], 5.0)
    assert np.isclose(stim.evaluate([10.0])[0], 5.0)


def test_pulse_stimulus():
    stim = Stimulus.pulse(
        start=1.0 * axs.ms,
        amplitude=3.0,
        duration=2.0 * axs.ms,
        baseline=0.0,
    )

    vals = stim.evaluate([0.5, 1.5, 3.5])

    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 3.0)
    assert np.isclose(vals[2], 0.0)


def test_biphasic_stimulus():
    stim = Stimulus.biphasic(
        start=1.0 * axs.ms,
        cathodic_amplitude=10.0,
        cathodic_duration=0.2 * axs.ms,
        interphase=0.1 * axs.ms,
    )

    vals = stim.evaluate([1.05, 1.25, 1.35])

    # first phase cathodic
    assert vals[0] < 0.0

    # interphase back to baseline
    assert np.isclose(vals[1], 0.0)

    # anodic second phase
    assert vals[2] > 0.0


def test_ramp_linear():
    stim = Stimulus.ramp(
        start=0.0 * axs.ms,
        duration=1.0 * axs.ms,
        start_value=0.0,
        stop_value=10.0,
        dt=0.1 * axs.ms,
    )

    vals = stim.evaluate([0.0, 0.5, 1.0])

    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 5.0, atol=1e-6)
    assert np.isclose(vals[2], 10.0)


def test_evaluate_accepts_pint_time_and_output_unit():
    stim = Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=1.0 * axs.ms,
        amplitude=2.0 * axs.nA,
    )

    vals_nA = stim.evaluate(np.asarray([0.5, 1.5, 2.5]) * axs.ms, unit=axs.nA)
    vals_A = stim.evaluate(np.asarray([1.5]) * axs.ms, unit=axs.A)

    assert stim.y_unit == "nanoampere"
    assert np.allclose(vals_nA, [0.0, 2.0, 0.0])
    assert np.allclose(vals_A, [2.0e-9])


def test_as_unit_reuses_already_canonical_stimulus():
    stim = Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=1.0 * axs.ms,
        amplitude=2.0e-6,
        unit="ampere",
    )

    assert stim.as_unit("ampere") is stim
    assert stim.as_unit(axs.A) is stim


def test_plot_accepts_unit_aware_time_grid():
    import matplotlib.pyplot as plt

    stim = Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=1.0 * axs.ms,
        amplitude=2.0 * axs.nA,
    )

    fig, ax = plt.subplots()
    try:
        returned = stim.plot(
            np.linspace(0.0, 3.0, 100) * axs.ms,
            ax=ax,
            amplitude_unit=axs.nA,
        )
        assert returned is ax
        assert len(ax.lines) == 1
        assert ax.get_xlabel() == "Time [ms]"
        assert ax.get_ylabel() == "Amplitude [nA]"
    finally:
        plt.close(fig)


# ==========================================================
# SORTING / DUPLICATES
# ==========================================================

def test_unsorted_times_are_sorted():
    stim = Stimulus(
        t=np.array([2.0, 0.0, 1.0]),
        y=np.array([2.0, 0.0, 1.0]),
    )

    assert np.allclose(stim.t, [0.0, 1.0, 2.0])
    assert np.allclose(stim.y, [0.0, 1.0, 2.0])


def test_duplicate_times_keep_last_value():
    stim = Stimulus(
        t=np.array([0.0, 1.0, 1.0, 2.0]),
        y=np.array([0.0, 5.0, 7.0, 0.0]),
    )

    val = stim.evaluate([1.0])[0]
    assert np.isclose(val, 7.0)


# ==========================================================
# ALGEBRA
# ==========================================================

def test_add_scalar():
    stim = Stimulus.constant(2.0)
    out = stim + 3.0

    assert np.isclose(out.evaluate([0])[0], 5.0)


def test_multiply_scalar():
    stim = Stimulus.constant(4.0)
    out = 0.5 * stim

    assert np.isclose(out.evaluate([0])[0], 2.0)


def test_add_two_stimuli():
    a = Stimulus.pulse(1.0 * axs.ms, 2.0, 1.0 * axs.ms)
    b = Stimulus.constant(1.0)

    c = a + b

    vals = c.evaluate([0.0, 1.5, 3.0])

    assert np.isclose(vals[0], 1.0)
    assert np.isclose(vals[1], 3.0)
    assert np.isclose(vals[2], 1.0)


def test_sub_two_stimuli():
    a = Stimulus.constant(5.0)
    b = Stimulus.constant(2.0)

    c = a - b

    assert np.isclose(c.evaluate([0])[0], 3.0)


# ==========================================================
# SHIFT / SCALE / OFFSET
# ==========================================================

def test_shift():
    stim = Stimulus.pulse(1.0 * axs.ms, 3.0, 1.0 * axs.ms)
    shifted = stim.shifted(2.0 * axs.ms)

    vals = shifted.evaluate([1.5, 3.5])

    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 3.0)


def test_scaled():
    stim = Stimulus.constant(2.0)
    out = stim.scaled(4.0)

    assert np.isclose(out.evaluate([0])[0], 8.0)


def test_offset():
    stim = Stimulus.constant(2.0)
    out = stim.offset(-1.0)

    assert np.isclose(out.evaluate([0])[0], 1.0)


# ==========================================================
# SYNCHRONIZATION
# ==========================================================

def test_synchronize():
    a = Stimulus.pulse(1.0 * axs.ms, 2.0, 1.0 * axs.ms)
    b = Stimulus.pulse(2.0 * axs.ms, 3.0, 1.0 * axs.ms)

    sa, sb = a.synchronize(b)

    assert np.allclose(sa.t, sb.t)
    assert len(sa.t) >= max(len(a.t), len(b.t))


# ==========================================================
# JAX BACKEND OBJECT
# ==========================================================

def test_compile_stimulus_callable():
    stim = Stimulus.pulse(1.0 * axs.ms, 5.0, 1.0 * axs.ms)
    jstim = compile_stimulus(stim)

    val = float(jstim(1.5))
    assert np.isclose(val, 5.0)


def test_physical_contexts_assign_canonical_current_units():
    stim = Stimulus.pulse(1.0 * axs.ms, 2.0, 1.0 * axs.ms)

    clamp = IntracellularCurrentClamp(position=100.0 * axs.um, current=stim)
    assert isinstance(clamp, IntracellularContext)
    assert clamp.position_um == 100.0
    assert clamp.current.y_unit == "nanoampere"
    assert np.allclose(clamp.current.y, stim.y)

    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um, stimulus=stim)
    drive = axs.analytical.point_source_drive(
        electrode,
        np.asarray([0.0]) * axs.um,
        sigma=0.3 * axs.S_per_m,
    )
    assert electrode.stimulus is not None
    assert electrode.stimulus.y_unit == "ampere"
    assert drive.stimulus.y_unit == "ampere"
    assert np.allclose(drive.stimulus.y, stim.y)


# ==========================================================
# INVALID INPUTS
# ==========================================================

def test_empty_stimulus_raises():
    with pytest.raises(ValueError):
        Stimulus([], [])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Stimulus([0, 1], [0])


def test_non_1d_raises():
    with pytest.raises(ValueError):
        Stimulus(np.zeros((2, 2)), np.zeros((2, 2)))


def test_stimulus_constructors_require_time_units():
    with pytest.raises(TypeError, match="start must include units compatible with time"):
        Stimulus.pulse(start=1.0, amplitude=1.0, duration=1.0 * axs.ms)

    with pytest.raises(TypeError, match="duration must include units compatible with time"):
        Stimulus.pulse(start=1.0 * axs.ms, amplitude=1.0, duration=1.0)

    with pytest.raises(TypeError, match="t must include units compatible with time"):
        Stimulus.from_samples([0.0, 1.0], [0.0, 1.0])

    stim = Stimulus.constant(1.0)
    with pytest.raises(TypeError, match="dt must include units compatible with time"):
        stim.shifted(1.0)
