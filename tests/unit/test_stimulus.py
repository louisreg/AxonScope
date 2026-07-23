import numpy as np
import pytest

import axonfleet as axs
from axonfleet.analytical import PointSourceElectrode
from axonfleet.stimulation import Stimulus
from axonfleet.stimulation import IntracellularCurrentClamp


# ==========================================================
# BASIC CONSTRUCTION
# ==========================================================

def test_constant_stimulus():
    stim = Stimulus.constant(5.0)

    assert np.isclose(stim.evaluate([0.0])[0], 5.0)
    assert np.isclose(stim.evaluate([10.0])[0], 5.0)


def test_stimulus_sample_buffers_are_read_only():
    stim = Stimulus.constant(5.0)

    assert not stim.t.flags.writeable
    assert not stim.y.flags.writeable
    with pytest.raises(ValueError):
        stim.y[0] = 1.0


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


def test_zero_balanced_biphasic_keeps_declared_phase_timing():
    zero = Stimulus.biphasic(
        start=1.0 * axs.ms,
        cathodic_amplitude=0.0,
        cathodic_duration=0.2 * axs.ms,
        interphase=0.1 * axs.ms,
    )
    driven = Stimulus.biphasic(
        start=1.0 * axs.ms,
        cathodic_amplitude=10.0,
        cathodic_duration=0.2 * axs.ms,
        interphase=0.1 * axs.ms,
    )

    np.testing.assert_allclose(zero.t, driven.t)
    np.testing.assert_allclose(zero.y, 0.0)
    assert zero._scale_shape == driven._scale_shape


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


def test_add_stimuli_on_the_union_of_sample_times():
    pulse = Stimulus.pulse(1.0 * axs.ms, 2.0 * axs.nA, 1.0 * axs.ms)
    baseline = Stimulus.constant(1.0 * axs.nA)

    combined = pulse + baseline

    np.testing.assert_allclose(combined.evaluate([0.0, 1.5, 3.0]), [1.0, 3.0, 1.0])
    assert combined.y_unit == "nanoampere"


def test_ramp_uses_linear_interpolation():
    ramp = Stimulus.ramp(
        start=0.0 * axs.ms,
        duration=1.0 * axs.ms,
        start_value=0.0,
        stop_value=10.0,
        dt=0.1 * axs.ms,
    )

    np.testing.assert_allclose(ramp.evaluate([0.0, 0.5, 1.0]), [0.0, 5.0, 10.0])


def test_stimulus_algebra_and_transformations():
    pulse = Stimulus.pulse(1.0 * axs.ms, 3.0, 1.0 * axs.ms)
    shifted = pulse.shifted(2.0 * axs.ms)
    transformed = 2.0 * shifted.scaled(0.5).offset(1.0) - 1.0

    np.testing.assert_allclose(transformed.evaluate([1.5, 3.5]), [1.0, 4.0])

    product = Stimulus.constant(2.0) * Stimulus.constant(4.0)
    assert product.evaluate(0.0) == 8.0


def test_synchronize_and_insert_samples_align_grids():
    first = Stimulus.pulse(1.0 * axs.ms, 2.0, 1.0 * axs.ms)
    second = Stimulus.pulse(2.0 * axs.ms, 3.0, 1.0 * axs.ms)

    left, right = first.synchronize(second)
    inserted = first.insert_samples(np.asarray([0.5, 1.5]) * axs.ms)

    np.testing.assert_array_equal(left.t, right.t)
    assert 0.5 in inserted.t
    assert 1.5 in inserted.t


def test_physical_contexts_assign_canonical_current_units():
    stim = Stimulus.pulse(1.0 * axs.ms, 2.0, 1.0 * axs.ms)

    clamp = IntracellularCurrentClamp(position=100.0 * axs.um, current=stim)
    assert isinstance(clamp, IntracellularCurrentClamp)
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

    with pytest.raises(TypeError, match="dt must include units compatible with time"):
        Stimulus.constant(1.0).shifted(1.0)
