# tests/test_stimulus.py

import numpy as np
import pytest

from axonscope.stimulus import Stimulus
from axonscope.solvers.stimulus_runtime import compile_stimulus
from axonscope.stimulus_eval import evaluate_stimulus_numpy


# ==========================================================
# BASIC CONSTRUCTION
# ==========================================================

def test_constant_stimulus():
    stim = Stimulus.constant(5.0)

    assert np.isclose(evaluate_stimulus_numpy(stim, [0.0])[0], 5.0)
    assert np.isclose(evaluate_stimulus_numpy(stim, [10.0])[0], 5.0)


def test_pulse_stimulus():
    stim = Stimulus.pulse(
        start=1.0,
        amplitude=3.0,
        duration=2.0,
        baseline=0.0,
    )

    vals = evaluate_stimulus_numpy(stim, [0.5, 1.5, 3.5])

    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 3.0)
    assert np.isclose(vals[2], 0.0)


def test_biphasic_stimulus():
    stim = Stimulus.biphasic(
        start=1.0,
        cathodic_amplitude=10.0,
        cathodic_duration=0.2,
        interphase=0.1,
    )

    vals = evaluate_stimulus_numpy(stim, [1.05, 1.25, 1.35])

    # first phase cathodic
    assert vals[0] < 0.0

    # interphase back to baseline
    assert np.isclose(vals[1], 0.0)

    # anodic second phase
    assert vals[2] > 0.0


def test_ramp_linear():
    stim = Stimulus.ramp(
        start=0.0,
        duration=1.0,
        start_value=0.0,
        stop_value=10.0,
        dt=0.1,
    )

    vals = evaluate_stimulus_numpy(stim, [0.0, 0.5, 1.0])

    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 5.0, atol=1e-6)
    assert np.isclose(vals[2], 10.0)


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

    val = evaluate_stimulus_numpy(stim, [1.0])[0]
    assert np.isclose(val, 7.0)


# ==========================================================
# ALGEBRA
# ==========================================================

def test_add_scalar():
    stim = Stimulus.constant(2.0)
    out = stim + 3.0

    assert np.isclose(evaluate_stimulus_numpy(out, [0])[0], 5.0)


def test_multiply_scalar():
    stim = Stimulus.constant(4.0)
    out = 0.5 * stim

    assert np.isclose(evaluate_stimulus_numpy(out, [0])[0], 2.0)


def test_add_two_stimuli():
    a = Stimulus.pulse(1.0, 2.0, 1.0)
    b = Stimulus.constant(1.0)

    c = a + b

    vals = evaluate_stimulus_numpy(c, [0.0, 1.5, 3.0])

    assert np.isclose(vals[0], 1.0)
    assert np.isclose(vals[1], 3.0)
    assert np.isclose(vals[2], 1.0)


def test_sub_two_stimuli():
    a = Stimulus.constant(5.0)
    b = Stimulus.constant(2.0)

    c = a - b

    assert np.isclose(evaluate_stimulus_numpy(c, [0])[0], 3.0)


# ==========================================================
# SHIFT / SCALE / OFFSET
# ==========================================================

def test_shift():
    stim = Stimulus.pulse(1.0, 3.0, 1.0)
    shifted = stim.shifted(2.0)

    vals = evaluate_stimulus_numpy(shifted, [1.5, 3.5])

    assert np.isclose(vals[0], 0.0)
    assert np.isclose(vals[1], 3.0)


def test_scaled():
    stim = Stimulus.constant(2.0)
    out = stim.scaled(4.0)

    assert np.isclose(evaluate_stimulus_numpy(out, [0])[0], 8.0)


def test_offset():
    stim = Stimulus.constant(2.0)
    out = stim.offset(-1.0)

    assert np.isclose(evaluate_stimulus_numpy(out, [0])[0], 1.0)


# ==========================================================
# SYNCHRONIZATION
# ==========================================================

def test_synchronize():
    a = Stimulus.pulse(1.0, 2.0, 1.0)
    b = Stimulus.pulse(2.0, 3.0, 1.0)

    sa, sb = a.synchronize(b)

    assert np.allclose(sa.t, sb.t)
    assert len(sa.t) >= max(len(a.t), len(b.t))


# ==========================================================
# JAX BACKEND OBJECT
# ==========================================================

def test_compile_stimulus_callable():
    stim = Stimulus.pulse(1.0, 5.0, 1.0)
    jstim = compile_stimulus(stim)

    val = float(jstim(1.5))
    assert np.isclose(val, 5.0)


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
