# tests/test_electrodes.py

import numpy as np
import pytest

from axonscope.stimulus import Stimulus
from axonscope.electrodes import PointSourceElectrode
from axonscope.solvers.stimulus_runtime import (
    CompiledExtracellularContext,
    compile_extracellular_context,
)
from axonscope.stimulation import ExtracellularContext
from axonscope.stimulus_eval import evaluate_extracellular_context_numpy


# =============================================================================
# PointSourceElectrode
# =============================================================================

def test_point_source_footprint_matches_analytical_formula():
    x = np.array([0.0, 1.0e-3, 2.0e-3])

    electrode = PointSourceElectrode(
        x0_m=1.0e-3,
        y0_m=0.0,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    fp = electrode.footprint(x)

    r = np.sqrt((x - 1.0e-3) ** 2 + (1.0e-3) ** 2)
    expected = 1.0 / (4.0 * np.pi * 0.3 * r)

    assert np.allclose(fp, expected)


def test_point_source_footprint_is_symmetric_around_electrode():
    x = np.array([-1.0e-3, 0.0, 1.0e-3])

    electrode = PointSourceElectrode(
        x0_m=0.0,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    fp = electrode.footprint(x)

    assert np.isclose(fp[0], fp[2])
    assert fp[1] > fp[0]


def test_point_source_min_distance_avoids_singularity():
    x = np.array([0.0])

    electrode = PointSourceElectrode(
        x0_m=0.0,
        y0_m=0.0,
        z0_m=0.0,
        sigma_S_m=0.3,
        min_distance_m=1.0e-6,
    )

    fp = electrode.footprint(x)

    expected = 1.0 / (4.0 * np.pi * 0.3 * 1.0e-6)

    assert np.isfinite(fp[0])
    assert np.isclose(fp[0], expected)


# =============================================================================
# attach_stimulus
# =============================================================================

def test_attach_stimulus_returns_extracellular_stimulus():
    electrode = PointSourceElectrode(
        x0_m=0.0,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    stim = Stimulus.pulse(
        start=1.0,
        amplitude=1.0e-6,
        duration=1.0,
    )

    extra = electrode.attach_stimulus(stim)

    assert isinstance(extra, ExtracellularContext)
    assert extra.electrode is electrode
    assert extra.stimulus is stim


# =============================================================================
# Extracellular context NumPy evaluation
# =============================================================================

def test_extracellular_stimulus_evaluate_numpy_shape():
    x = np.linspace(0.0, 1.0e-3, 5)
    t = np.linspace(0.0, 3.0, 7)

    electrode = PointSourceElectrode(
        x0_m=0.5e-3,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    stim = Stimulus.pulse(
        start=1.0,
        amplitude=2.0e-6,
        duration=1.0,
    )

    extra = electrode.attach_stimulus(stim)

    Vext = evaluate_extracellular_context_numpy(extra, x, t)

    assert Vext.shape == (len(t), len(x))


def test_extracellular_stimulus_evaluate_numpy_values():
    x = np.array([0.0, 1.0e-3])
    t = np.array([0.5, 1.5, 2.5])

    electrode = PointSourceElectrode(
        x0_m=0.0,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    stim = Stimulus.pulse(
        start=1.0,
        amplitude=2.0e-6,
        duration=1.0,
    )

    extra = electrode.attach_stimulus(stim)

    Vext = evaluate_extracellular_context_numpy(extra, x, t)

    fp = electrode.footprint(x)

    assert np.allclose(Vext[0], 0.0)
    assert np.allclose(Vext[1], 2.0e-6 * fp)
    assert np.allclose(Vext[2], 0.0)


# =============================================================================
# Compile to JAX
# =============================================================================

def test_compile_returns_jax_ready_object():
    x = np.linspace(0.0, 1.0e-3, 5)

    electrode = PointSourceElectrode(
        x0_m=0.5e-3,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    stim = Stimulus.pulse(
        start=1.0,
        amplitude=1.0e-6,
        duration=1.0,
    )

    extra = electrode.attach_stimulus(stim)
    compiled = compile_extracellular_context(extra, x)

    assert isinstance(compiled, CompiledExtracellularContext)
    assert compiled.footprint_V_per_A.shape == (len(x),)


def test_compiled_extracellular_stimulus_matches_numpy():
    x = np.linspace(0.0, 1.0e-3, 5)

    electrode = PointSourceElectrode(
        x0_m=0.5e-3,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    stim = Stimulus.pulse(
        start=1.0,
        amplitude=2.0e-6,
        duration=1.0,
    )

    extra = electrode.attach_stimulus(stim)
    compiled = compile_extracellular_context(extra, x)

    expected = evaluate_extracellular_context_numpy(extra, x, np.array([1.5]))[0]
    got = np.asarray(compiled(1.5))

    assert np.allclose(got, expected)


# =============================================================================
# Units sanity check
# =============================================================================

def test_point_source_units_scale_linearly_with_current():
    x = np.array([1.0e-3])

    electrode = PointSourceElectrode(
        x0_m=0.0,
        z0_m=1.0e-3,
        sigma_S_m=0.3,
    )

    stim_1 = Stimulus.constant(1.0e-6)
    stim_2 = Stimulus.constant(2.0e-6)

    extra_1 = electrode.attach_stimulus(stim_1)
    extra_2 = electrode.attach_stimulus(stim_2)

    V1 = evaluate_extracellular_context_numpy(extra_1, x, np.array([0.0]))[0, 0]
    V2 = evaluate_extracellular_context_numpy(extra_2, x, np.array([0.0]))[0, 0]

    assert np.isclose(V2, 2.0 * V1)
