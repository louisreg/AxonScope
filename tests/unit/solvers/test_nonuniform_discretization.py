"""Numerical coherence of uniform and non-uniform cable discretizations."""

import numpy as np

import axonfleet as axs
from axonfleet.analysis import ConductionVelocity
from axonfleet.axons.unmyelinated import RattayAberham
from axonfleet.stimulation import Stimulus


LENGTH_UM = 1000.0
COMPARTMENTS = 101
DIAMETER_UM = 1.0
ENA_MV = 50.0
DURATION_MS = 5.0
DT_MS = 0.001
PULSE_AMPLITUDE_NA = 5.0
PULSE_DURATION_MS = 1.0
PULSE_START_MS = 1.0
FOCUS_FACTOR = 1.0


def _focused_positions_um() -> np.ndarray:
    normalized = np.linspace(-1.0, 1.0, COMPARTMENTS)
    transformed = np.sinh(FOCUS_FACTOR * normalized) / np.sinh(FOCUS_FACTOR)
    return np.asarray((transformed + 1.0) * LENGTH_UM / 2.0, dtype=np.float32)


def _run(axon: RattayAberham):
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=(LENGTH_UM / 2.0) * axs.um,
        current=Stimulus.pulse(
            start=PULSE_START_MS * axs.ms,
            duration=PULSE_DURATION_MS * axs.ms,
            amplitude=PULSE_AMPLITUDE_NA,
        ),
    )
    return axs.AxonSimulation(
        instance,
        duration=DURATION_MS * axs.ms,
        dt=DT_MS * axs.ms,
    ).run().single


def _peak_metrics(result, position_um: float) -> tuple[float, float]:
    positions_um = np.asarray(result.axon.layout.position_values(unit=axs.um))
    index = int(np.argmin(np.abs(positions_um - position_um)))
    trace = np.asarray(result.Vm[:, index])
    peak_index = int(np.argmax(trace))
    return float(trace[peak_index]), float(np.asarray(result.t)[peak_index])


def test_uniform_and_nonuniform_discretizations_are_numerically_coherent():
    dx_um = LENGTH_UM / COMPARTMENTS
    uniform_positions_um = np.asarray(
        (np.arange(COMPARTMENTS, dtype=np.float64) + 0.5) * dx_um,
        dtype=np.float32,
    )
    uniform = RattayAberham(
        length=LENGTH_UM * axs.um,
        diameter=DIAMETER_UM * axs.um,
        compartments=COMPARTMENTS,
        ena=ENA_MV,
    )
    explicit_uniform = RattayAberham(
        x=uniform_positions_um * axs.um,
        diameter=DIAMETER_UM * axs.um,
        ena=ENA_MV,
    )
    focused = RattayAberham(
        x=_focused_positions_um() * axs.um,
        diameter=DIAMETER_UM * axs.um,
        ena=ENA_MV,
    )

    uniform_result = _run(uniform)
    explicit_result = _run(explicit_uniform)
    focused_result = _run(focused)

    np.testing.assert_allclose(
        explicit_result.Vm,
        uniform_result.Vm,
        rtol=0.0,
        atol=1e-8,
    )

    probe_positions_um = (
        LENGTH_UM / 4.0,
        LENGTH_UM / 3.0,
        2.0 * LENGTH_UM / 3.0,
        3.0 * LENGTH_UM / 4.0,
    )
    uniform_metrics = [_peak_metrics(uniform_result, x) for x in probe_positions_um]
    focused_metrics = [_peak_metrics(focused_result, x) for x in probe_positions_um]
    peak_difference_mV = max(
        abs(reference[0] - candidate[0])
        for reference, candidate in zip(uniform_metrics, focused_metrics)
    )
    arrival_difference_ms = max(
        abs(reference[1] - candidate[1])
        for reference, candidate in zip(uniform_metrics, focused_metrics)
    )
    velocity = ConductionVelocity()

    assert peak_difference_mV < 0.2
    assert arrival_difference_ms < 0.01
    np.testing.assert_allclose(
        velocity.detect(focused_result),
        velocity.detect(uniform_result),
        rtol=0.01,
    )
