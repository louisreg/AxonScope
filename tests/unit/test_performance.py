from __future__ import annotations

import pytest

import axonscope as axs


def _hh(compartments: int = 5):
    return axs.axons.HodgkinHuxley(
        length=40.0 * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )


def _clamped_instance(axon) -> axs.AxonInstance:
    instance = axs.AxonInstance(axon)
    instance.add_current_clamp(
        position=20.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=0.5 * axs.nA,
        ),
    )
    return instance


def test_simulation_estimate_counts_center_recording_memory():
    axon = _hh(compartments=5)
    simulation = axs.AxonSimulation(
        axs.AxonPopulation([_clamped_instance(axon), _clamped_instance(axon)]),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    estimate = simulation.estimate()

    assert isinstance(estimate, axs.SimulationEstimate)
    assert estimate.axon_count == 2
    assert estimate.step_count == 2
    assert estimate.max_compartments == 5
    assert estimate.recording_width_max == 1
    assert estimate.item("outputs.recorded_vm").shape == (2, 2, 1)
    assert estimate.item("inputs.intracellular_current_density").shape == (2, 2, 5)
    assert any("zero-field baseline" in text for text in estimate.recommendations)
    assert "outputs.recorded_vm" in estimate.format()


def test_observer_only_population_estimate_uses_sparse_current_clamp_inputs():
    axon = _hh(compartments=5)
    peak = axs.analysis.PeakVoltage(target=axs.positions.CENTER)
    simulation = axs.AxonSimulation(
        axs.AxonPopulation([_clamped_instance(axon), _clamped_instance(axon)]),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[peak],
    )

    estimate = simulation.estimate()

    assert estimate.metadata["intracellular_input_format"] == "sparse_current_clamp"
    assert estimate.item("inputs.intracellular_current_density_sparse").shape == (2, 2, 1)
    assert estimate.item("inputs.intracellular_current_indices").shape == (2, 1)
    assert estimate.item("outputs.recorded_vm").shape == (2, 2, 0)
    with pytest.raises(KeyError):
        estimate.item("inputs.intracellular_current_density")


def test_extracellular_estimate_surfaces_dense_vstim_and_factorized_footprint():
    axon = _hh(compartments=5)
    stimulus = axs.Stimulus.pulse(
        start=0.05 * axs.ms,
        duration=0.05 * axs.ms,
        amplitude=25.0 * axs.uA,
    )
    electrode = axs.PointSourceElectrode(
        x=20.0 * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[electrode],
        sigma=0.3 * axs.S_per_m,
    )
    instances = []
    for y_um in (-10.0, 10.0):
        instance = axs.AxonInstance(axon, y=y_um * axs.um)
        instance.add_extracellular_context(context=context)
        instances.append(instance)

    estimate = axs.estimate_simulation(
        axs.AxonPopulation(instances),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert estimate.metadata["context_count"] == 2
    assert estimate.metadata["electrode_rows"] == 2
    assert estimate.item("inputs.extracellular_potential_mid").shape == (2, 2, 5)
    assert estimate.item("footprints.factorized_rows").shape == (2, 5)
    assert any("Vstim[B,Nt,Nx]" in text for text in estimate.warnings)


def test_one_row_population_estimate_uses_pool_recording_width():
    axon = _hh(compartments=5)
    simulation = axs.AxonSimulation(
        axs.AxonPopulation([_clamped_instance(axon)]),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    estimate = simulation.estimate()

    assert simulation.is_population
    assert estimate.metadata["population_lifecycle"] is True
    assert estimate.recording_width_max == 1
    assert estimate.item("outputs.recorded_vm").shape == (1, 2, 1)


def test_runtime_device_and_precision_policy_are_typed_public_values():
    estimate = axs.estimate_simulation(
        _hh(compartments=3),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        runtime=axs.Runtime.JAX,
        device=axs.Device.gpu(0),
        precision=axs.PrecisionPolicy.mixed(
            state_dtype="float32",
            solver_dtype="float32",
            accumulation_dtype="float64",
        ),
    )

    assert estimate.runtime is axs.Runtime.JAX
    assert estimate.device == axs.Device.gpu(0)
    assert estimate.precision.accumulation_dtype == "float64"
    assert estimate.to_dict()["device"] == {"kind": "gpu", "index": 0}

    with pytest.raises(ValueError, match="Only GPU"):
        axs.Device("cpu", index=0)
