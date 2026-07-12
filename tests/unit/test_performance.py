from __future__ import annotations

import numpy as np
import pytest

import axonscope as axs
from axonscope.performance_views import format_simulation_estimate


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


def _run_simulation(axons, **kwargs):
    return axs.AxonSimulation(axons, **kwargs).run()


def _estimate_simulation(axons, **kwargs):
    simulation_keys = {
        "duration",
        "dt",
        "recording",
        "batch_options",
        "observers",
        "execution_policy",
    }
    simulation_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in simulation_keys}
    return axs.AxonSimulation(axons, **simulation_kwargs).estimate(**kwargs)


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
    assert len(estimate.groups) == 1
    assert estimate.groups[0].route == "batch"
    assert estimate.groups[0].retained_vm_width == 1
    assert estimate.item("outputs.recorded_vm").shape == (2, 2, 1)
    assert estimate.item("inputs.intracellular_current_density").shape == (2, 2, 5)
    assert any("zero-field baseline" in text for text in estimate.recommendations)
    assert "groups:" in estimate.format()
    assert "outputs.recorded_vm" in estimate.format()
    assert format_simulation_estimate(estimate) == estimate.format()
    assert estimate.rows(section="items")[0]["name"]
    assert estimate.rows(section="groups")[0]["group_id"] == 0
    assert "name" in estimate.to_dataframe(section="items").columns
    assert "group_id" in estimate.to_dataframe(section="groups").columns


def test_estimate_retained_vm_bytes_matches_small_run():
    axon = _hh(compartments=5)
    simulation = axs.AxonSimulation(
        [axs.AxonInstance(axon), axs.AxonInstance(axon)],
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    estimate = simulation.estimate()
    result = simulation.run()
    actual_vm_bytes = sum(np.asarray(row.Vm).nbytes for row in result)

    assert estimate.item("outputs.recorded_vm").bytes == actual_vm_bytes
    assert estimate.retained_bytes >= actual_vm_bytes


def test_observer_only_population_estimate_uses_sparse_current_clamp_inputs():
    axon = _hh(compartments=5)
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    simulation = axs.AxonSimulation(
        axs.AxonPopulation([_clamped_instance(axon), _clamped_instance(axon)]),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    estimate = simulation.estimate()

    assert estimate.metadata["intracellular_input_format"] == "sparse_current_clamp"
    assert estimate.item("inputs.intracellular_current_density_sparse").shape == (2, 2, 1)
    assert estimate.item("inputs.intracellular_current_indices").shape == (2, 1)
    assert estimate.item("outputs.recorded_vm").shape == (2, 2, 0)
    with pytest.raises(KeyError):
        estimate.item("inputs.intracellular_current_density")


def test_extracellular_estimate_surfaces_factorized_footprint_without_dense_vstim():
    axon = _hh(compartments=5)
    stimulus = axs.Stimulus.pulse(
        start=0.05 * axs.ms,
        duration=0.05 * axs.ms,
        amplitude=25.0 * axs.uA,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=20.0 * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )
    instances = []
    for y_um in (-10.0, 10.0):
        stimulation = axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            sigma=0.3 * axs.S_per_m,
            axon_y=y_um * axs.um,
            axon_z=0.0 * axs.um,
        )
        instance = axs.AxonInstance(axon)
        instance.add_extracellular_stimulation(stimulation=stimulation)
        instances.append(instance)

    estimate = _estimate_simulation(
        axs.AxonPopulation(instances),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert estimate.metadata["extracellular_stimulation_count"] == 2
    assert estimate.metadata["extracellular_drive_count"] == 2
    assert estimate.metadata["intracellular_input_format"] == "zero_no_intracellular_context"
    assert estimate.metadata["skipped_dense_iinj_shape"] == [2, 2, 5]
    assert estimate.metadata["skipped_dense_iinj_nbytes"] > 0
    assert estimate.groups[0].extracellular_format == "factorized_footprint"
    assert estimate.item("footprints.factorized_rows").shape == (2, 5)
    with pytest.raises(KeyError):
        estimate.item("inputs.extracellular_potential_mid")
    with pytest.raises(KeyError):
        estimate.item("inputs.intracellular_current_density")
    assert not any("Vstim[B,Nt,Nx]" in text for text in estimate.warnings)


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
    assert estimate.groups[0].route == "batch"


def test_single_full_recording_estimate_requires_future_batch_observable_route():
    axon = _hh(compartments=5)
    simulation = axs.AxonSimulation(
        axon,
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.full(),
    )

    with pytest.raises(NotImplementedError, match="Vm only"):
        simulation.estimate()


def test_runtime_device_and_precision_policy_are_typed_public_values():
    estimate = _estimate_simulation(
        _hh(compartments=3),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        runtime=axs.runtime.jax,
        device=axs.Device.gpu(0),
        precision=axs.PrecisionPolicy.mixed(
            state_dtype="float32",
            solver_dtype="float32",
            accumulation_dtype="float64",
        ),
    )

    assert estimate.runtime is axs.runtime.jax
    assert estimate.device == axs.Device.gpu(0)
    assert estimate.precision.accumulation_dtype == "float64"
    assert estimate.to_dict()["device"] == {"kind": "gpu", "index": 0}

    with pytest.raises(ValueError, match="Only GPU"):
        axs.Device("cpu", index=0)


def test_solver_policy_is_typed_public_execution_policy_state():
    policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.gpu(0),
        precision=axs.PrecisionPolicy.float32(),
        solvers=axs.SolverPolicy(
            single_cable=axs.runtime.jax.SingleCableSolver.jax_tridiagonal(),
            double_cable=axs.runtime.jax.gpu.DoubleCableSolver.pcr_soa(
                adaptive_threshold=2048
            ),
        ),
    )

    assert (
        policy.solver_policy.single_cable.kind
        is axs.runtime.jax.SingleCableSolverKind.JAX_TRIDIAGONAL
    )
    assert (
        policy.solver_policy.double_cable.kind
        is axs.runtime.jax.DoubleCableSolverKind.JAX_PCR_SOA
    )
    assert policy.solver_policy.double_cable.pcr_options.adaptive_threshold == 2048
    assert axs.ExecutionPolicy().solver_policy == axs.SolverPolicy()
    assert (
        axs.runtime.jax.cpu.DoubleCableSolver.thomas().kind
        is axs.runtime.jax.DoubleCableSolverKind.THOMAS
    )

    tiled = axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
        block_b=64,
        allow_fallback=True,
    )
    assert tiled.kind is axs.runtime.jax.DoubleCableSolverKind.TILED_THOMAS
    assert tiled.tiled_thomas_options.block_b == 64
    assert tiled.tiled_thomas_options.allow_fallback is True


def test_solver_policy_rejects_untyped_values():
    with pytest.raises(TypeError, match="SolverPolicy"):
        axs.ExecutionPolicy(solvers="gpu")
    with pytest.raises(TypeError, match="single-cable solver request"):
        axs.SolverPolicy(single_cable="jax_tridiagonal")
    with pytest.raises(TypeError, match="double-cable solver request"):
        axs.SolverPolicy(double_cable="pcr_soa")
    with pytest.raises(TypeError, match="SingleCableSolverKind"):
        axs.runtime.jax.SingleCableSolver(kind="jax_tridiagonal")
    with pytest.raises(TypeError, match="DoubleCableSolverKind"):
        axs.runtime.jax.DoubleCableSolver(kind="jax_pcr_soa")
    with pytest.raises(ValueError, match="adaptive_threshold"):
        axs.runtime.jax.gpu.DoubleCableSolver.pcr(adaptive_threshold=0)
    with pytest.raises(ValueError, match="block_b"):
        axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(block_b=0)


def test_jax_solver_engine_resolves_typed_solver_policy():
    from axonscope.runtime.jax.solver_engines import resolve_jax_solver_engine

    cpu_engine = resolve_jax_solver_engine(
        axs.ExecutionPolicy(
            device=axs.Device.cpu(),
            solvers=axs.SolverPolicy(
                double_cable=axs.runtime.jax.cpu.DoubleCableSolver.thomas()
            ),
        ),
        platform="cpu",
    )
    gpu_engine = resolve_jax_solver_engine(
        axs.ExecutionPolicy(
            device=axs.Device.gpu(0),
            solvers=axs.SolverPolicy(
                double_cable=axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
                    block_b=64
                )
            ),
        ),
        platform="gpu",
    )

    assert cpu_engine is not None
    assert cpu_engine.name == "jax_cpu_thomas"
    assert cpu_engine.single_cable_solver == "jax_tridiagonal"
    assert cpu_engine.double_cable_block_solver == "thomas"
    assert gpu_engine is not None
    assert gpu_engine.name == "jax_gpu_tiled_thomas"
    assert gpu_engine.single_cable_solver == "jax_tridiagonal"
    assert gpu_engine.double_cable_block_solver == "jax_triton_loop_xb"
    assert gpu_engine.allow_internal_double_cable_block_solver is True
    assert gpu_engine.tiled_thomas_block_b == 64


def test_runtime_solver_route_report_resolves_once_from_execution_policy():
    from axonscope.runtime.execution import solver_route_from_execution_policy

    route = solver_route_from_execution_policy(
        axs.ExecutionPolicy(
            device=axs.Device.gpu(0),
            solvers=axs.SolverPolicy(
                single_cable=axs.runtime.jax.gpu.SingleCableSolver.jax_tridiagonal(),
                double_cable=axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
                    block_b=64
                ),
            ),
        )
    )

    assert route is not None
    assert route.runtime == "jax"
    assert route.platform == "gpu"
    assert route.engine_name == "jax_gpu_tiled_thomas"
    assert route.single_cable is not None
    assert route.single_cable.cable == "single_cable"
    assert route.single_cable.requested == "jax_tridiagonal"
    assert route.single_cable.backend_route == "jax_tridiagonal"
    assert route.double_cable is not None
    assert route.double_cable.cable == "double_cable"
    assert route.double_cable.requested == "tiled_thomas"
    assert route.double_cable.backend_route == "jax_triton_loop_xb"
    assert route.double_cable.internal is True
    assert route.double_cable.options == (("block_b", 64),)
    assert route.for_cable("single") == route.single_cable
    assert route.for_cable("double") == route.double_cable


def test_jax_execution_policy_resolution_cache_reuses_resolved_device(monkeypatch):
    from axonscope.runtime.jax import execution_policy as jax_execution_policy

    calls = 0
    sentinel_device = object()

    def fake_resolve_device(device):
        nonlocal calls
        calls += 1
        return sentinel_device

    policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.gpu(0),
        precision=axs.PrecisionPolicy.float32(),
        solvers=axs.SolverPolicy(
            double_cable=axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(block_b=32)
        ),
    )

    jax_execution_policy.clear_jax_execution_policy_cache()
    monkeypatch.setattr(jax_execution_policy, "_resolve_device", fake_resolve_device)

    first = jax_execution_policy._resolve_jax_execution_policy(policy)
    second = jax_execution_policy._resolve_jax_execution_policy(policy)

    assert first is second
    assert first.device is sentinel_device
    assert first.platform == "gpu"
    assert first.solver_engine is not None
    assert first.solver_engine.name == "jax_gpu_tiled_thomas"
    assert calls == 1

    jax_execution_policy.clear_jax_execution_policy_cache()


def test_jax_precision_validation_cache_reuses_exact_instance_tuple(monkeypatch):
    from axonscope.runtime.jax import execution_policy as jax_execution_policy

    calls = 0
    instances = (_clamped_instance(_hh(compartments=5)),)
    precision = axs.PrecisionPolicy.float32()

    def fake_validate_precision_uncached(precision, *, instances):
        nonlocal calls
        calls += 1

    jax_execution_policy.clear_jax_precision_validation_cache()
    monkeypatch.setattr(
        jax_execution_policy,
        "_validate_precision_uncached",
        fake_validate_precision_uncached,
    )

    jax_execution_policy._validate_precision(precision, instances=instances)
    jax_execution_policy._validate_precision(precision, instances=instances)

    assert calls == 1

    jax_execution_policy.clear_jax_precision_validation_cache()


def test_execution_policy_runs_jax_cpu_float32_simulation():
    axon = _hh(compartments=5)

    run = _run_simulation(
        axon,
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        execution_policy=axs.ExecutionPolicy(
            runtime=axs.runtime.jax,
            device=axs.Device.cpu(),
            precision=axs.PrecisionPolicy.float32(),
        ),
    )
    result = run.single

    assert result.Vm.shape == (2, 5)


def test_execution_policy_rejects_unsupported_runtime_for_simulation():
    with pytest.raises(NotImplementedError, match="axs.runtime.numpy"):
        _run_simulation(
            _hh(compartments=5),
            duration=0.10 * axs.ms,
            dt=0.05 * axs.ms,
            execution_policy=axs.ExecutionPolicy(runtime=axs.runtime.numpy),
        )


def test_execution_policy_rejects_unavailable_or_mixed_precision_for_simulation():
    with pytest.raises((RuntimeError, NotImplementedError), match="float64|Mixed"):
        _run_simulation(
            _hh(compartments=5),
            duration=0.10 * axs.ms,
            dt=0.05 * axs.ms,
            execution_policy=axs.ExecutionPolicy(
                runtime=axs.runtime.jax,
                device=axs.Device.cpu(),
                precision=axs.PrecisionPolicy.mixed(
                    state_dtype="float32",
                    solver_dtype="float32",
                    accumulation_dtype="float64",
                ),
            ),
        )


def test_execution_policy_rejects_implicit_precision_casting():
    membrane = axs.membranes.Passive(
        Rm=1e4,
        EL=-70.0,
        dtype=np.float64,
    )
    section = axs.axons.Section(
        "axon",
        membrane=membrane,
        diameter=0.9 * axs.um,
    )
    axon = axs.axons.Axon(
        layout=axs.axons.Layout.single_uniform(
            section,
            length=40.0 * axs.um,
            compartments=5,
        ),
        formulation=axs.axons.CableFormulation.SINGLE_CABLE,
    )

    with pytest.raises(ValueError, match="does not cast"):
        _run_simulation(
            axon,
            duration=0.10 * axs.ms,
            dt=0.05 * axs.ms,
            execution_policy=axs.ExecutionPolicy(
                runtime=axs.runtime.jax,
                device=axs.Device.cpu(),
                precision=axs.PrecisionPolicy.float32(),
            ),
        )
