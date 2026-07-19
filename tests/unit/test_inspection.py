from __future__ import annotations

import io

import axonscope as axs


def _pool() -> axs.AxonPopulation:
    axon = axs.axons.HodgkinHuxley(
        length=40.0 * axs.um,
        diameter=0.9 * axs.um,
        compartments=5,
        celsius=6.3 * axs.degC,
    )
    return axs.AxonPopulation([axs.AxonInstance(axon), axs.AxonInstance(axon)])


def _clamped_pool() -> axs.AxonPopulation:
    axon = axs.axons.HodgkinHuxley(
        length=40.0 * axs.um,
        diameter=0.9 * axs.um,
        compartments=5,
        celsius=6.3 * axs.degC,
    )
    instances = []
    for _ in range(2):
        instance = axs.AxonInstance(axon)
        instance.add_current_clamp(
            position=20.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.05 * axs.ms,
                duration=0.05 * axs.ms,
                amplitude=0.5 * axs.nA,
            ),
        )
        instances.append(instance)
    return axs.AxonPopulation(instances)


def _clamped_instance() -> axs.AxonInstance:
    return _clamped_pool()[0]


def _inspect_simulation(axons, **kwargs):
    return axs.AxonSimulation(axons, **kwargs).inspect()


def test_inspect_simulation_prints_planning_dispatch_and_prepare():
    policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )

    report = _inspect_simulation(
        _pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        execution_policy=policy,
    )

    assert isinstance(report, axs.SimulationInspection)
    assert report.planning.step_count == 2
    assert report.dispatch_groups[0].will_batch is True
    assert report.preparations[0].x_positions_shape == (2, 5)
    assert report.membrane_sources[0].unique_membrane_count == 1
    assert report.membrane_sources[0].kinds == ("composite",)
    assert report.membrane_sources[0].source_count == 2
    assert set(report.membrane_sources[0].cache_statuses) <= {"hit", "miss"}
    assert report.membrane_sources[0].cache_reasons
    assert report.membrane_sources[0].cache_keys
    assert report.lowerings[0].intracellular_format == "zero_no_intracellular_context"
    assert report.lowerings[0].extracellular_format == "dense"
    assert report.lowerings[0].dense_vstim_shape == (2, 2, 5)
    assert report.kernels[0].kernel == "SingleCableVStimBatchKernel"
    assert report.kernels[0].solver.cable == "single_cable"
    assert report.kernels[0].solver.requested == "auto"
    assert report.kernels[0].solver.runtime_route == "jax_tridiagonal"
    assert report.result_assembly[0].record_kind == "dispatch row records"

    text = report.format()
    assert "planning:" in text
    assert "dispatch/batch:" in text
    assert "prepare:" in text
    assert "membranes:" in text
    assert "reasons=" in text
    assert "lowering:" in text
    assert "kernel:" in text
    assert "solver=jax_tridiagonal requested=auto" in text
    assert "result assembly:" in text
    assert "execution_policy=runtime=jax, device=cpu, precision=float32" in text

    buffer = io.StringIO()
    report.print(file=buffer)
    assert buffer.getvalue().startswith("AxonScope solver pipeline inspection")


def test_root_simulation_exposes_pipeline_inspection():
    simulation = axs.AxonSimulation(
        _pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
    )

    report = simulation.inspect()

    assert report.planning.axon_count == 2
    assert report.dispatch_groups[0].batch_kind == "strict-single-cable"


def test_inspection_reports_typed_double_cable_solver_policy():
    axon = axs.axons.MRG(diameter=5.7 * axs.um, nodes=3)
    population = axs.AxonPopulation([axs.AxonInstance(axon), axs.AxonInstance(axon)])

    report = _inspect_simulation(
        population,
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        execution_policy=axs.ExecutionPolicy(
            device=axs.Device.gpu(0),
            solvers=axs.SolverPolicy(
                double_cable=axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
                    block_b=64
                )
            ),
        ),
    )

    assert report.kernels[0].cable_mode == "double"
    assert report.kernels[0].solver.cable == "double_cable"
    assert report.kernels[0].solver.requested == "tiled_thomas"
    assert report.kernels[0].solver.runtime_route == "jax_triton_loop_xb"
    assert report.kernels[0].solver.internal is True
    assert report.kernels[0].solver.options == (("block_b", 64),)


def test_inspection_reports_observer_only_lowering_and_compact_results():
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = _inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    lowering = report.lowerings[0]
    assert lowering.intracellular_format == "sparse_current_clamp"
    assert lowering.extracellular_format == "zero_no_extracellular_stimulation"
    assert lowering.observer_format == "activation"
    assert lowering.retained_vm_width == 0
    assert lowering.materializes_dense_vstim is False

    padding = report.padding[0]
    assert padding.row_nx == (5, 5)
    assert padding.padded_compartments == 0

    probes = report.probes[0]
    assert probes.observer_names == ("activation",)
    assert probes.thresholds_mV == (-80.0,)
    assert probes.row_probe_counts == ((1,), (1,))
    assert probes.retained_shape == (2, 1)
    assert probes.retained_bytes == 2

    memory = report.memory[0]
    assert memory.observer_bytes == 2
    assert memory.retained_public_bytes == 2
    assert memory.retained_public_mib > 0.0

    assembly = report.result_assembly[0]
    assert assembly.record_kind == "compact dispatch cohort"
    assert assembly.vm_output == "none"
    assert assembly.observation_output == 'observations["activation"]'

    detail = report.assembly_details[0]
    assert detail.vm_shape is None
    assert detail.observation_shape == (2, 1)
    assert detail.observations_are_batched is True
    assert detail.public_rows == 1

    text = report.format()
    assert "padding:" in text
    assert "probes:" in text
    assert "memory:" in text
    assert "assembly details:" in text


def test_inspection_reports_compact_latency_first_crossing_storage():
    latency = axs.analysis.Latency(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = _inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[latency],
    )

    assert report.lowerings[0].observer_format == "first_crossing"
    assert report.probes[0].retained_shape == (2, 1)
    assert report.probes[0].retained_bytes == 8
    assert report.memory[0].observer_bytes == 8
    assert report.result_assembly[0].observation_output == 'observations["latency"]'


def test_inspection_reports_constant_memory_spike_summary_storage():
    spike_count = axs.analysis.SpikeCount(
        threshold=-20.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = _inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[spike_count],
    )

    assert report.lowerings[0].observer_format == "spike_summary"
    assert report.probes[0].retained_shape == (2, 1, 1, 4)
    assert report.probes[0].retained_bytes == 32
    assert report.memory[0].observer_bytes == 32
    assert report.result_assembly[0].observation_output == (
        'observations["spike_count"]'
    )


def test_inspection_reports_bounded_spike_event_and_downsampled_raster_storage():
    bounded = axs.analysis.SpikeCount(
        target=axs.positions.CENTER,
        max_spikes=3,
    )
    bounded_report = _inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[bounded],
    )

    assert bounded_report.lowerings[0].observer_format == "spike_events"
    assert bounded_report.probes[0].retained_shape == (2, 1, 1, 8)
    assert bounded_report.probes[0].retained_bytes == 64

    raster = axs.analysis.VmRaster(
        target=axs.positions.CENTER,
        every_n_steps=10,
    )
    raster_report = _inspect_simulation(
        _clamped_pool(),
        duration=10.0 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.none(),
        observers=[raster],
    )

    assert raster_report.lowerings[0].observer_format == "vm_raster"
    assert raster_report.probes[0].retained_shape == (2, 1, 1, 4)
    assert raster_report.probes[0].retained_bytes == 32


def test_inspection_reports_singleton_observer_only_batch_route():
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = _inspect_simulation(
        _clamped_instance(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    assert report.dispatch_groups[0].will_batch is True
    assert report.lowerings[0].route == "batch"
    assert report.kernels[0].route == "batch"
    assert report.result_assembly[0].record_kind == "compact dispatch cohort"
    assert report.assembly_details[0].public_rows == 1


def test_inspection_detailed_plot_covers_padding_memory_probes_and_results():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = _inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    axes = report.plot_details()

    assert len(axes) == 4
    assert [ax.get_title() for ax in axes] == [
        "padding",
        "memory estimate",
        "VmRaster probes",
        "result assembly",
    ]
    plt.close(axes[0].figure)


def test_inspection_marks_unsupported_observer_only_requests():
    peak = axs.analysis.PeakVoltage(target=axs.positions.CENTER)
    report = _inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[peak],
    )

    assert report.lowerings[0].observer_format == "unsupported_observer_only"
    assert report.result_assembly[0].observation_output == "unsupported_observer_only"
    assert report.probes[0].retained_shape is None
    assert report.assembly_details[0].observation_shape is None
