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


def test_inspect_simulation_prints_planning_dispatch_and_prepare():
    policy = axs.ExecutionPolicy(
        runtime=axs.Runtime.JAX,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )

    report = axs.inspect_simulation(
        _pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        execution_policy=policy,
    )

    assert isinstance(report, axs.SimulationInspection)
    assert report.planning.step_count == 2
    assert report.dispatch_groups[0].will_batch is True
    assert report.preparations[0].x_positions_shape == (2, 5)
    assert report.lowerings[0].intracellular_format == "zero_no_intracellular_context"
    assert report.lowerings[0].extracellular_format == "dense"
    assert report.lowerings[0].dense_vstim_shape == (2, 2, 5)
    assert report.kernels[0].kernel == "SingleCableVStimBatchKernel"
    assert report.result_assembly[0].record_kind == "DispatchResult rows"

    text = report.format()
    assert "planning:" in text
    assert "dispatch/batch:" in text
    assert "prepare:" in text
    assert "lowering:" in text
    assert "kernel:" in text
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


def test_inspection_reports_observer_only_lowering_and_compact_results():
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = axs.inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[activation],
    )

    lowering = report.lowerings[0]
    assert lowering.intracellular_format == "sparse_current_clamp"
    assert lowering.extracellular_format == "zero_no_context"
    assert lowering.observer_format == "vm_raster"
    assert lowering.retained_vm_width == 0
    assert lowering.materializes_dense_vstim is False

    padding = report.padding[0]
    assert padding.row_nx == (5, 5)
    assert padding.padded_compartments == 0

    probes = report.probes[0]
    assert probes.observer_names == ("activation",)
    assert probes.thresholds_mV == (-80.0,)
    assert probes.row_probe_counts == ((1,), (1,))
    assert probes.packed_shape == (2, 1, 1, 1)
    assert probes.packed_bytes == 8

    memory = report.memory[0]
    assert memory.vm_raster_bytes == 8
    assert memory.retained_public_bytes == 8
    assert memory.retained_public_mib > 0.0

    assembly = report.result_assembly[0]
    assert assembly.record_kind == "DispatchCohortResult"
    assert assembly.vm_output == "none"
    assert assembly.observation_output == 'observations["vm_raster"]'

    detail = report.assembly_details[0]
    assert detail.vm_shape is None
    assert detail.observation_shape == (2, 1, 1, 1)
    assert detail.observations_are_batched is True
    assert detail.public_rows == 1

    text = report.format()
    assert "padding:" in text
    assert "probes:" in text
    assert "memory:" in text
    assert "assembly details:" in text


def test_inspection_detailed_plot_covers_padding_memory_probes_and_results():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    report = axs.inspect_simulation(
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
    report = axs.inspect_simulation(
        _clamped_pool(),
        duration=0.10 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.none(),
        observers=[peak],
    )

    assert report.lowerings[0].observer_format == "unsupported_observer_only"
    assert report.result_assembly[0].observation_output == "unsupported_observer_only"
    assert report.probes[0].packed_shape is None
    assert report.assembly_details[0].observation_shape is None
