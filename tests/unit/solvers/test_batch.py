from __future__ import annotations

from dataclasses import replace

import numpy as np
import jax.numpy as jnp
import pytest

import axonscope as axs
import axonscope.backends.jax.batch_kernels as batch_kernels
from axonscope import AxonInstance
from axonscope.axons import HodgkinHuxley
from axonscope.backends.jax.input_batches import (
    build_factorized_vstim_midpoint_batch,
    build_footprint_vstim_initial_previous_batch,
    build_footprint_vstim_midpoint_batch,
    build_sparse_intracellular_current_density_batch,
    build_vstim_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_and_initial_previous_batch,
    build_vstim_midpoint_batch,
)
from axonscope.preparation.runtime_batches import (
    scale_extracellular_stimulations,
)
from axonscope.analytical import PointSourceElectrode
from axonscope.backends.jax.batch_kernels import (
    DoubleCableBatchKernel,
    SingleCableVStimBatchKernel,
)
from axonscope.backends.execution import batch_options_for_execution_context
from axonscope.backends.jax.recording import batch_options_from_recording
from axonscope.backends.jax.kernels import DoubleCableKernel
from axonscope.solvers import (
    BatchOptions,
    BatchRecording,
    DEFAULT_OBSERVER_TIME_CHUNK_STEPS,
    resolve_double_cable_block_solver,
)
from axonscope.backends.jax.batch_kernels import (
    _resolve_double_cable_kernel_block_solver,
    _use_batch_native_double_cable_pcr_soa_solver,
)
from axonscope.backends.jax.batch_inputs import (
    FactorizedExtracellularPotentialBatch,
    materialize_factorized_extracellular_potential_initial_previous,
    materialize_factorized_extracellular_potential_batch,
)
from axonscope.results import VM_RASTER_OBSERVATION_KEY
from axonscope.backends.jax.observer_runtime import build_vm_raster_plan
from axonscope.backends.jax.experimental import CrankNicholsonVStimForcing
from axonscope.backends.jax.runtime import prepare_solver_runtime
from axonscope.stimulation import Stimulus


def _hh_extracellular_axon(*, current_clamp: bool = True) -> AxonInstance:
    axon = AxonInstance(
        HodgkinHuxley(
            length=400.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=41,
            celsius=6.3 * axs.degC,
        )
    )
    if current_clamp:
        axon.add_current_clamp(
            position=200.0 * axs.um,
            current=Stimulus.pulse(start=0.4 * axs.ms, duration=0.05 * axs.ms, amplitude=0.8),
        )
    electrode = PointSourceElectrode(
        x=200e-6 * axs.m,
        y=100e-6 * axs.m,
        z=100e-6 * axs.m,
    )
    stim = Stimulus.pulse(start=0.3 * axs.ms, amplitude=20e-6, duration=0.1 * axs.ms, baseline=0.0)
    axon.add_extracellular_stimulation(
        stimulation=axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            stimulus=stim,
            sigma=0.3 * axs.S_per_m,
        ),
        replace=True,
    )
    return axon


def _drive_footprint_for_positions(drive, x_positions_m) -> np.ndarray:
    footprint = drive.footprint
    values = np.asarray(footprint.values_for_axon(), dtype=float)
    x_um = np.asarray(x_positions_m, dtype=float) * 1e6
    support_um = np.asarray(footprint.positions_um, dtype=float)
    if x_um.shape == support_um.shape and np.allclose(x_um, support_um):
        return values
    return np.interp(x_um, support_um, values)


def test_batch_recording_resolves_common_policies():
    assert BatchRecording.full().indices_for(5) is None
    np.testing.assert_array_equal(BatchRecording.center().indices_for(5), np.asarray([2]))
    np.testing.assert_array_equal(
        BatchRecording.probes(3).indices_for(5),
        np.asarray([0, 2, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        BatchRecording.indices([0, 4]).indices_for(5),
        np.asarray([0, 4], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="within"):
        BatchRecording.indices([5]).indices_for(5)
    assert BatchOptions.center().double_cable_block_solver == "auto"
    assert BatchOptions.center(double_cable_block_solver="auto").double_cable_block_solver == "auto"
    assert BatchOptions.center(double_cable_block_solver="pcr").double_cable_block_solver == "pcr"
    assert (
        BatchOptions.center(double_cable_block_solver="pcr_soa").double_cable_block_solver
        == "pcr_soa"
    )
    assert (
        BatchOptions.center(double_cable_block_solver="pcr_adaptive").double_cable_block_solver
        == "pcr_adaptive"
    )
    assert resolve_double_cable_block_solver("auto", platform="cpu") == "thomas"
    assert resolve_double_cable_block_solver("auto", platform="gpu") == "pcr_adaptive"
    assert resolve_double_cable_block_solver("thomas", platform="gpu") == "thomas"
    assert resolve_double_cable_block_solver("pcr_soa", platform="gpu") == "pcr_soa"
    with pytest.raises(ValueError, match="double_cable_block_solver"):
        BatchOptions(double_cable_block_solver="dense")


def test_batch_options_none_defaults_to_observer_chunking():
    default = BatchOptions.none()
    unchunked = BatchOptions.none(time_chunk_steps=None)

    assert default.recording.mode == "none"
    assert default.time_chunk_steps == DEFAULT_OBSERVER_TIME_CHUNK_STEPS
    assert unchunked.recording.mode == "none"
    assert unchunked.time_chunk_steps is None


def test_recording_none_lowers_to_default_observer_chunking():
    options = batch_options_from_recording(axs.Recording.none())

    assert options is not None
    assert options.recording.mode == "none"
    assert options.time_chunk_steps == DEFAULT_OBSERVER_TIME_CHUNK_STEPS


def test_pcr_adaptive_prefers_soa_through_p100_calibrated_batch_range():
    assert (
        _resolve_double_cable_kernel_block_solver("pcr_adaptive", batch_size=4096)
        == "pcr_soa"
    )
    assert _resolve_double_cable_kernel_block_solver("pcr_adaptive", batch_size=4097) == "pcr"


def test_pcr_soa_batch_native_route_starts_at_realistic_batches():
    assert not _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=15)
    assert _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=25)
    assert _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=50)
    assert _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=2048)
    assert not _use_batch_native_double_cable_pcr_soa_solver("pcr", batch_size=50)


def test_execution_context_auto_keeps_cpu_solver_policy():
    context = type("Context", (), {"platform": "cpu"})()

    full = batch_options_for_execution_context(BatchOptions.full(), context)
    compact = batch_options_for_execution_context(
        BatchOptions.none(),
        context,
    )
    explicit = batch_options_for_execution_context(
        BatchOptions.none(double_cable_block_solver="thomas"),
        context,
    )

    assert full is not None
    assert compact is not None
    assert explicit is not None
    assert full.double_cable_block_solver == "thomas"
    assert compact.double_cable_block_solver == "thomas"
    assert explicit.double_cable_block_solver == "thomas"


def test_single_cable_vstim_batch_matches_scalar_reference_row():
    axon = _hh_extracellular_axon()
    tsim = 1.2
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )
    assert runtime.stimulation.intracellular_current_density_mid is not None
    assert runtime.stimulation.extracellular_potential_mid_mV is not None

    vext_mid = runtime.stimulation.extracellular_potential_mid_mV
    vext_batch = jnp.stack([vext_mid, 0.5 * vext_mid])
    batch = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
    ).run(extracellular_potential_mid_mV=vext_batch)
    scalar = CrankNicholsonVStimForcing().solve(_hh_extracellular_axon(), tsim=tsim, dt=dt)

    assert batch.Vm.shape == (2, scalar.Vm.shape[0], scalar.Vm.shape[1])
    np.testing.assert_allclose(np.asarray(batch.t), np.asarray(scalar.t), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(batch.Vm[0]), np.asarray(scalar.Vm), atol=1e-3, rtol=0.0)
    assert np.isfinite(np.asarray(batch.Vm)).all()
    assert float(np.max(np.abs(np.asarray(batch.Vm[0]) - np.asarray(batch.Vm[1])))) > 1e-8


def test_single_cable_vstim_batch_validates_shapes():
    axon = _hh_extracellular_axon()
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=0.2,
        dt_ms=0.01,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )
    kernel = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
    )

    with pytest.raises(ValueError, match="extracellular_potential_mid_mV"):
        kernel.run(extracellular_potential_mid_mV=jnp.zeros((runtime.grid.Nt, axon.n_compartments + 1)))


def test_single_cable_vstim_batch_records_probes_in_time_chunks():
    axon = _hh_extracellular_axon()
    tsim = 0.6
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )
    assert runtime.stimulation.extracellular_potential_mid_mV is not None
    vext_mid = runtime.stimulation.extracellular_potential_mid_mV
    vext_batch = jnp.stack([vext_mid, 0.5 * vext_mid])
    kernel = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
    )

    full = kernel.run(extracellular_potential_mid_mV=vext_batch).Vm
    probe_indices = jnp.asarray([0, axon.n_compartments // 2, axon.n_compartments - 1])
    probes = kernel.run(
        extracellular_potential_mid_mV=vext_batch,
        options=BatchOptions(
            recording=BatchRecording.indices(probe_indices.tolist()),
            time_chunk_steps=17,
        ),
    ).Vm

    assert probes.shape == (2, runtime.grid.Nt, 3)
    np.testing.assert_allclose(
        np.asarray(probes),
        np.asarray(full[:, :, probe_indices]),
        atol=1e-3,
        rtol=0.0,
    )


def test_build_vstim_midpoint_batch_matches_runtime_and_scales_stimulations():
    axon = _hh_extracellular_axon()
    tsim = 1.2
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)

    vext_batch = build_vstim_midpoint_batch(
        axon,
        [
            base_stimulations[0],
            scale_extracellular_stimulations(base_stimulations, 0.5),
            None,
        ],
        tsim_ms=tsim,
        dt_ms=dt,
    )

    assert runtime.stimulation.extracellular_potential_mid_mV is not None
    assert vext_batch.shape == (3, runtime.grid.Nt, axon.n_compartments)
    np.testing.assert_allclose(
        np.asarray(vext_batch[0]),
        np.asarray(runtime.stimulation.extracellular_potential_mid_mV),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(vext_batch[1]),
        0.5 * np.asarray(vext_batch[0]),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(vext_batch[2]), 0.0, atol=0.0, rtol=0.0)


def test_build_vstim_midpoint_batch_rejects_partial_final_step():
    axon = _hh_extracellular_axon()

    with pytest.raises(ValueError, match="integer multiple"):
        build_vstim_midpoint_batch(axon, [None], tsim_ms=1.0, dt_ms=0.3)


def test_combined_vstim_builder_matches_separate_double_cable_inputs():
    axon = _hh_extracellular_axon()
    stimulations = tuple(axon.extracellular_stimulations)
    tsim = 0.2
    dt = 0.05

    separate_mid = build_vstim_midpoint_batch(
        axon,
        [stimulations, stimulations],
        tsim_ms=tsim,
        dt_ms=dt,
    )
    separate_previous = build_vstim_initial_previous_batch(
        axon,
        [stimulations, stimulations],
        dt_ms=dt,
    )
    combined_mid, combined_previous = build_vstim_midpoint_and_initial_previous_batch(
        axon,
        [stimulations, stimulations],
        tsim_ms=tsim,
        dt_ms=dt,
    )

    np.testing.assert_allclose(
        np.asarray(combined_mid),
        np.asarray(separate_mid),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(combined_previous),
        np.asarray(separate_previous),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_vstim_batch_accepts_per_row_positions():
    axon = _hh_extracellular_axon()
    stimulations = tuple(axon.extracellular_stimulations)
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    shifted_x_m = base_x_m + 25e-6

    vext_batch = build_vstim_batch(
        axon,
        [stimulations, stimulations],
        t_ms=jnp.asarray([0.31]),
        x_positions_m=np.stack([base_x_m, shifted_x_m]),
    )

    assert vext_batch.shape == (2, 1, axon.n_compartments)
    assert float(np.max(np.abs(np.asarray(vext_batch[0] - vext_batch[1])))) > 0.0


def test_build_vstim_batch_shared_point_source_matches_drive_footprint():
    axon = _hh_extracellular_axon()
    stimulation = axon.extracellular_stimulations[0]
    drive = stimulation.drives[0]
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    t_ms = np.asarray([0.35], dtype=float)

    vext_batch = build_vstim_batch(
        axon,
        [stimulation, stimulation],
        t_ms=jnp.asarray(t_ms),
        x_positions_m=np.stack([base_x_m, base_x_m]),
    )

    current_A = drive.stimulus.evaluate(t_ms, unit="ampere")
    row = current_A[:, None] * _drive_footprint_for_positions(drive, base_x_m)[None, :] * 1e3
    expected = np.stack([row, row], axis=0)
    np.testing.assert_allclose(np.asarray(vext_batch), expected, rtol=1e-6, atol=1e-6)


def test_build_vstim_batch_uses_sampled_stimulation_offsets():
    axon = _hh_extracellular_axon()
    base = axon.extracellular_stimulation
    assert base is not None
    drive = base.drives[0]
    shifted = axs.ExtracellularStimulation(
        [
            replace(
                drive,
                footprint=axs.analytical.point_source_footprint(
                    PointSourceElectrode(x=200.0 * axs.um, y=100.0 * axs.um, z=100.0 * axs.um),
                    axon.layout.position_values(unit=axs.um) * axs.um,
                    sigma=0.3 * axs.S_per_m,
                    axon_y=50.0 * axs.um,
                ),
            )
        ]
    )

    vext_batch = build_vstim_batch(
        axon,
        [base, shifted],
        t_ms=jnp.asarray([0.35]),
    )

    assert float(np.max(np.abs(np.asarray(vext_batch[0] - vext_batch[1])))) > 0.0


def test_factorized_footprint_batch_matches_dense_builder_and_observer_raster():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    stimulation = axon.extracellular_stimulations[0]
    stimulations = [stimulation, stimulation]
    dense = build_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    factorized = build_factorized_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )

    assert factorized is not None
    materialized = materialize_factorized_extracellular_potential_batch(factorized)
    np.testing.assert_allclose(
        np.asarray(materialized),
        np.asarray(dense),
        rtol=1e-6,
        atol=1e-6,
    )
    footprint = jnp.asarray(factorized.footprint_mV_per_A, dtype=runtime.membrane.dtype)
    lower_rows = jnp.broadcast_to(runtime.cable.lower[None, :], footprint.shape)
    upper_rows = jnp.broadcast_to(runtime.cable.upper[None, :], footprint.shape)
    forcing = batch_kernels._single_cable_factorized_forcing_footprint(
        footprint,
        lower=lower_rows,
        upper=upper_rows,
        dtype_local=runtime.membrane.dtype,
    )
    expected_forcing = jnp.stack(
        [
            batch_kernels.apply_diffusion_operator(
                row,
                runtime.cable.lower,
                runtime.cable.diag,
                runtime.cable.upper,
            )
            for row in footprint
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        np.asarray(forcing),
        np.asarray(expected_forcing),
        rtol=1e-6,
        atol=1e-6,
    )

    sparse_iinj = build_sparse_intracellular_current_density_batch(
        [axon, axon],
        runtime,
        solver_axons=[runtime.axon, runtime.axon],
    )
    activation = axs.analysis.Activation(
        threshold=-20.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
        has_driven_extracellular=True,
    )

    dense_out = kernel.run(
        extracellular_potential_mid_mV=dense,
        intracellular_current_density_mid=sparse_iinj,
        options=BatchOptions.none(),
        observers=observer,
    )
    factorized_out = kernel.run(
        extracellular_potential_mid_mV=factorized,
        intracellular_current_density_mid=sparse_iinj,
        options=BatchOptions.none(),
        observers=observer,
    )
    factorized_chunked = kernel.run(
        extracellular_potential_mid_mV=factorized,
        intracellular_current_density_mid=sparse_iinj,
        options=BatchOptions.none(time_chunk_steps=17),
        observers=observer,
    )

    assert dense_out.Vm is None
    assert factorized_out.Vm is None
    assert factorized_chunked.Vm is None
    assert dense_out.observations is not None
    assert factorized_out.observations is not None
    assert factorized_chunked.observations is not None
    np.testing.assert_array_equal(
        np.asarray(factorized_out.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense_out.observations[VM_RASTER_OBSERVATION_KEY].words),
    )
    np.testing.assert_array_equal(
        np.asarray(factorized_chunked.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense_out.observations[VM_RASTER_OBSERVATION_KEY].words),
    )


def test_factorized_footprint_batch_supports_row_specific_currents():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    stimulation = axon.extracellular_stimulations[0]
    stimulations = [
        scale_extracellular_stimulations((stimulation,), 1.0)[0],
        scale_extracellular_stimulations((stimulation,), 0.5)[0],
    ]
    dense = build_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    factorized = build_factorized_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
        include_initial_previous=True,
    )

    assert factorized is not None
    assert factorized.current_mid_A.shape == (2, int(tsim / dt))
    assert factorized.current_initial_previous_A is not None
    assert factorized.current_initial_previous_A.shape == (2,)
    assert factorized.shared_current is False
    materialized = materialize_factorized_extracellular_potential_batch(factorized)
    np.testing.assert_allclose(
        np.asarray(materialized),
        np.asarray(dense),
        rtol=1e-6,
        atol=1e-6,
    )


def test_factorized_footprint_batch_supports_multi_drive_observer_without_dense_fallback():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    stimulation = axon.extracellular_stimulations[0]
    first = stimulation.drives[0]
    second = axs.ExtracellularDrive(
        id=axs.DriveId("second_point_source"),
        footprint=first.footprint,
        stimulus=Stimulus.pulse(
            start=0.35 * axs.ms,
            duration=0.08 * axs.ms,
            amplitude=10e-6,
        ),
    )
    multi_drive = axs.ExtracellularStimulation([first, second])
    stimulations = [multi_drive, multi_drive]
    dense = build_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    factorized = build_factorized_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )

    assert factorized is not None
    assert factorized.drive_count == 2
    assert factorized.current_mid_A.shape == (2, 2, int(tsim / dt))
    assert factorized.footprint_mV_per_A.shape == (2, 2, axon.n_compartments)
    materialized = materialize_factorized_extracellular_potential_batch(factorized)
    np.testing.assert_allclose(
        np.asarray(materialized),
        np.asarray(dense),
        rtol=1e-6,
        atol=1e-6,
    )

    sparse_iinj = build_sparse_intracellular_current_density_batch(
        [axon, axon],
        runtime,
        solver_axons=[runtime.axon, runtime.axon],
    )
    activation = axs.analysis.Activation(
        threshold=-20.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
        has_driven_extracellular=True,
    )

    dense_out = kernel.run(
        extracellular_potential_mid_mV=dense,
        intracellular_current_density_mid=sparse_iinj,
        options=BatchOptions.none(),
        observers=observer,
    )
    factorized_out = kernel.run(
        extracellular_potential_mid_mV=factorized,
        intracellular_current_density_mid=sparse_iinj,
        options=BatchOptions.none(),
        observers=observer,
    )

    assert dense_out.observations is not None
    assert factorized_out.observations is not None
    np.testing.assert_array_equal(
        np.asarray(factorized_out.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense_out.observations[VM_RASTER_OBSERVATION_KEY].words),
    )


def test_single_cable_factorized_recorded_vm_avoids_dense_vstim_materialization(monkeypatch):
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    stimulation = axon.extracellular_stimulations[0]
    first = stimulation.drives[0]
    second = axs.ExtracellularDrive(
        id=axs.DriveId("second_point_source"),
        footprint=first.footprint,
        stimulus=Stimulus.pulse(
            start=0.35 * axs.ms,
            duration=0.08 * axs.ms,
            amplitude=10e-6,
        ),
    )
    multi_drive = axs.ExtracellularStimulation([first, second])
    stimulations = [multi_drive, multi_drive]
    dense = build_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    factorized = build_factorized_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    assert factorized is not None

    kernel = SingleCableVStimBatchKernel(
        runtime=runtime,
        Cm_uF_cm2=jnp.asarray(runtime.axon.Cm_uF_cm2, dtype=runtime.membrane.dtype),
        has_driven_extracellular=True,
    )
    dense_out = kernel.run(
        extracellular_potential_mid_mV=dense,
        options=BatchOptions(),
    )

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("factorized recorded-Vm path materialized dense Vstim")

    monkeypatch.setattr(
        batch_kernels,
        "materialize_factorized_extracellular_potential_batch",
        fail_materialize,
    )
    factorized_out = kernel.run(
        extracellular_potential_mid_mV=factorized,
        options=BatchOptions(),
    )

    assert dense_out.Vm is not None
    assert factorized_out.Vm is not None
    np.testing.assert_allclose(
        np.asarray(factorized_out.Vm),
        np.asarray(dense_out.Vm),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_footprint_vstim_batch_matches_generic_stimulation_builder():
    axon = _hh_extracellular_axon()
    base_stimulation = axon.extracellular_stimulations[0]
    drive = base_stimulation.drives[0]
    tsim = 1.2
    dt = 0.01
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    shifted_x_m = base_x_m + 25e-6
    x_positions_m = np.stack([base_x_m, shifted_x_m])
    footprint = np.stack(
        [
            _drive_footprint_for_positions(drive, base_x_m),
            _drive_footprint_for_positions(drive, shifted_x_m),
        ]
    )

    generic = build_vstim_midpoint_batch(
        axon,
        [
            base_stimulation,
            scale_extracellular_stimulations((base_stimulation,), 0.5),
        ],
        tsim_ms=tsim,
        dt_ms=dt,
        x_positions_m=x_positions_m,
    )
    from_footprint = build_footprint_vstim_midpoint_batch(
        stimulus=drive.stimulus,
        footprint_V_per_A=footprint,
        amplitude_scale=jnp.asarray([1.0, 0.5]),
        tsim_ms=tsim,
        dt_ms=dt,
    )
    previous = build_footprint_vstim_initial_previous_batch(
        stimulus=drive.stimulus,
        footprint_V_per_A=footprint,
        amplitude_scale=jnp.asarray([1.0, 0.5]),
        dt_ms=dt,
    )

    assert from_footprint.shape == generic.shape
    assert previous.shape == (2, axon.n_compartments)
    np.testing.assert_allclose(
        np.asarray(from_footprint),
        np.asarray(generic),
        atol=1e-6,
        rtol=1e-6,
    )


def test_double_cable_batch_matches_scalar_loop_rows():
    axon = _hh_extracellular_axon()
    tsim = 0.8
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=True,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [
        base_stimulations,
        scale_extracellular_stimulations(base_stimulations, 0.5),
    ]
    vext_mid = build_vstim_midpoint_batch(axon, stimulation_batch, tsim_ms=tsim, dt_ms=dt)
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )

    batch = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    ).run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
    )

    scalar_rows = []
    for batch_index in range(vext_mid.shape[0]):
        row_stimulation = replace(
            runtime.stimulation,
            extracellular_potential_mid_mV=vext_mid[batch_index],
            extracellular_potential_initial_previous_mV=vext_previous[batch_index],
        )
        row_runtime = replace(runtime, stimulation=row_stimulation)
        scalar_rows.append(
            DoubleCableKernel(
                runtime=row_runtime,
                Veinit_mV=float(axon.Veinit),
            ).run().Vm
        )
    scalar = jnp.stack(scalar_rows)

    assert batch.Vm.shape == scalar.shape
    np.testing.assert_allclose(
        np.asarray(batch.t),
        np.asarray(runtime.grid.t_vec_ms),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(batch.Vm),
        np.asarray(scalar),
        atol=1e-3,
        rtol=0.0,
    )


def test_double_cable_batch_absent_intracellular_matches_explicit_zero_input():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [
        base_stimulations,
        scale_extracellular_stimulations(base_stimulations, 0.5),
    ]
    vext_mid = build_vstim_midpoint_batch(axon, stimulation_batch, tsim_ms=tsim, dt_ms=dt)
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )
    zero_iinj = jnp.zeros(
        (vext_mid.shape[0], runtime.grid.Nt, runtime.membrane.Nx),
        dtype=runtime.membrane.dtype,
    )
    kernel = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    )

    implicit_zero = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
    ).Vm
    explicit_zero = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        intracellular_current_density_mid=zero_iinj,
    ).Vm

    assert implicit_zero.shape == explicit_zero.shape
    np.testing.assert_allclose(
        np.asarray(implicit_zero),
        np.asarray(explicit_zero),
        atol=1e-3,
        rtol=0.0,
    )


def test_double_cable_batch_pcr_solver_matches_default_thomas_solver():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [
        base_stimulations,
        scale_extracellular_stimulations(base_stimulations, 0.5),
    ]
    vext_mid = build_vstim_midpoint_batch(axon, stimulation_batch, tsim_ms=tsim, dt_ms=dt)
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )
    kernel = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    )

    thomas = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.center(),
    ).Vm
    for solver in ("pcr", "pcr_soa", "pcr_adaptive"):
        pcr = kernel.run(
            extracellular_potential_mid_mV=vext_mid,
            extracellular_potential_initial_previous_mV=vext_previous,
            options=BatchOptions.center(double_cable_block_solver=solver),
        ).Vm

        assert pcr.shape == thomas.shape
        np.testing.assert_allclose(
            np.asarray(pcr),
            np.asarray(thomas),
            atol=1e-3,
            rtol=0.0,
        )

    pcr_soa_center = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.center(double_cable_block_solver="pcr_soa"),
    ).Vm
    pcr_soa_chunked = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.center(
            time_chunk_steps=7,
            double_cable_block_solver="pcr_soa",
        ),
    ).Vm
    np.testing.assert_allclose(
        np.asarray(pcr_soa_chunked),
        np.asarray(pcr_soa_center),
        atol=1e-3,
        rtol=0.0,
    )

    pcr_soa_none = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.none(double_cable_block_solver="pcr_soa"),
    ).Vm
    assert pcr_soa_none.shape == (2, int(round(tsim / dt)), 0)


def test_double_cable_compact_event_observer_thomas_matches_full_vm():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [
        base_stimulations,
        scale_extracellular_stimulations(base_stimulations, 0.5),
    ]
    vext_mid = build_vstim_midpoint_batch(axon, stimulation_batch, tsim_ms=tsim, dt_ms=dt)
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    )

    compact = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.none(double_cable_block_solver="thomas"),
        observers=observer,
    )
    full = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.full(double_cable_block_solver="thomas"),
    )

    center = axon.n_compartments // 2
    assert compact.Vm is None
    assert compact.observations is not None
    raster = compact.observations[VM_RASTER_OBSERVATION_KEY]
    np.testing.assert_array_equal(
        np.any(raster.unpack()[:, 0, 0, :], axis=1),
        np.any(np.asarray(full.Vm)[:, :, center] >= -80.0, axis=1),
    )


def test_double_cable_compact_event_observer_pcr_soa_batch_native_matches_full_vm(
    monkeypatch,
):
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [
        base_stimulations,
        scale_extracellular_stimulations(base_stimulations, 0.5),
    ]
    vext_mid = build_vstim_midpoint_batch(axon, stimulation_batch, tsim_ms=tsim, dt_ms=dt)
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = DoubleCableBatchKernel(
        runtime=runtime,
        Veinit_mV=float(axon.Veinit),
    )

    monkeypatch.setattr(
        batch_kernels,
        "_DOUBLE_CABLE_BATCH_NATIVE_PCR_SOA_MIN_BATCH",
        1,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.none(double_cable_block_solver="pcr_soa"),
        observers=observer,
    )
    full = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.full(double_cable_block_solver="pcr_soa"),
    )

    center = axon.n_compartments // 2
    assert compact.Vm is None
    assert compact.observations is not None
    raster = compact.observations[VM_RASTER_OBSERVATION_KEY]
    np.testing.assert_array_equal(
        np.any(raster.unpack()[:, 0, 0, :], axis=1),
        np.any(np.asarray(full.Vm)[:, :, center] >= -80.0, axis=1),
    )


def test_double_cable_factorized_footprint_observer_matches_dense_thomas():
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [base_stimulations, base_stimulations]
    dense_mid = build_vstim_midpoint_batch(
        axon,
        stimulation_batch,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    dense_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )
    factorized = build_factorized_vstim_midpoint_batch(
        axon,
        stimulation_batch,
        tsim_ms=tsim,
        dt_ms=dt,
        include_initial_previous=True,
    )
    assert factorized is not None

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = DoubleCableBatchKernel(runtime=runtime, Veinit_mV=float(axon.Veinit))

    dense = kernel.run(
        extracellular_potential_mid_mV=dense_mid,
        extracellular_potential_initial_previous_mV=dense_previous,
        options=BatchOptions.none(double_cable_block_solver="thomas"),
        observers=observer,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=factorized,
        options=BatchOptions.none(double_cable_block_solver="thomas"),
        observers=observer,
    )
    chunked = kernel.run(
        extracellular_potential_mid_mV=factorized,
        options=BatchOptions.none(
            double_cable_block_solver="thomas",
            time_chunk_steps=17,
        ),
        observers=observer,
    )

    assert dense.observations is not None
    assert compact.observations is not None
    assert chunked.observations is not None
    np.testing.assert_array_equal(
        np.asarray(compact.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense.observations[VM_RASTER_OBSERVATION_KEY].words),
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense.observations[VM_RASTER_OBSERVATION_KEY].words),
    )


def test_double_cable_factorized_footprint_observer_matches_dense_pcr_soa(
    monkeypatch,
):
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [base_stimulations, base_stimulations]
    dense_mid = build_vstim_midpoint_batch(
        axon,
        stimulation_batch,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    dense_previous = build_vstim_initial_previous_batch(
        axon,
        stimulation_batch,
        dt_ms=dt,
    )
    factorized = build_factorized_vstim_midpoint_batch(
        axon,
        stimulation_batch,
        tsim_ms=tsim,
        dt_ms=dt,
        include_initial_previous=True,
    )
    assert factorized is not None
    assert factorized.current_initial_previous_A is not None

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = DoubleCableBatchKernel(runtime=runtime, Veinit_mV=float(axon.Veinit))

    monkeypatch.setattr(
        batch_kernels,
        "_DOUBLE_CABLE_BATCH_NATIVE_PCR_SOA_MIN_BATCH",
        1,
    )
    dense = kernel.run(
        extracellular_potential_mid_mV=dense_mid,
        extracellular_potential_initial_previous_mV=dense_previous,
        options=BatchOptions.none(double_cable_block_solver="pcr_soa"),
        observers=observer,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=factorized,
        options=BatchOptions.none(double_cable_block_solver="pcr_soa"),
        observers=observer,
    )

    assert dense.Vm is None
    assert compact.Vm is None
    assert dense.observations is not None
    assert compact.observations is not None
    np.testing.assert_array_equal(
        np.asarray(compact.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense.observations[VM_RASTER_OBSERVATION_KEY].words),
    )


def test_double_cable_factorized_row_specific_current_observer_matches_dense_pcr_soa(
    monkeypatch,
):
    axon = _hh_extracellular_axon(current_clamp=False)
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )
    base_stimulations = tuple(axon.extracellular_stimulations)
    stimulation_batch = [base_stimulations, base_stimulations]
    shared = build_factorized_vstim_midpoint_batch(
        axon,
        stimulation_batch,
        tsim_ms=tsim,
        dt_ms=dt,
        include_initial_previous=True,
    )
    assert shared is not None
    assert shared.current_initial_previous_A is not None
    scale = jnp.asarray([1.0, 0.5], dtype=runtime.membrane.dtype)
    row_specific = FactorizedExtracellularPotentialBatch(
        current_mid_A=scale[:, None] * shared.current_mid_A[None, :],
        current_initial_previous_A=scale * shared.current_initial_previous_A,
        footprint_mV_per_A=shared.footprint_mV_per_A,
        target_nx=shared.target_nx,
    )
    dense_mid = materialize_factorized_extracellular_potential_batch(row_specific)
    dense_previous = materialize_factorized_extracellular_potential_initial_previous(
        row_specific
    )

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    observer = build_vm_raster_plan(
        (activation,),
        positions_um=runtime.axon.x_um,
        dtype=runtime.membrane.dtype,
    )
    assert observer is not None
    kernel = DoubleCableBatchKernel(runtime=runtime, Veinit_mV=float(axon.Veinit))

    monkeypatch.setattr(
        batch_kernels,
        "_DOUBLE_CABLE_BATCH_NATIVE_PCR_SOA_MIN_BATCH",
        1,
    )
    dense = kernel.run(
        extracellular_potential_mid_mV=dense_mid,
        extracellular_potential_initial_previous_mV=dense_previous,
        options=BatchOptions.none(double_cable_block_solver="pcr_soa"),
        observers=observer,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=row_specific,
        options=BatchOptions.none(double_cable_block_solver="pcr_soa"),
        observers=observer,
    )

    assert dense.observations is not None
    assert compact.observations is not None
    np.testing.assert_array_equal(
        np.asarray(compact.observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense.observations[VM_RASTER_OBSERVATION_KEY].words),
    )


def test_double_cable_batch_requires_extracellular_runtime():
    axon = _hh_extracellular_axon()
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=0.2,
        dt_ms=0.01,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=True,
        precompute_extracellular=True,
    )

    with pytest.raises(ValueError, match="include_extracellular=True"):
        DoubleCableBatchKernel(runtime=runtime).run()


def test_double_cable_materialized_chunks_match_full_batch():
    axon = _hh_extracellular_axon()
    tsim = 0.4
    dt = 0.01
    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=tsim,
        dt_ms=dt,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=True,
        precompute_extracellular=False,
    )
    base_stimulation = axon.extracellular_stimulations[0]
    drive = base_stimulation.drives[0]
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    footprint = np.stack(
        [
            _drive_footprint_for_positions(drive, base_x_m),
            _drive_footprint_for_positions(drive, base_x_m),
        ]
    )
    amplitude_scale = jnp.asarray([1.0, 0.5])
    vext_mid = build_footprint_vstim_midpoint_batch(
        stimulus=drive.stimulus,
        footprint_V_per_A=footprint,
        amplitude_scale=amplitude_scale,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    vext_previous = build_footprint_vstim_initial_previous_batch(
        stimulus=drive.stimulus,
        footprint_V_per_A=footprint,
        amplitude_scale=amplitude_scale,
        dt_ms=dt,
    )
    kernel = DoubleCableBatchKernel(runtime=runtime, Veinit_mV=float(axon.Veinit))

    full = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
    ).Vm
    chunked = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions(time_chunk_steps=11),
    ).Vm
    probe_indices = jnp.asarray([0, axon.n_compartments // 2, axon.n_compartments - 1])
    probe_chunks = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions(
            recording=BatchRecording.indices(probe_indices.tolist()),
            time_chunk_steps=11,
        ),
    ).Vm

    assert chunked.shape == full.shape
    assert probe_chunks.shape == (2, runtime.grid.Nt, 3)
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), atol=1e-3, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(probe_chunks),
        np.asarray(full[:, :, probe_indices]),
        atol=1e-3,
        rtol=0.0,
    )
