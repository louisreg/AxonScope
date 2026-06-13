from __future__ import annotations

from dataclasses import replace

import numpy as np
import jax.numpy as jnp
import pytest

import axonscope as axs
from axonscope import AxonSimulation
from axonscope.axons import HodgkinHuxley
from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.dispatcher.runtime_batches import (
    build_footprint_vstim_initial_previous_batch,
    build_footprint_vstim_midpoint_batch,
    build_vstim_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
    scale_extracellular_contexts,
)
from axonscope.solvers import (
    BatchOptions,
    BatchRecording,
    DoubleCableBatchKernel,
    DoubleCableKernel,
    SingleCableVStimBatchKernel,
)
from axonscope.solvers.experimental import CrankNicholsonVStimForcing
from axonscope.solvers.runtime import prepare_solver_runtime
from axonscope.stimulation import Stimulus


def _context(electrode: PointSourceElectrode, stimulus: Stimulus, *, sigma=0.3):
    return AnalyticalExtracellularContext(electrodes=[electrode.with_stimulus(stimulus)], sigma=sigma)


def _hh_extracellular_axon() -> AxonSimulation:
    axon = AxonSimulation(
        HodgkinHuxley(
            length=400.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=41,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_current_clamp(
        position_um=200.0,
        current=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    electrode = PointSourceElectrode(
        x0_m=200e-6,
        y0_m=100e-6,
        z0_m=100e-6,
    )
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    axon.add_extracellular_context(context=_context(electrode, stim), replace=True)
    return axon


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


def test_build_vstim_midpoint_batch_matches_runtime_and_scales_contexts():
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
    base_contexts = tuple(axon.extracellular_contexts)

    vext_batch = build_vstim_midpoint_batch(
        axon,
        [
            base_contexts[0],
            scale_extracellular_contexts(base_contexts, 0.5),
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


def test_build_vstim_batch_accepts_per_row_positions():
    axon = _hh_extracellular_axon()
    contexts = tuple(axon.extracellular_contexts)
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    shifted_x_m = base_x_m + 25e-6

    vext_batch = build_vstim_batch(
        axon,
        [contexts, contexts],
        t_ms=jnp.asarray([0.31]),
        x_positions_m=np.stack([base_x_m, shifted_x_m]),
    )

    assert vext_batch.shape == (2, 1, axon.n_compartments)
    assert float(np.max(np.abs(np.asarray(vext_batch[0] - vext_batch[1])))) > 0.0


def test_build_footprint_vstim_batch_matches_generic_context_builder():
    axon = _hh_extracellular_axon()
    base_context = axon.extracellular_contexts[0]
    tsim = 1.2
    dt = 0.01
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    shifted_x_m = base_x_m + 25e-6
    x_positions_m = np.stack([base_x_m, shifted_x_m])
    footprint = np.stack(
        [
            base_context.footprint_for_electrode(base_context.electrodes[0], base_x_m),
            base_context.footprint_for_electrode(base_context.electrodes[0], shifted_x_m),
        ]
    )

    generic = build_vstim_midpoint_batch(
        axon,
        [
            base_context,
            scale_extracellular_contexts((base_context,), 0.5),
        ],
        tsim_ms=tsim,
        dt_ms=dt,
        x_positions_m=x_positions_m,
    )
    from_footprint = build_footprint_vstim_midpoint_batch(
        stimulus=base_context.electrodes[0].stimulus,
        footprint_V_per_A=footprint,
        amplitude_scale=jnp.asarray([1.0, 0.5]),
        tsim_ms=tsim,
        dt_ms=dt,
    )
    previous = build_footprint_vstim_initial_previous_batch(
        stimulus=base_context.electrodes[0].stimulus,
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
    base_contexts = tuple(axon.extracellular_contexts)
    context_batch = [
        base_contexts,
        scale_extracellular_contexts(base_contexts, 0.5),
    ]
    vext_mid = build_vstim_midpoint_batch(axon, context_batch, tsim_ms=tsim, dt_ms=dt)
    vext_previous = build_vstim_initial_previous_batch(
        axon,
        context_batch,
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
    base_context = axon.extracellular_contexts[0]
    base_x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    footprint = np.stack(
        [
            base_context.footprint_for_electrode(base_context.electrodes[0], base_x_m),
            base_context.footprint_for_electrode(base_context.electrodes[0], base_x_m),
        ]
    )
    amplitude_scale = jnp.asarray([1.0, 0.5])
    vext_mid = build_footprint_vstim_midpoint_batch(
        stimulus=base_context.electrodes[0].stimulus,
        footprint_V_per_A=footprint,
        amplitude_scale=amplitude_scale,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    vext_previous = build_footprint_vstim_initial_previous_batch(
        stimulus=base_context.electrodes[0].stimulus,
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
