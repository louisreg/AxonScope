from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import axonscope as axs
import axonscope.runtime.jax.batch_kernels as batch_kernels
from axonscope.preparation.runtime_batches import scale_extracellular_stimulations
from axonscope.results import VM_RASTER_OBSERVATION_KEY
from axonscope.runtime.jax.batch_inputs import (
    FactorizedExtracellularPotentialBatch,
    materialize_factorized_extracellular_potential_batch,
    materialize_factorized_extracellular_potential_initial_previous,
)
from axonscope.runtime.jax.batch_kernels import (
    DoubleCableBatchKernel,
    _resolve_double_cable_kernel_block_solver,
    _resolve_double_cable_run_block_solver,
    _use_batch_native_double_cable_integrated_solver,
    _use_batch_native_double_cable_pcr_soa_solver,
)
from axonscope.runtime.jax.input_batches import (
    build_factorized_vstim_midpoint_batch,
    build_vstim_initial_previous_batch,
    build_vstim_midpoint_batch,
)
from axonscope.runtime.jax.observer_runtime import build_vm_raster_plan
from axonscope.runtime.jax.runtime import prepare_solver_runtime
from axonscope.solvers import BatchOptions, BatchRecording

from tests.unit.solvers._batch_helpers import (
    DIAGNOSTIC_DOUBLE_CABLE_BLOCK_SOLVERS,
    diagnostic_double_cable_solver_engine,
    hh_extracellular_axon,
    kernel_observations,
)


def test_gpu_pcr_adaptive_prefers_soa_through_p100_calibrated_batch_range():
    assert (
        _resolve_double_cable_kernel_block_solver("pcr_adaptive", batch_size=4096)
        == "pcr_soa"
    )
    assert _resolve_double_cable_kernel_block_solver("pcr_adaptive", batch_size=4097) == "pcr"


def test_gpu_diagnostic_pcr_soa_batch_native_route_starts_at_realistic_batches():
    assert not _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=15)
    assert _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=25)
    assert _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=50)
    assert _use_batch_native_double_cable_pcr_soa_solver("pcr_soa", batch_size=2048)
    assert not _use_batch_native_double_cable_pcr_soa_solver("pcr", batch_size=50)
    assert _use_batch_native_double_cable_integrated_solver(
        "jax_triton_loop_xb",
        batch_size=1,
    )


def test_internal_jax_triton_solver_is_benchmark_override_only():
    with pytest.raises(ValueError, match="double_cable_block_solver"):
        _resolve_double_cable_run_block_solver(
            diagnostic_double_cable_solver_engine("jax_triton_loop_xb"),
            platform="gpu",
        )
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        _resolve_double_cable_run_block_solver(
            diagnostic_double_cable_solver_engine(
                "jax_triton_loop_xb",
                allow_internal=True,
            ),
            platform="cpu",
        )
    assert (
        _resolve_double_cable_run_block_solver(
            diagnostic_double_cable_solver_engine(
                "jax_triton_loop_xb",
                platform="gpu",
                allow_internal=True,
            ),
            platform="gpu",
        )
        == "jax_triton_loop_xb"
    )


def test_diagnostic_double_cable_batch_pcr_solvers_match_thomas_kernel_reference():
    """Diagnostic kernel equivalence only; CPU production policy is Thomas-only."""

    axon = hh_extracellular_axon(current_clamp=False)
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
    for solver in DIAGNOSTIC_DOUBLE_CABLE_BLOCK_SOLVERS:
        pcr = kernel.run(
            extracellular_potential_mid_mV=vext_mid,
            extracellular_potential_initial_previous_mV=vext_previous,
            options=BatchOptions.center(),
            solver_engine=diagnostic_double_cable_solver_engine(solver),
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
        options=BatchOptions.center(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
    ).Vm
    pcr_soa_chunked = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.center(time_chunk_steps=7),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
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
        options=BatchOptions.none(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
    ).Vm
    assert pcr_soa_none.shape == (2, int(round(tsim / dt)), 0)


def test_diagnostic_double_cable_compact_event_observer_pcr_soa_matches_full_vm(
    monkeypatch,
):
    axon = hh_extracellular_axon(current_clamp=False)
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
        options=BatchOptions.none(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
        observers=observer,
    )
    full = kernel.run(
        extracellular_potential_mid_mV=vext_mid,
        extracellular_potential_initial_previous_mV=vext_previous,
        options=BatchOptions.full(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
    )

    center = axon.n_compartments // 2
    assert compact.Vm is None
    raster = kernel_observations(compact)[VM_RASTER_OBSERVATION_KEY]
    np.testing.assert_array_equal(
        np.any(raster.unpack()[:, 0, 0, :], axis=1),
        np.any(np.asarray(full.Vm)[:, :, center] >= -80.0, axis=1),
    )


def test_diagnostic_double_cable_factorized_footprint_observer_matches_dense_pcr_soa(
    monkeypatch,
):
    axon = hh_extracellular_axon(current_clamp=False)
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
        options=BatchOptions.none(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
        observers=observer,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=factorized,
        options=BatchOptions.none(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
        observers=observer,
    )

    assert dense.Vm is None
    assert compact.Vm is None
    dense_observations = kernel_observations(dense)
    compact_observations = kernel_observations(compact)
    np.testing.assert_array_equal(
        np.asarray(compact_observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense_observations[VM_RASTER_OBSERVATION_KEY].words),
    )


def test_diagnostic_double_cable_factorized_row_specific_current_observer_matches_dense_pcr_soa(
    monkeypatch,
):
    axon = hh_extracellular_axon(current_clamp=False)
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
        options=BatchOptions.none(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
        observers=observer,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=row_specific,
        options=BatchOptions.none(),
        solver_engine=diagnostic_double_cable_solver_engine("pcr_soa"),
        observers=observer,
    )

    dense_observations = kernel_observations(dense)
    compact_observations = kernel_observations(compact)
    np.testing.assert_array_equal(
        np.asarray(compact_observations[VM_RASTER_OBSERVATION_KEY].words),
        np.asarray(dense_observations[VM_RASTER_OBSERVATION_KEY].words),
    )


@pytest.mark.parametrize("solver", ["thomas", "pcr_soa"])
def test_diagnostic_double_cable_scaled_factorized_probe_vm_avoids_dense_vstim_materialization(
    monkeypatch,
    solver,
):
    axon = hh_extracellular_axon(current_clamp=False)
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
    stimulation = axon.extracellular_stimulations[0]
    stimulations = [
        scale_extracellular_stimulations((stimulation,), 1.0),
        scale_extracellular_stimulations((stimulation,), 0.5),
    ]
    dense_mid = build_vstim_midpoint_batch(
        axon,
        stimulations,
        tsim_ms=tsim,
        dt_ms=dt,
    )
    dense_previous = build_vstim_initial_previous_batch(
        axon,
        stimulations,
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
    assert factorized.scaled_shared_waveform is True

    if solver == "pcr_soa":
        monkeypatch.setattr(
            batch_kernels,
            "_DOUBLE_CABLE_BATCH_NATIVE_PCR_SOA_MIN_BATCH",
            1,
        )
    kernel = DoubleCableBatchKernel(runtime=runtime, Veinit_mV=float(axon.Veinit))
    options = BatchOptions(
        recording=BatchRecording.indices([axon.n_compartments // 2]),
        time_chunk_steps=17,
    )
    dense = kernel.run(
        extracellular_potential_mid_mV=dense_mid,
        extracellular_potential_initial_previous_mV=dense_previous,
        options=options,
        solver_engine=diagnostic_double_cable_solver_engine(solver),
    )

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("double-cable compact probe path materialized dense Vstim")

    monkeypatch.setattr(
        batch_kernels,
        "materialize_factorized_extracellular_potential_batch",
        fail_materialize,
    )
    monkeypatch.setattr(
        batch_kernels,
        "materialize_factorized_extracellular_potential_initial_previous",
        fail_materialize,
    )
    compact = kernel.run(
        extracellular_potential_mid_mV=factorized,
        options=options,
        solver_engine=diagnostic_double_cable_solver_engine(solver),
    )

    assert dense.Vm is not None
    assert compact.Vm is not None
    np.testing.assert_allclose(
        np.asarray(compact.Vm),
        np.asarray(dense.Vm),
        atol=1e-3,
        rtol=0.0,
    )
