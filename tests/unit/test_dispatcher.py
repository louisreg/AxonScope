import re
from types import SimpleNamespace

import numpy as np

import axonscope as axs
import axonscope.backends.jax.group_runner as group_runner
import axonscope.backends.jax.input_batches as input_batches
from axonscope.backends.jax import runtime_caches, runtime_preparation, shape_bucketing
import axonscope.dispatcher.plan as dispatch_plan_module
import axonscope.dispatcher.progress as progress_module
from axonscope.benchmarking import benchmark_span
from axonscope.backends.jax.batch_inputs import (
    materialize_factorized_extracellular_potential_batch,
)
from axonscope.analytical import PointSourceElectrode
from axonscope.backends.jax.input_batches import (
    build_factorized_vstim_midpoint_batch,
    build_intracellular_current_density_batch,
    build_sparse_intracellular_current_density_batch,
    build_vstim_midpoint_batch,
)
from axonscope.dispatcher import build_dispatch_plan, run_pool
from axonscope.dispatcher._records import DispatchCohortRecord
from axonscope.preparation.runtime_batches import (
    extracellular_stimulation_rows,
    x_positions_batch_m,
)
from axonscope.solvers.axon_runtime import build_solver_axon
from axonscope.backends.jax.batch_inputs import (
    materialize_sparse_intracellular_current_density_batch,
)
from axonscope.solvers import BatchOptions
from axonscope.backends.jax.runtime import (
    prepare_cable_runtime,
    prepare_extracellular_runtime,
    prepare_solver_runtime,
)
from axonscope.stimulation import Stimulus


def _run_simulation(axons, **kwargs):
    return axs.AxonSimulation(axons, **kwargs).run()


def _hh_axon(*, nx: int, amp_nA: float, y_um: float = 0.0, z_um: float = 20.0):
    length_um = 100.0
    axon_model = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.5 * axs.um,
        compartments=nx,
        celsius=6.3 * axs.degC,
    )
    axon = axs.AxonInstance(axon_model)
    axon.add_current_clamp(
        position=(length_um / 2.0) * axs.um,
        current=Stimulus.pulse(
            start=0.02 * axs.ms,
            duration=0.04 * axs.ms,
            amplitude=amp_nA,
        ),
    )
    electrode = PointSourceElectrode(
        x=50.0 * axs.um,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    axon.add_extracellular_stimulation(
        stimulation=axs.analytical.point_source_stimulation(
            electrode,
            axon_model.layout.position_values(unit=axs.um) * axs.um,
            stimulus=Stimulus.pulse(
                start=0.0 * axs.ms,
                duration=0.05 * axs.ms,
                amplitude=10e-6,
            ),
            sigma=0.3 * axs.S_per_m,
            axon_y=y_um * axs.um,
            axon_z=z_um * axs.um,
        ),
    )
    return axon


def _passive_double_cable_axon(
    *,
    amp_nA: float,
    compartments: int = 11,
    length_um: float = 100.0,
):
    axon_model = axs.axons.Axon(
        layout=axs.axons.Layout.single_uniform(
            axs.axons.Section(
                "axon",
                membrane=axs.membranes.Passive(Rm=1e4, EL=-70.0),
                diameter=1.0 * axs.um,
                periaxonal=axs.axons.PeriaxonalLayer(
                    radial_conductance=1e-3 * axs.S_per_cm2,
                    radial_capacitance=0.01 * axs.uF_per_cm2,
                    axial_resistance=1e8 * axs.MOhm_per_cm,
                ),
            ),
            length=length_um * axs.um,
            compartments=compartments,
        ),
        formulation=axs.axons.CableFormulation.DOUBLE_CABLE,
        v_init=-70.0 * axs.mV,
    )
    axon = axs.AxonInstance(axon_model)
    axon.add_current_clamp(
        position=50.0 * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=amp_nA),
    )
    return axon


def _mrg_axon(
    *,
    diameter_um: float,
    amp_nA: float,
    nodes: int = 5,
    x_shift_um: float = 0.0,
):
    axon_model = axs.axons.MRG(
        diameter=diameter_um * axs.um,
        nodes=nodes,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
        x_shift=x_shift_um * axs.um,
    )
    axon = axs.AxonInstance(axon_model)
    axon.add_current_clamp(
        position=float(axon_model.layout.position_values(unit=axs.um)[0]) * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=amp_nA),
    )
    return axon


def test_pool_public_api_uses_local_contexts_not_axon_positions():
    axon_a = _hh_axon(nx=11, amp_nA=0.4, y_um=20.0, z_um=30.0)
    axon_b = _hh_axon(nx=13, amp_nA=0.2, y_um=60.0, z_um=10.0)

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert len(result) == 2
    assert [axon_result.diagnostics["pool_index"] for axon_result in result] == [0, 1]
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 1), (2, 1)]
    assert not hasattr(result[0].simulation, "y_um")
    assert not hasattr(result[1].simulation, "z_um")
    assert np.asarray([np.max(np.asarray(axon_result.Vm)) for axon_result in result]).shape == (2,)


def test_pool_dispatch_batches_compatible_axons_with_recording_filter():
    axon_a = _hh_axon(nx=11, amp_nA=0.4, y_um=20.0, z_um=30.0)
    axon_b = _hh_axon(nx=11, amp_nA=0.2, y_um=60.0, z_um=10.0)

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert len(result) == 2
    assert {axon_result.diagnostics["dispatch_method"] for axon_result in result} == {
        "batch-single-cable"
    }
    assert {axon_result.diagnostics["dispatch_group_size"] for axon_result in result} == {2}
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 1), (2, 1)]
    assert [axon_result.record_indices for axon_result in result] == [(5,), (5,)]


def test_pool_dispatch_batches_context_and_no_context_rows():
    length_um = 100.0
    axon_a = _hh_axon(nx=11, amp_nA=0.4, y_um=20.0, z_um=30.0)
    axon_b_model = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_b = axs.AxonInstance(axon_b_model)
    axon_b.add_current_clamp(
        position=(length_um / 2.0) * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=0.2),
    )

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert {axon_result.diagnostics["dispatch_method"] for axon_result in result} == {
        "batch-single-cable"
    }
    assert [axon_result.record_indices for axon_result in result] == [(5,), (5,)]


def test_pool_dispatch_keeps_incompatible_axons_scalar():
    axon_a = _hh_axon(nx=11, amp_nA=0.4, y_um=20.0, z_um=30.0)
    axon_b = _hh_axon(nx=13, amp_nA=0.2, y_um=60.0, z_um=10.0)

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert [axon_result.diagnostics["dispatch_method"] for axon_result in result] == [
        "scalar",
        "scalar",
    ]
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 1), (2, 1)]


def test_pool_dispatch_batches_compatible_double_cable_axons():
    axon_a = _passive_double_cable_axon(amp_nA=0.1)
    axon_b = _passive_double_cable_axon(amp_nA=0.2)

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert {axon_result.diagnostics["dispatch_method"] for axon_result in result} == {
        "batch-double-cable"
    }
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 1), (2, 1)]


def test_pool_dispatch_pads_compatible_double_cable_axons():
    axon_a = _passive_double_cable_axon(amp_nA=0.1, compartments=11)
    axon_b = _passive_double_cable_axon(amp_nA=0.2, compartments=13)

    plan = build_dispatch_plan([axon_a, axon_b])
    assert len(plan.groups) == 1
    assert plan.groups[0].has_padding
    assert plan.groups[0].nx == 13

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.voltage(),
    )

    assert {axon_result.diagnostics["dispatch_method"] for axon_result in result} == {
        "parameter-batch-double-cable"
    }
    assert {axon_result.diagnostics["dispatch_has_padding"] for axon_result in result} == {
        True
    }
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 11), (2, 13)]


def test_dispatch_plan_groups_shifted_mrg_double_cable_rows():
    axons = [
        _mrg_axon(diameter_um=10.0, amp_nA=0.1, x_shift_um=0.0),
        _mrg_axon(diameter_um=10.0, amp_nA=0.2, x_shift_um=80.0),
        _mrg_axon(diameter_um=10.0, amp_nA=0.3, x_shift_um=160.0),
    ]

    plan = dispatch_plan_module.build_dispatch_plan(axons)

    assert len(plan.groups) == 1
    assert plan.groups[0].mode == "double"
    assert plan.groups[0].size == len(axons)
    assert plan.groups[0].batch_kind == "parameter-double-cable"


def test_dispatch_plan_cache_reuses_stable_simulation_instances(monkeypatch):
    axons = [
        _passive_double_cable_axon(amp_nA=0.1 + 0.01 * index)
        for index in range(3)
    ]
    first = dispatch_plan_module.build_dispatch_plan(axons)

    def fail_solver_rebuild(_simulation):
        raise AssertionError("stable pool should reuse the cached dispatch plan")

    monkeypatch.setattr(dispatch_plan_module, "build_solver_axon", fail_solver_rebuild)

    second = dispatch_plan_module.build_dispatch_plan(axons)

    assert second is first


def test_dispatch_plan_parameter_batches_mrg_diameter_sweep():
    axons = [
        _mrg_axon(diameter_um=diameter_um, amp_nA=0.1)
        for diameter_um in (4.0, 10.0, 20.0)
    ]

    plan = build_dispatch_plan(axons)

    assert len(plan.groups) == 1
    group = plan.groups[0]
    assert group.mode == "double"
    assert group.size == 3
    assert group.batch_kind == "parameter-double-cable"
    assert group.has_padding


def test_double_cable_shape_bucketing_is_internal_opt_in(monkeypatch):
    axons = [
        _mrg_axon(diameter_um=diameter_um, amp_nA=0.1)
        for diameter_um in (4.0, 10.0, 20.0)
    ]
    group = build_dispatch_plan(axons).groups[0]

    monkeypatch.delenv("AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING", raising=False)
    assert shape_bucketing.double_cable_kernel_group(group) is group

    monkeypatch.setenv("AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING", "1")
    kernel_group = shape_bucketing.double_cable_kernel_group(group)

    assert kernel_group is not group
    assert kernel_group.size >= group.size
    assert kernel_group.nx >= group.nx
    assert kernel_group.items[: group.size] == group.items
    assert kernel_group.items[-1] == group.items[-1]


def test_gated_leak_stack_initializes_gated_compartment_gates_from_model(monkeypatch):
    monkeypatch.delenv("AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING", raising=False)
    axons = [_mrg_axon(diameter_um=10.0, amp_nA=0.1)]
    group = build_dispatch_plan(axons).groups[0]

    fast_stack = runtime_preparation.try_stack_gated_leak_membrane_from_group(
        group,
        target_nx=group.nx,
        dtype_local=group.items[0].solver_axon.dtype,
        solver_options=None,
    )

    assert fast_stack is not None
    gated_gate_count = fast_stack.backend.gated_gate_count
    gated_mask = fast_stack.gates0_rows[0, :, gated_gate_count + 2].astype(bool)
    expected = np.asarray(
        fast_stack.backend.gated_model.init_gates(
            np.asarray([float(getattr(axons[0], "v_init", 0.0))], dtype=np.float32)
        )
    )[0]
    actual = fast_stack.gates0_rows[0, gated_mask, :gated_gate_count]
    np.testing.assert_allclose(
        actual,
        np.broadcast_to(expected, actual.shape),
        rtol=1e-6,
        atol=1e-7,
    )


def test_double_cable_mrg_membrane_stack_uses_structural_gated_leak_backend(monkeypatch):
    from axonscope.backends.jax.membrane_program import JaxMembraneProgram

    monkeypatch.delenv("AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING", raising=False)
    axons = [
        _mrg_axon(diameter_um=diameter_um, amp_nA=0.1)
        for diameter_um in (4.0, 10.0, 20.0)
    ]
    group = build_dispatch_plan(axons).groups[0]
    fast_stack = runtime_preparation.try_stack_gated_leak_membrane_from_group(
        group,
        target_nx=group.nx,
        dtype_local=group.items[0].solver_axon.dtype,
        solver_options=None,
    )

    runtime_caches.clear_batch_runtime_caches()
    runtime = runtime_preparation.prepare_batch_runtime(
        group,
        tsim_ms=0.05,
        dt_ms=0.01,
        solver_options=None,
        mode="double",
        include_extracellular=True,
        include_area=True,
    )

    assert fast_stack is not None
    assert fast_stack.source == "solver_axon_membrane_models"
    assert type(runtime.membrane.backend).__name__ == "GatedLeakStackMembraneBackend"
    assert runtime.membrane.gates0.shape == (len(axons), group.nx, 7)
    assert runtime.membrane.backend.n_gates_max == 7
    assert runtime.membrane.membrane is runtime.membrane.backend.gated_model
    assert isinstance(runtime.membrane.backend.gated_model, JaxMembraneProgram)


def test_pool_dispatch_parameter_batched_mrg_matches_scalar_rows():
    axons = [
        _mrg_axon(diameter_um=4.0, amp_nA=0.1, x_shift_um=0.0),
        _mrg_axon(diameter_um=10.0, amp_nA=0.2, x_shift_um=120.0),
    ]

    batched = _run_simulation(
        axons,
        duration=0.05 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    )
    scalar = [
        _run_simulation(
            [axon],
            duration=0.05 * axs.ms,
            dt=0.01 * axs.ms,
            recording=axs.Recording.voltage(),
        )[0]
        for axon in axons
    ]

    assert {result.diagnostics["dispatch_method"] for result in batched} == {
        "parameter-batch-double-cable"
    }
    assert {result.diagnostics["dispatch_has_padding"] for result in batched} == {True}
    assert [result.Vm.shape for result in batched] == [result.Vm.shape for result in scalar]
    for batched_row, scalar_row in zip(batched, scalar, strict=True):
        np.testing.assert_allclose(
            np.asarray(batched_row.Vm),
            np.asarray(scalar_row.Vm),
            atol=1e-2,
            rtol=0.0,
        )


def test_run_pool_returns_internal_dispatch_results():
    axon = _hh_axon(nx=11, amp_nA=0.1, y_um=12.0, z_um=34.0)

    result = run_pool(
        [axon],
        tsim_ms=0.1,
        dt_ms=0.05,
    )

    assert result[0].index == 0
    assert result[0].simulation is axon
    assert result[0].axon is axon.axon
    assert np.asarray(result[0].Vm).shape == (2, 11)


def test_run_pool_observer_only_keeps_one_compact_cohort_record():
    axons = [
        _hh_axon(nx=11, amp_nA=0.4, y_um=12.0, z_um=34.0),
        _hh_axon(nx=11, amp_nA=0.5, y_um=12.0, z_um=34.0),
    ]
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    result = run_pool(
        axons,
        tsim_ms=0.1,
        dt_ms=0.05,
        batch_options=BatchOptions.none(),
        observers=(activation,),
    )

    assert len(result) == 1
    assert isinstance(result[0], DispatchCohortRecord)
    assert result[0].indices == (0, 1)
    assert result[0].Vm is None
    assert result[0].observations is not None
    assert result[0].observations[axs.VM_RASTER_OBSERVATION_KEY].words.shape == (2, 1, 1, 1)


def test_run_pool_observer_only_batches_singleton_groups():
    axons = [
        _hh_axon(nx=11, amp_nA=0.4, y_um=12.0, z_um=34.0),
        _hh_axon(nx=13, amp_nA=0.5, y_um=22.0, z_um=44.0),
    ]
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    result = run_pool(
        axons,
        tsim_ms=0.1,
        dt_ms=0.05,
        batch_options=BatchOptions.none(),
        observers=(activation,),
    )

    assert len(result) == 2
    assert all(isinstance(row, DispatchCohortRecord) for row in result)
    assert [row.indices for row in result] == [(0,), (1,)]
    assert [row.method for row in result] == ["batch-single-cable", "batch-single-cable"]
    assert [row.Vm for row in result] == [None, None]
    assert all(row.observations is not None for row in result)


def test_run_pool_double_cable_observer_only_keeps_one_compact_cohort_record(
):
    axons = [
        _passive_double_cable_axon(amp_nA=0.1),
        _passive_double_cable_axon(amp_nA=0.2),
    ]
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    result = run_pool(
        axons,
        tsim_ms=0.1,
        dt_ms=0.05,
        batch_options=BatchOptions.none(),
        observers=(activation,),
    )

    assert len(result) == 1
    assert isinstance(result[0], DispatchCohortRecord)
    assert result[0].method == "batch-double-cable"
    assert result[0].indices == (0, 1)
    assert result[0].Vm is None
    assert result[0].observations is not None
    assert result[0].observations[axs.VM_RASTER_OBSERVATION_KEY].words.shape == (2, 1, 1, 1)


def test_run_pool_double_cable_observer_only_batches_singleton_groups():
    axons = [
        _passive_double_cable_axon(amp_nA=0.1, compartments=11),
        _mrg_axon(diameter_um=10.0, amp_nA=0.2, nodes=3),
    ]
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    result = run_pool(
        axons,
        tsim_ms=0.1,
        dt_ms=0.05,
        batch_options=BatchOptions.none(),
        observers=(activation,),
    )

    assert len(result) == 2
    assert all(isinstance(row, DispatchCohortRecord) for row in result)
    assert [row.indices for row in result] == [(0,), (1,)]
    assert [row.method for row in result] == ["batch-double-cable", "batch-double-cable"]
    assert [row.Vm for row in result] == [None, None]
    assert all(row.observations is not None for row in result)


def test_run_pool_double_cable_observer_uses_factorized_footprint_vstim():
    stimulus = Stimulus.pulse(
        start=0.0 * axs.ms,
        duration=0.05 * axs.ms,
        amplitude=10e-6,
    )
    electrode = PointSourceElectrode(
        x=50.0 * axs.um,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    axons = [
        _passive_double_cable_axon(amp_nA=0.1, compartments=11),
        _passive_double_cable_axon(amp_nA=0.2, compartments=13),
    ]
    for axon in axons:
        axon.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                axon.layout.position_values(unit=axs.um) * axs.um,
                stimulus=stimulus,
                sigma=0.3 * axs.S_per_m,
            )
        )

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    axs.enable_benchmark(
        "/tmp/axonscope-double-factorized-vstim-test",
        print_summary=False,
        save=False,
    )
    try:
        result = run_pool(
            axons,
            tsim_ms=0.1,
            dt_ms=0.05,
            batch_options=BatchOptions.none(),
            observers=(activation,),
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert isinstance(result[0], DispatchCohortRecord)
    assert result[0].Vm is None
    assert result[0].observations is not None
    raster = result[0].observations[axs.VM_RASTER_OBSERVATION_KEY]
    assert raster.batch_size == len(axons)
    assert raster.words.shape[0] == len(axons)
    assert report is not None
    extracellular_events = [
        event for event in report.events if event.name == "inputs.extracellular"
    ]
    assert len(extracellular_events) == 1
    metadata = extracellular_events[0].metadata
    assert metadata["input_format"] == "factorized_footprint"
    assert metadata["dense_vstim_avoided"] is True
    assert "vstim_mid" not in metadata
    assert "vstim_previous" not in metadata

    enqueue_events = [event for event in report.events if event.name == "kernel.enqueue"]
    assert len(enqueue_events) == 1
    enqueue_metadata = enqueue_events[0].metadata
    assert enqueue_metadata["mode"] == "double"
    assert enqueue_metadata["recording_mode"] == "none"

    dispatch_events = [event for event in report.events if event.name == "kernel.dispatch_jax"]
    assert len(dispatch_events) == 1
    assert dispatch_events[0].parent_event_id == enqueue_events[0].event_id
    dispatch_metadata = dispatch_events[0].metadata
    assert dispatch_metadata["mode"] == "double"
    assert dispatch_metadata["observer"] == "vm_raster"
    assert dispatch_metadata["factorized_vext"] is True

    finalize_events = [
        event for event in report.events if event.name == "kernel.finalize_observer"
    ]
    assert len(finalize_events) == 1
    assert finalize_events[0].parent_event_id == enqueue_events[0].event_id

    group_events = [event for event in report.events if event.name == "dispatch.group.total"]
    assert len(group_events) == 1
    group_metadata = group_events[0].metadata
    components = group_metadata["memory_estimate_components_nbytes"]
    assert group_metadata["has_padding"] is True
    assert group_metadata["memory_estimate_extracellular_format"] == "factorized_footprint"
    assert components["vm_output"] == 0
    assert components["vstim_mid"] < group_metadata["memory_estimate_vstim_dense_equivalent_nbytes"]
    assert components["vstim_previous"] < 2 * 11 * 8


def test_run_pool_single_cable_observer_uses_rank_k_factorized_vstim_for_multi_drive():
    def with_second_drive(axon):
        stimulation = axon.extracellular_stimulation
        assert stimulation is not None
        first = stimulation.drives[0]
        second = axs.ExtracellularDrive(
            id=axs.DriveId("second_point_source"),
            footprint=first.footprint,
            stimulus=Stimulus.pulse(
                start=0.01 * axs.ms,
                duration=0.03 * axs.ms,
                amplitude=5e-6,
            ),
        )
        axon.add_extracellular_stimulation(
            stimulation=axs.ExtracellularStimulation([first, second]),
            replace=True,
        )
        return axon

    axons = [
        with_second_drive(_hh_axon(nx=11, amp_nA=0.1)),
        with_second_drive(_hh_axon(nx=11, amp_nA=0.2)),
    ]
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )

    axs.enable_benchmark(
        "/tmp/axonscope-single-rank-k-factorized-vstim-test",
        print_summary=False,
        save=False,
    )
    try:
        result = run_pool(
            axons,
            tsim_ms=0.1,
            dt_ms=0.05,
            batch_options=BatchOptions.none(),
            observers=(activation,),
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert result[0].Vm is None
    assert result[0].observations is not None
    assert report is not None
    extracellular_events = [
        event for event in report.events if event.name == "inputs.extracellular"
    ]
    assert len(extracellular_events) == 1
    metadata = extracellular_events[0].metadata
    assert metadata["input_format"] == "factorized_footprint"
    assert metadata["vstim_factorized_rank"] == 2
    assert metadata["dense_vstim_avoided"] is True
    assert "vstim_mid" not in metadata


def test_scalar_retained_vm_emits_standard_hotpath_spans():
    axon = _hh_axon(nx=11, amp_nA=0.1)

    axs.enable_benchmark(
        "/tmp/axonscope-scalar-hotpath-span-test",
        print_summary=False,
        save=False,
    )
    try:
        result = run_pool(
            [axon],
            tsim_ms=0.1,
            dt_ms=0.05,
            batch_options=BatchOptions.center(),
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert result[0].Vm is not None
    assert report is not None
    names = [event.name for event in report.events]
    for name in (
        "runtime.prepare",
        "inputs.positions",
        "observer.plan",
        "inputs.intracellular",
        "inputs.extracellular",
        "kernel.enqueue",
        "kernel.wait",
        "results.split_batch",
    ):
        assert name in names

    scalar_events = [
        event
        for event in report.events
        if event.name in {"kernel.enqueue", "kernel.wait", "inputs.extracellular"}
    ]
    assert scalar_events
    assert all(event.metadata["route"] == "scalar" for event in scalar_events)

    extracellular_event = next(
        event for event in report.events if event.name == "inputs.extracellular"
    )
    metadata = extracellular_event.metadata
    assert metadata["input_format"] == "dense_precomputed"
    assert metadata["extracellular_stimulation_count"] == 1
    assert "vstim_mid" in metadata

    result_event = next(
        event for event in report.events if event.name == "results.split_batch"
    )
    assert result_event.metadata["recording_mode"] == "center"


def test_double_cable_batch_extracellular_stack_matches_row_runtime():
    axons = [
        _passive_double_cable_axon(amp_nA=0.1, compartments=11),
        _passive_double_cable_axon(amp_nA=0.2, compartments=13),
    ]
    group = build_dispatch_plan(axons).groups[0]
    dtype_local = prepare_solver_runtime(
        axons[0],
        tsim_ms=0.1,
        dt_ms=0.05,
        include_extracellular=True,
    ).membrane.dtype

    stacked = runtime_preparation.stack_extracellular_runtime(
        group,
        dtype_local=dtype_local,
    )

    def pad_space(values, *, mode):
        arr = np.asarray(values)
        pad_count = int(group.nx) - int(arr.shape[0])
        if pad_count == 0:
            return arr
        if mode == "edge":
            pad_values = np.broadcast_to(arr[-1], (pad_count,)).astype(arr.dtype, copy=False)
        else:
            pad_values = np.zeros((pad_count,), dtype=arr.dtype)
        return np.concatenate([arr, pad_values], axis=0)

    def pad_edge(values):
        arr = np.asarray(values)
        pad_count = max(int(group.nx) - 1, 0) - int(arr.shape[0])
        if pad_count == 0:
            return arr
        return np.concatenate([arr, np.zeros((pad_count,), dtype=arr.dtype)], axis=0)

    expected_rows = []
    for item in group.items:
        cable = prepare_cable_runtime(
            item.solver_axon,
            dtype_local,
            include_area=True,
        )
        row = prepare_extracellular_runtime(item.solver_axon, dtype_local, cable)
        expected_rows.append(
            {
                "Cm_abs": pad_space(row.Cm_abs, mode="edge"),
                "Cx_abs": pad_space(row.Cx_abs, mode="edge"),
                "Gx_abs": pad_space(row.Gx_abs, mode="edge"),
                "Gax_e": pad_edge(row.Gax_e),
                "Gax_i": pad_edge(row.Gax_i),
                "left_i": pad_space(row.left_i, mode="zero"),
                "right_i": pad_space(row.right_i, mode="zero"),
                "left_e": pad_space(row.left_e, mode="zero"),
                "right_e": pad_space(row.right_e, mode="zero"),
            }
        )

    for field_name in (
        "Cm_abs",
        "Cx_abs",
        "Gx_abs",
        "Gax_e",
        "Gax_i",
        "left_i",
        "right_i",
        "left_e",
        "right_e",
    ):
        expected = np.stack(
            [row[field_name] for row in expected_rows],
            axis=0,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(stacked, field_name)),
            expected,
            rtol=1e-6,
            atol=1e-8,
        )


def test_batch_runtime_cache_reuses_equivalent_rebuilt_pool():
    def make_pool():
        return [
            _hh_axon(nx=11, amp_nA=0.1, y_um=20.0, z_um=30.0),
            _hh_axon(nx=11, amp_nA=0.2, y_um=20.0, z_um=30.0),
        ]

    runtime_caches.clear_batch_runtime_caches()
    axs.enable_benchmark(
        "/tmp/axonscope-structural-runtime-cache-test",
        print_summary=False,
        save=False,
    )
    try:
        run_pool(
            make_pool(),
            tsim_ms=0.1,
            dt_ms=0.05,
            batch_options=BatchOptions.center(),
        )
        run_pool(
            make_pool(),
            tsim_ms=0.1,
            dt_ms=0.05,
            batch_options=BatchOptions.center(),
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert report is not None
    runtime_events = [event for event in report.events if event.name == "runtime.prepare"]
    runtime_cache_events = [
        event.metadata.get("batch_runtime_cache") for event in runtime_events
    ]
    assert runtime_cache_events == ["miss", "hit"]


def test_batch_static_runtime_cache_reuses_equivalent_pool_with_new_time_grid():
    def make_pool():
        return [
            _hh_axon(nx=11, amp_nA=0.1, y_um=20.0, z_um=30.0),
            _hh_axon(nx=11, amp_nA=0.2, y_um=20.0, z_um=30.0),
        ]

    runtime_caches.clear_batch_runtime_caches()
    axs.enable_benchmark(
        "/tmp/axonscope-static-runtime-cache-test",
        print_summary=False,
        save=False,
    )
    try:
        first = run_pool(
            make_pool(),
            tsim_ms=0.1,
            dt_ms=0.05,
            batch_options=BatchOptions.center(),
        )
        second = run_pool(
            make_pool(),
            tsim_ms=0.2,
            dt_ms=0.05,
            batch_options=BatchOptions.center(),
        )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert [row.Vm.shape for row in first] == [(2, 1), (2, 1)]
    assert [row.Vm.shape for row in second] == [(4, 1), (4, 1)]
    assert report is not None
    runtime_events = [event for event in report.events if event.name == "runtime.prepare"]
    runtime_cache_events = [
        event.metadata.get("batch_runtime_cache") for event in runtime_events
    ]
    static_cache_events = [
        event.metadata.get("batch_static_runtime_cache") for event in runtime_events
    ]
    assert runtime_cache_events == ["miss", "miss"]
    assert static_cache_events == ["miss", "hit"]


def test_prepared_cohort_cache_refreshes_replaced_stimulus_rows():
    axon = _hh_axon(nx=11, amp_nA=0.1, y_um=20.0, z_um=30.0)
    group = build_dispatch_plan([axon]).groups[0]

    runtime_caches.clear_prepared_cohort_cache()
    first = runtime_preparation.prepared_cohort_for_group(group)
    stimulation = first.stimulations[0][0]
    drive = stimulation.drives[0]
    updated = stimulation.replace_drive(
        drive.id,
        stimulus=Stimulus.pulse(
            start=0.0 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=20e-6,
        ),
    )
    axon.add_extracellular_stimulation(stimulation=updated, replace=True)

    second = runtime_preparation.prepared_cohort_for_group(group)

    assert second is not first
    assert second.x_positions_m is first.x_positions_m
    assert second.stimulations == ((updated,),)
    second_peak = np.asarray(second.stimulations[0][0].drives[0].stimulus.y).max()
    first_peak = np.asarray(first.stimulations[0][0].drives[0].stimulus.y).max()
    assert second_peak > first_peak


def test_factorized_footprint_cache_survives_stimulus_replacement(tmp_path):
    axon = _hh_axon(nx=11, amp_nA=0.1, y_um=20.0, z_um=30.0)
    stimulation = axon.extracellular_stimulations[0]
    drive = stimulation.drives[0]
    updated = stimulation.replace_drive(
        drive.id,
        stimulus=Stimulus.pulse(
            start=0.0 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=20e-6,
        ),
    )

    input_batches._FOOTPRINT_CACHE.clear()
    axs.enable_benchmark(tmp_path, print_summary=False, save=False)
    try:
        with benchmark_span("inputs.extracellular"):
            first = build_factorized_vstim_midpoint_batch(
                axon,
                [(stimulation,), (stimulation,)],
                tsim_ms=0.1,
                dt_ms=0.05,
                dtype_local=np.float32,
            )
        with benchmark_span("inputs.extracellular"):
            second = build_factorized_vstim_midpoint_batch(
                axon,
                [(updated,), (updated,)],
                tsim_ms=0.1,
                dt_ms=0.05,
                dtype_local=np.float32,
            )
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        axs.disable_benchmark(print_summary=False, save=False)

    assert first is not None
    assert second is not None
    statuses = [
        event.metadata.get("vstim_footprint_cache")
        for event in report.events
        if event.name == "inputs.extracellular"
    ]
    assert statuses == ["miss", "hit"]
    np.testing.assert_allclose(
        np.asarray(second.footprint_mV_per_A),
        np.asarray(first.footprint_mV_per_A),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(materialize_factorized_extracellular_potential_batch(second)),
        2.0 * np.asarray(materialize_factorized_extracellular_potential_batch(first)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_batch_runtime_cache_separates_backend_context_scope():
    pool = [
        _hh_axon(nx=11, amp_nA=0.1, y_um=20.0, z_um=30.0),
        _hh_axon(nx=11, amp_nA=0.2, y_um=20.0, z_um=30.0),
    ]
    group = build_dispatch_plan(pool).groups[0]
    cpu_context = SimpleNamespace(
        policy=axs.ExecutionPolicy(
            runtime=axs.Runtime.JAX,
            device=axs.Device.cpu(),
            precision=axs.PrecisionPolicy.float32(),
        ),
        platform="cpu",
        device="cpu:0",
    )
    gpu_context = SimpleNamespace(
        policy=axs.ExecutionPolicy(
            runtime=axs.Runtime.JAX,
            device=axs.Device.gpu(0),
            precision=axs.PrecisionPolicy.float32(),
        ),
        platform="gpu",
        device="gpu:0",
    )

    runtime_caches.clear_batch_runtime_caches()
    first = runtime_preparation.prepare_batch_runtime(
        group,
        tsim_ms=0.1,
        dt_ms=0.05,
        solver_options=None,
        mode="single",
        include_extracellular=False,
        include_area=False,
        backend_context=cpu_context,
    )
    second = runtime_preparation.prepare_batch_runtime(
        group,
        tsim_ms=0.1,
        dt_ms=0.05,
        solver_options=None,
        mode="single",
        include_extracellular=False,
        include_area=False,
        backend_context=cpu_context,
    )
    third = runtime_preparation.prepare_batch_runtime(
        group,
        tsim_ms=0.1,
        dt_ms=0.05,
        solver_options=None,
        mode="single",
        include_extracellular=False,
        include_area=False,
        backend_context=gpu_context,
    )

    assert second is first
    assert third is not first


def test_dispatch_plan_preserves_pool_indices():
    axon_a = _hh_axon(nx=11, amp_nA=0.1, y_um=12.0, z_um=34.0)
    axon_b = _hh_axon(nx=11, amp_nA=0.2, y_um=56.0, z_um=78.0)

    plan = build_dispatch_plan([axon_a, axon_b])

    assert [item.index for item in plan.items] == [0, 1]
    assert tuple(index for group in plan.groups for index in group.pool_indices) == (0, 1)


def test_dispatch_plan_reuses_solver_axon_for_shared_model_instances():
    model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(model)
    axon_b = axs.AxonInstance(model)

    plan = build_dispatch_plan([axon_a, axon_b])

    assert plan.items[0].solver_axon is plan.items[1].solver_axon
    assert len(plan.groups) == 1
    assert plan.groups[0].size == 2


def test_dispatch_plan_parameter_batches_equal_nx_different_geometry():
    axon_a = _hh_axon(nx=11, amp_nA=0.1, y_um=12.0, z_um=34.0)
    axon_b_model = axs.axons.HodgkinHuxley(
        length=150.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_b = axs.AxonInstance(axon_b_model)
    axon_b.add_current_clamp(
        position=75.0 * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=0.1),
    )

    plan = build_dispatch_plan([axon_a, axon_b])

    assert len(plan.groups) == 1
    assert not plan.groups[0].geometry_shared
    assert plan.groups[0].batch_kind == "parameter-single-cable"


def test_pool_dispatch_parameter_batches_equal_nx_different_geometry():
    axon_a = _hh_axon(nx=11, amp_nA=0.1, y_um=12.0, z_um=34.0)
    axon_b_model = axs.axons.HodgkinHuxley(
        length=150.0 * axs.um,
        diameter=0.75 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_b = axs.AxonInstance(axon_b_model)
    axon_b.add_current_clamp(
        position=75.0 * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=0.1),
    )

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert {axon_result.diagnostics["dispatch_method"] for axon_result in result} == {
        "parameter-batch-single-cable"
    }
    assert {axon_result.diagnostics["dispatch_geometry_shared"] for axon_result in result} == {
        False
    }
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 1), (2, 1)]


def test_pool_dispatch_accepts_plain_progress(capsys):
    axon_a = _hh_axon(nx=11, amp_nA=0.1)
    axon_b = _hh_axon(nx=11, amp_nA=0.2)

    result = _run_simulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        progress="plain",
    )

    captured = capsys.readouterr()
    assert len(result) == 2
    assert "building dispatch plan" in captured.out
    assert "Dispatch progress" in captured.out
    assert "group   g0 1/1 batch-single-cable" in captured.out
    assert "route   g0 1/1  compatible batch route" in captured.out
    assert "prepare g0 1/1  runtime" in captured.out
    assert "batch   g0 1/1  recording plan" in captured.out
    assert "lower   g0 1/1  inputs" in captured.out
    assert "compiling JAX kernel if needed" in captured.out
    assert "solving JAX kernel" in captured.out
    assert "completed JAX kernel" in captured.out
    assert "result  g0 1/1  assemble batch output" in captured.out
    assert re.search(r"\d{2}:\d{2}:\d{2} .*building dispatch plan", captured.out)
    assert "Simulation run completed:" in captured.out
    assert "cold_start=" in captured.out
    assert "rss=" in captured.out or "memory=n/a" in captured.out


def test_plain_chunk_progress_is_throttled():
    assert progress_module._should_render_plain_chunk_progress(1, 60)
    assert progress_module._should_render_plain_chunk_progress(6, 60)
    assert not progress_module._should_render_plain_chunk_progress(7, 60)
    assert progress_module._should_render_plain_chunk_progress(60, 60)
    assert all(
        progress_module._should_render_plain_chunk_progress(done, 12)
        for done in range(1, 13)
    )


def test_pool_dispatch_plain_progress_reports_scalar_fallback(capsys):
    axon = _hh_axon(nx=11, amp_nA=0.1)

    result = _run_simulation(
        [axon],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        progress="plain",
    )

    captured = capsys.readouterr()
    assert len(result) == 1
    assert "route   g0 1/1  single row group" in captured.out
    assert "(scalar" in captured.out
    assert "single row group" in captured.out
    assert "compiling scalar kernel if needed" in captured.out
    assert "completed scalar kernel" in captured.out
    assert "assembled scalar rows" in captured.out


def test_simulation_inspection_mentions_parameter_batch():
    axon_a = _hh_axon(nx=11, amp_nA=0.1)
    axon_b_model = axs.axons.HodgkinHuxley(
        length=150.0 * axs.um,
        diameter=0.75 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_b = axs.AxonInstance(axon_b_model)
    axon_b.add_current_clamp(
        position=75.0 * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=0.1),
    )

    text = axs.AxonSimulation(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
    ).inspect().format()

    assert "parameter-batch" in text
    assert "geometry=parameterized" in text


def test_pool_vstim_batch_uses_sampled_point_source_stimulation():
    def axon_at(y_um: float):
        axon_model = axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
        axon = axs.AxonInstance(axon_model)
        electrode = PointSourceElectrode(
            x=50.0 * axs.um,
            y=0.0 * axs.um,
            z=0.0 * axs.um,
        )
        axon.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                axon_model.layout.position_values(unit=axs.um) * axs.um,
                stimulus=Stimulus.constant(10e-6, start=0.0 * axs.ms),
                sigma=0.3 * axs.S_per_m,
                axon_y=y_um * axs.um,
                axon_z=0.0 * axs.um,
            ),
        )
        return axon

    near = axon_at(10.0)
    far = axon_at(100.0)
    vstim = build_vstim_midpoint_batch(
        near,
        extracellular_stimulation_rows([near, far]),
        tsim_ms=0.1,
        dt_ms=0.05,
        x_positions_m=x_positions_batch_m([near, far]),
    )

    center = near.n_compartments // 2
    assert float(vstim[0, 0, center]) > float(vstim[1, 0, center])
    assert np.allclose(
        np.asarray(vstim[0, 0]),
        np.asarray(near.extracellular_potential_mV(0.025)),
    )
    assert np.allclose(
        np.asarray(vstim[1, 0]),
        np.asarray(far.extracellular_potential_mV(0.025)),
    )


def test_pool_vstim_batch_empty_context_rows_returns_zero_without_yz():
    axon_model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(axon_model)
    axon_b = axs.AxonInstance(axon_model)

    vstim = build_vstim_midpoint_batch(
        axon_a,
        [None, None],
        tsim_ms=0.1,
        dt_ms=0.05,
        x_positions_m=x_positions_batch_m([axon_a, axon_b]),
    )

    assert np.asarray(vstim).shape == (2, 2, 11)
    np.testing.assert_allclose(np.asarray(vstim), 0.0)


def test_intracellular_current_density_batch_uses_current_clamps():
    model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(model)
    axon_b = axs.AxonInstance(model)
    axon_a.add_current_clamp(
        position=50.0 * axs.um,
        current=Stimulus.constant(2.0 * axs.nA, start=0.0 * axs.ms),
    )
    axon_b.add_current_clamp(
        position=50.0 * axs.um,
        current=Stimulus.constant(-1.0 * axs.nA, start=0.0 * axs.ms),
    )
    solver_axon = build_solver_axon(axon_a)
    runtime = prepare_solver_runtime(
        axon_a,
        tsim_ms=0.1,
        dt_ms=0.05,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )

    batch = build_intracellular_current_density_batch(
        [axon_a, axon_b],
        runtime,
        solver_axons=[solver_axon, solver_axon],
    )

    expected = np.zeros((2, 2, 11), dtype=np.float32)
    idx = int(np.argmin(np.abs(np.asarray(solver_axon.x_um) - 50.0)))
    area_cm2 = (
        np.pi
        * np.asarray(solver_axon.diam_um)
        * 1e-4
        * np.asarray(solver_axon.compartment_lengths_um)
        * 1e-4
    )
    expected[0, :, idx] = 2.0e-3 / area_cm2[idx]
    expected[1, :, idx] = -1.0e-3 / area_cm2[idx]
    np.testing.assert_allclose(np.asarray(batch), expected)


def test_sparse_intracellular_current_density_batch_matches_dense_clamps():
    model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(model)
    axon_b = axs.AxonInstance(model)
    axon_a.add_current_clamp(
        position=50.0 * axs.um,
        current=Stimulus.constant(2.0 * axs.nA, start=0.0 * axs.ms),
    )
    axon_a.add_current_clamp(
        position=50.0 * axs.um,
        current=Stimulus.constant(0.5 * axs.nA, start=0.0 * axs.ms),
    )
    axon_b.add_current_clamp(
        position=30.0 * axs.um,
        current=Stimulus.constant(-1.0 * axs.nA, start=0.0 * axs.ms),
    )
    solver_axon = build_solver_axon(axon_a)
    runtime = prepare_solver_runtime(
        axon_a,
        tsim_ms=0.1,
        dt_ms=0.05,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )

    dense = build_intracellular_current_density_batch(
        [axon_a, axon_b],
        runtime,
        solver_axons=[solver_axon, solver_axon],
    )
    sparse = build_sparse_intracellular_current_density_batch(
        [axon_a, axon_b],
        runtime,
        solver_axons=[solver_axon, solver_axon],
    )

    assert sparse.density_mid.shape == (2, 2, 2)
    assert sparse.indices.shape == (2, 2)
    assert sparse.mask.shape == (2, 2)
    np.testing.assert_allclose(
        np.asarray(materialize_sparse_intracellular_current_density_batch(sparse)),
        np.asarray(dense),
    )


def test_sparse_intracellular_pulse_batch_matches_dense_clamps():
    model = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    axon_a = axs.AxonInstance(model)
    axon_b = axs.AxonInstance(model)
    axon_a.add_current_clamp(
        position=50.0 * axs.um,
        current=Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=0.5 * axs.nA,
        ),
    )
    axon_b.add_current_clamp(
        position=40.0 * axs.um,
        current=Stimulus.pulse(
            start=0.05 * axs.ms,
            duration=0.05 * axs.ms,
            amplitude=0.8 * axs.nA,
        ),
    )
    solver_axon = build_solver_axon(axon_a)
    runtime = prepare_solver_runtime(
        axon_a,
        tsim_ms=0.15,
        dt_ms=0.05,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
    )

    dense = build_intracellular_current_density_batch(
        [axon_a, axon_b],
        runtime,
        solver_axons=[solver_axon, solver_axon],
    )
    sparse = build_sparse_intracellular_current_density_batch(
        [axon_a, axon_b],
        runtime,
        solver_axons=[solver_axon, solver_axon],
    )

    assert sparse.density_mid.shape == (2, 3, 1)
    assert sparse.indices.shape == (2, 1)
    assert sparse.mask.shape == (2, 1)
    np.testing.assert_allclose(
        np.asarray(materialize_sparse_intracellular_current_density_batch(sparse)),
        np.asarray(dense),
    )
