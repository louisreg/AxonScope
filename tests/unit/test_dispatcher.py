import numpy as np

import axonscope as axs
from axonscope.stimulation import AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.backends.jax.input_batches import (
    build_intracellular_current_density_batch,
    build_vstim_midpoint_batch,
)
from axonscope.dispatcher import build_dispatch_plan, run_pool
from axonscope.dispatcher import describe_dispatch_plan
from axonscope.dispatcher.runtime_batches import (
    extracellular_context_rows,
    x_positions_batch_m,
)
from axonscope.solvers.axon_runtime import build_solver_axon
from axonscope.solvers.runtime import prepare_solver_runtime
from axonscope.stimulation import Stimulus


def _context(electrode: PointSourceElectrode, stimulus: Stimulus):
    return AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )


def _hh_axon(*, nx: int, amp_nA: float, y_um: float = 0.0, z_um: float = 20.0):
    length_um = 100.0
    axon_model = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.5 * axs.um,
        compartments=nx,
        celsius=6.3 * axs.degC,
    )
    axon = axs.AxonInstance(axon_model)
    axon.set_position(y=y_um * axs.um, z=z_um * axs.um)
    axon.add_current_clamp(
        position=(length_um / 2.0) * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=amp_nA),
    )
    electrode = PointSourceElectrode(
        x=50.0 * axs.um,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    axon.add_extracellular_context(
        context=_context(
            electrode,
            Stimulus.pulse(start=0.0 * axs.ms, duration=0.05 * axs.ms, amplitude=10e-6)
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
):
    axon_model = axs.axons.MRG(
        diameter=diameter_um * axs.um,
        nodes=nodes,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
    )
    axon = axs.AxonInstance(axon_model)
    axon.add_current_clamp(
        position=float(axon_model.layout.position_values(unit=axs.um)[0]) * axs.um,
        current=Stimulus.pulse(start=0.02 * axs.ms, duration=0.04 * axs.ms, amplitude=amp_nA),
    )
    return axon


def test_pool_public_api_uses_axon_positions_and_contexts():
    axon_a = _hh_axon(nx=11, amp_nA=0.4, y_um=20.0, z_um=30.0)
    axon_b = _hh_axon(nx=13, amp_nA=0.2, y_um=60.0, z_um=10.0)

    result = axs.simulate_pool(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
    )

    assert len(result) == 2
    assert [axon_result.diagnostics["pool_index"] for axon_result in result] == [0, 1]
    assert [axon_result.Vm.shape for axon_result in result] == [(2, 1), (2, 1)]
    assert result[0].simulation.y_um == 20.0
    assert result[1].simulation.z_um == 10.0
    assert np.asarray([np.max(np.asarray(axon_result.Vm)) for axon_result in result]).shape == (2,)


def test_pool_dispatch_batches_compatible_axons_with_recording_filter():
    axon_a = _hh_axon(nx=11, amp_nA=0.4, y_um=20.0, z_um=30.0)
    axon_b = _hh_axon(nx=11, amp_nA=0.2, y_um=60.0, z_um=10.0)

    result = axs.simulate_pool(
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

    result = axs.simulate_pool(
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

    result = axs.simulate_pool(
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

    result = axs.simulate_pool(
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

    result = axs.simulate_pool(
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


def test_pool_dispatch_parameter_batched_mrg_matches_scalar_rows():
    axons = [
        _mrg_axon(diameter_um=4.0, amp_nA=0.1),
        _mrg_axon(diameter_um=20.0, amp_nA=0.2),
    ]

    batched = axs.simulate_pool(
        axons,
        duration=0.05 * axs.ms,
        dt=0.01 * axs.ms,
        recording=axs.Recording.voltage(),
    )
    scalar = [
        axs.simulate_pool(
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
            atol=1e-6,
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
    axon_a = axs.AxonInstance(model, y=0.0 * axs.um)
    axon_b = axs.AxonInstance(model, y=50.0 * axs.um)

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

    result = axs.simulate_pool(
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

    result = axs.simulate_pool(
        [axon_a, axon_b],
        duration=0.1 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        progress="plain",
    )

    captured = capsys.readouterr()
    assert len(result) == 2
    assert "Dispatch progress" in captured.out
    assert "group 0" in captured.out


def test_dispatch_plan_description_mentions_parameter_batch():
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

    text = describe_dispatch_plan([axon_a, axon_b])

    assert "parameter-batch-single-cable" in text
    assert "batched" in text


def test_pool_vstim_batch_uses_global_yz_positions_for_point_sources():
    def axon_at(y_um: float):
        axon_model = axs.axons.HodgkinHuxley(
            length=100.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
        axon = axs.AxonInstance(axon_model)
        axon.set_position(y=y_um * axs.um, z=0.0 * axs.um)
        electrode = PointSourceElectrode(
            x=50.0 * axs.um,
            y=0.0 * axs.um,
            z=0.0 * axs.um,
        )
        axon.add_extracellular_context(
            context=_context(electrode, Stimulus.constant(10e-6, start=0.0 * axs.ms)),
        )
        return axon

    near = axon_at(10.0)
    far = axon_at(100.0)
    vstim = build_vstim_midpoint_batch(
        near,
        extracellular_context_rows([near, far]),
        tsim_ms=0.1,
        dt_ms=0.05,
        x_positions_m=x_positions_batch_m([near, far]),
        axon_y_um=np.asarray([near.y_um, far.y_um]),
        axon_z_um=np.asarray([near.z_um, far.z_um]),
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
