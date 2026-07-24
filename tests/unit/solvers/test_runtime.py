from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import axonfleet as axs
from axonfleet import AxonInstance
from axonfleet.axons import Axon, Layout, Section
from axonfleet.axons import HodgkinHuxley, MRG
from axonfleet.analytical import PointSourceElectrode
from axonfleet.membranes.model import ensure_membrane_model
from axonfleet.runtime.jax.preparation.base import (
    _membrane_runtime_cache_key,
    prepare_cable_runtime,
    prepare_extracellular_runtime,
    prepare_membrane_runtime,
    prepare_simulation_grid,
    prepare_solver_runtime,
)
from axonfleet.runtime.jax.preparation import caches as runtime_caches
from axonfleet.runtime.jax.preparation.caches import (
    get_batched_static_array,
    store_batched_static_array,
)
from axonfleet.runtime.jax.membranes.program import JaxMembraneProgram
from axonfleet.runtime.solver_axon import build_solver_axon
from axonfleet.stimulation import Stimulus


def test_batched_static_array_cache_rejects_recycled_source_identity():
    runtime_caches._BATCH_RUNTIME_CACHE.clear()
    runtime_caches._BATCH_STATIC_RUNTIME_CACHE.clear()
    runtime_caches._BATCHED_STATIC_ARRAY_CACHE.clear()
    source = jnp.asarray([1.0, 2.0])
    other_source = jnp.asarray([3.0, 4.0])
    cached = jnp.asarray([[1.0, 2.0]])
    key = ("identity-key", id(source))

    store_batched_static_array(key, cached, sources=(source,))

    assert get_batched_static_array(key, sources=(source,)) is cached
    assert get_batched_static_array(key, sources=(other_source,)) is None


def _attach_point_source_stimulation(
    axon: AxonInstance,
    electrode: PointSourceElectrode,
    stimulus: Stimulus,
    *,
    sigma=0.2 * axs.S_per_m,
    replace: bool = True,
) -> None:
    axon.add_extracellular_stimulation(
        stimulation=axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit=axs.um) * axs.um,
            stimulus=stimulus,
            sigma=sigma,
        ),
        replace=replace,
    )


def test_prepare_simulation_grid_ends_at_requested_duration():
    grid = prepare_simulation_grid(tsim_ms=1.0, dt_ms=0.2, dtype_local=np.float32)

    assert grid.Nt == 5
    np.testing.assert_allclose(np.asarray(grid.t_vec_ms), [0.2, 0.4, 0.6, 0.8, 1.0])


def test_prepare_simulation_grid_rejects_partial_final_step():
    with pytest.raises(ValueError, match="integer multiple"):
        prepare_simulation_grid(tsim_ms=1.0, dt_ms=0.3, dtype_local=np.float32)


def test_prepare_solver_runtime_collects_static_solver_arrays():
    axon = AxonInstance(HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC))
    axon.add_current_clamp(position=150.0 * axs.um,
        current=Stimulus.pulse(start=0.2 * axs.ms, duration=0.1 * axs.ms, amplitude=1.0),
    )

    runtime = prepare_solver_runtime(axon, tsim_ms=1.0, dt_ms=0.1)

    assert runtime.grid.Nt == 10
    assert runtime.grid.t_vec_ms.shape == (10,)
    assert runtime.membrane.Nx == 11
    assert runtime.membrane.Vm0_mV.shape == (11,)
    assert runtime.membrane.gates0.shape[0] == 11
    assert runtime.cable.lower.shape == (11,)
    assert runtime.cable.area_cm2.shape == (11,)

def test_prepare_membrane_runtime_reuses_static_runtime_for_same_signature():
    axon = HodgkinHuxley(
        length=300.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )

    first = prepare_membrane_runtime(axon)
    second = prepare_membrane_runtime(axon)

    assert second is first


def test_prepare_rattay_initial_state_uses_generic_membrane_backend():
    axon = axs.axons.RattayAberham(
        length=500.0 * axs.um,
        diameter=0.8 * axs.um,
        compartments=21,
    )
    runtime = prepare_membrane_runtime(axon)

    assert isinstance(runtime.membrane, JaxMembraneProgram)
    assert runtime.observable_names["currents"] == ("I_na", "I_k", "I_l")
    assert runtime.observable_names["conductances"] == (
        "rattay_aberham.g_na",
        "rattay_aberham.g_k",
        "rattay_aberham.g_l",
        "passive.g_l",
    )
    np.testing.assert_allclose(
        np.asarray(runtime.gates0),
        np.asarray(runtime.membrane.init_gates(runtime.Vm0_mV)),
        rtol=2e-6,
        atol=2e-7,
    )


def test_prepare_mrg_initial_state_uses_model_ir_numpy_for_heterogeneous_backend(tmp_path):
    axon = axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=3,
        v_init=-78.123 * axs.mV,
    )

    report = None
    axs.enable_benchmark(tmp_path, print_summary=False, save=False)
    try:
        runtime = prepare_membrane_runtime(axon)
        report = axs.disable_benchmark(print_summary=False, save=False)
    finally:
        if report is None:
            axs.disable_benchmark(print_summary=False, save=False)

    init_events = [
        event
        for event in report.events
        if event.name == "runtime.prepare.membrane_init"
    ]
    assert init_events
    assert (
        init_events[-1].metadata["membrane_init_source"]
        == "heterogeneous_model_ir_numpy"
    )
    np.testing.assert_allclose(
        np.asarray(runtime.gates0),
        np.asarray(runtime.backend.init_gates(runtime.Vm0_mV)),
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(np.asarray(runtime.background_current), 0.0)


def test_prepare_solver_runtime_reuses_batch_safe_runtime_with_existing_solver_axon():
    axon = AxonInstance(
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    electrode = PointSourceElectrode(x=150.0 * axs.um, z=100.0 * axs.um)
    _attach_point_source_stimulation(
        axon,
        electrode,
        Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=10.0 * axs.uA,
        ),
        sigma=0.3 * axs.S_per_m,
    )
    solver_axon = build_solver_axon(axon)

    first = prepare_solver_runtime(
        axon,
        tsim_ms=1.0,
        dt_ms=0.1,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
    )
    _attach_point_source_stimulation(
        axon,
        electrode,
        Stimulus.pulse(
            start=0.2 * axs.ms,
            duration=0.1 * axs.ms,
            amplitude=20.0 * axs.uA,
        ),
        sigma=0.3 * axs.S_per_m,
    )
    second = prepare_solver_runtime(
        axon,
        tsim_ms=1.0,
        dt_ms=0.1,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
    )
    different_grid = prepare_solver_runtime(
        axon,
        tsim_ms=1.1,
        dt_ms=0.1,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
    )

    assert second is first
    assert different_grid is not first


def test_prepare_membrane_runtime_keeps_initial_voltage_in_cache_key():
    axon_a = HodgkinHuxley(
        length=300.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
        v_init=-67.5 * axs.mV,
    )
    axon_b = HodgkinHuxley(
        length=300.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
        v_init=-65.0 * axs.mV,
    )

    runtime_a = prepare_membrane_runtime(axon_a)
    runtime_b = prepare_membrane_runtime(axon_b)

    assert runtime_b is not runtime_a
    assert np.asarray(runtime_a.Vm0_mV)[0] == pytest.approx(-67.5)
    assert np.asarray(runtime_b.Vm0_mV)[0] == pytest.approx(-65.0)


def test_membrane_dtype_participates_in_static_and_runtime_cache_identity():
    membrane32 = axs.membranes.Passive(
        Rm=1e4,
        EL=-70.0,
        dtype=np.float32,
    )
    membrane64 = axs.membranes.Passive(
        Rm=1e4,
        EL=-70.0,
        dtype=np.float64,
    )
    assert ensure_membrane_model(membrane32)._static_signature() != ensure_membrane_model(
        membrane64
    )._static_signature()

    axon32 = Axon(
        layout=Layout.single_uniform(
            Section(
                "axon",
                membrane=membrane32,
                diameter=1.0 * axs.um,
            ),
            length=100.0 * axs.um,
            compartments=5,
        )
    )
    axon64 = Axon(
        layout=Layout.single_uniform(
            Section(
                "axon",
                membrane=membrane64,
                diameter=1.0 * axs.um,
            ),
            length=100.0 * axs.um,
            compartments=5,
        )
    )
    solver32 = build_solver_axon(axon32)
    solver64 = build_solver_axon(axon64)

    assert solver32.dtype == np.dtype("float32")
    assert solver64.dtype == np.dtype("float64")
    assert _membrane_runtime_cache_key(axon32, solver32) != (
        _membrane_runtime_cache_key(axon64, solver64)
    )


def test_prepare_cable_runtime_reuses_static_geometry_runtime():
    axon = HodgkinHuxley(
        length=300.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    solver_axon = build_solver_axon(axon)
    membrane = prepare_membrane_runtime(axon, solver_axon=solver_axon)

    first = prepare_cable_runtime(solver_axon, membrane.dtype)
    second = prepare_cable_runtime(solver_axon, membrane.dtype)
    without_area = prepare_cable_runtime(solver_axon, membrane.dtype, include_area=False)

    assert second is first
    assert without_area is not first


def test_prepare_extracellular_runtime_reuses_static_layer_runtime():
    axon = AxonInstance(
        MRG(
            diameter=10.0 * axs.um,
            nodes=3,
        )
    )
    solver_axon = build_solver_axon(axon)
    membrane = prepare_membrane_runtime(axon, solver_axon=solver_axon)
    cable = prepare_cable_runtime(solver_axon, membrane.dtype)

    first = prepare_extracellular_runtime(solver_axon, membrane.dtype, cable)
    second = prepare_extracellular_runtime(solver_axon, membrane.dtype, cable)

    assert second is first
