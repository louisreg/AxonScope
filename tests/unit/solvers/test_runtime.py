from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import axonscope as axs
from axonscope import AxonInstance
from axonscope.axons import Axon, Layout, Section
from axonscope.axons import HodgkinHuxley
from axonscope.analytical import PointSourceElectrode
from axonscope.solvers import RateTableConfig
from axonscope.stimulation import IntracellularContext
from axonscope.backends.jax.runtime import (
    compile_membrane_model,
    _membrane_runtime_cache_key,
    precompute_extracellular_potential_mV,
    prepare_cable_runtime,
    prepare_extracellular_runtime,
    prepare_membrane_runtime,
    prepare_simulation_grid,
    prepare_solver_runtime,
)
from axonscope.backends.jax.membrane_program import JaxMembraneProgram
from axonscope.solvers.axon_runtime import build_solver_axon
from axonscope.solvers import SolverOptions
from axonscope.stimulation import Stimulus
from axonscope.backends.jax.stimulation_runtime import (
    CompiledExtracellularStimulations,
    CompiledIntracellularContexts,
    compile_extracellular_stimulations,
    compile_intracellular_contexts,
)
from axonscope.utils import units


class _UnsupportedIntracellularContext(IntracellularContext):
    pass


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


def test_prepare_solver_runtime_collects_membrane_cable_and_stimulus_arrays():
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

    inj_on = np.asarray(runtime.stimulation.intracellular_current_density(0.25))
    inj_off = np.asarray(runtime.stimulation.intracellular_current_density(0.5))
    assert inj_on.max() > 0.0
    assert np.allclose(inj_off, 0.0)


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
    assert runtime.observable_names["conductances"] == ("g_na", "g_k", "g_l")
    np.testing.assert_allclose(
        np.asarray(runtime.gates0),
        np.asarray(runtime.membrane.init_gates(runtime.Vm0_mV)),
        rtol=2e-6,
        atol=2e-7,
    )


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
        precompute_intracellular=False,
        precompute_extracellular=False,
        compile_stimulation=False,
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
        precompute_intracellular=False,
        precompute_extracellular=False,
        compile_stimulation=False,
    )
    different_grid = prepare_solver_runtime(
        axon,
        tsim_ms=1.1,
        dt_ms=0.1,
        solver_axon=solver_axon,
        include_extracellular=False,
        include_area=False,
        precompute_intracellular=False,
        precompute_extracellular=False,
        compile_stimulation=False,
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
    ).to_membrane_model()
    membrane64 = axs.membranes.Passive(
        Rm=1e4,
        EL=-70.0,
        dtype=np.float64,
    ).to_membrane_model()
    assert membrane32._static_signature() != membrane64._static_signature()

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
    assert _membrane_runtime_cache_key(axon32, solver32, SolverOptions()) != (
        _membrane_runtime_cache_key(axon64, solver64, SolverOptions())
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
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((axon.n_compartments,), 1e8, dtype=float),
        xg_S_per_cm2=np.full((axon.n_compartments,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((axon.n_compartments,), 0.01, dtype=float),
        use_extracellular=True,
    )
    solver_axon = build_solver_axon(axon)
    membrane = prepare_membrane_runtime(axon, solver_axon=solver_axon)
    cable = prepare_cable_runtime(solver_axon, membrane.dtype)

    first = prepare_extracellular_runtime(solver_axon, membrane.dtype, cable)
    second = prepare_extracellular_runtime(solver_axon, membrane.dtype, cable)

    assert second is first


def test_compile_intracellular_contexts_returns_callable_collection():
    axon = AxonInstance(
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_current_clamp(
        position=150.0 * axs.um,
        current=Stimulus.pulse(start=0.2 * axs.ms, duration=0.1 * axs.ms, amplitude=1.0),
    )

    compiled = compile_intracellular_contexts(axon)

    assert isinstance(compiled, CompiledIntracellularContexts)
    assert compiled.n_compartments == axon.n_compartments
    assert np.asarray(compiled(0.25)).max() > 0.0


def test_intracellular_runtime_rejects_unknown_context_with_clear_error():
    axon = AxonInstance(
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_intracellular_context(context=_UnsupportedIntracellularContext())

    with pytest.raises(NotImplementedError, match="Only IntracellularCurrentClamp"):
        prepare_solver_runtime(axon, tsim_ms=1.0, dt_ms=0.1)


def test_compile_extracellular_stimulations_returns_callable_collection():
    axon = AxonInstance(
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    electrode = PointSourceElectrode(
        x=150e-6 * axs.m,
        y=100e-6 * axs.m,
        z=1000.0 * axs.um,
    )
    stim = Stimulus.pulse(start=0.2 * axs.ms, duration=0.1 * axs.ms, amplitude=10e-6)
    _attach_point_source_stimulation(axon, electrode, stim)

    compiled = compile_extracellular_stimulations(axon)

    assert isinstance(compiled, CompiledExtracellularStimulations)
    assert compiled.n_compartments == axon.n_compartments
    assert np.asarray(compiled(0.25)).max() > 0.0


def test_prepare_solver_runtime_applies_rate_table_config():
    axon = HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC)

    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=1.0,
        dt_ms=0.1,
        solver_options=SolverOptions(
            rate_table_config=RateTableConfig(step_mV=1.0),
        ),
    )

    assert runtime.membrane.membrane.rate_table_config == RateTableConfig(step_mV=1.0)


def test_precompute_extracellular_potential_matches_axon_method():
    axon = AxonInstance(HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC))
    electrode = PointSourceElectrode(
        x=150e-6 * axs.m,
        y=100e-6 * axs.m,
        z=1000.0 * axs.um,
    )
    stim = Stimulus.pulse(start=0.2 * axs.ms, duration=0.1 * axs.ms, amplitude=10e-6)
    _attach_point_source_stimulation(axon, electrode, stim)

    t_ms = np.asarray([0.1, 0.25, 0.5], dtype=float)
    sampled = np.asarray(precompute_extracellular_potential_mV(axon, t_ms))

    assert sampled.shape == (3, axon.n_compartments)
    assert np.allclose(sampled[0], np.asarray(axon.extracellular_potential_mV(0.1)))
    assert np.allclose(sampled[1], np.asarray(axon.extracellular_potential_mV(0.25)))
    assert np.allclose(sampled[2], np.asarray(axon.extracellular_potential_mV(0.5)))


def test_extracellular_potential_uses_sampled_point_source_offsets():
    axon = AxonInstance(HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC))
    electrode = PointSourceElectrode(
        x=125.0 * axs.um,
        y=10.0 * axs.um,
        z=20.0 * axs.um,
    )
    stim = Stimulus.constant(10e-6, start=0.0 * axs.ms)
    axon.add_extracellular_stimulation(
        stimulation=axs.analytical.point_source_stimulation(
            electrode,
            axon.layout.position_values(unit="micrometer") * axs.um,
            stimulus=stim,
            sigma=0.2 * axs.S_per_m,
            axon_y=40.0 * axs.um,
            axon_z=-10.0 * axs.um,
        ),
        replace=True,
    )

    x_m = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    r = np.sqrt(
        (x_m - 125e-6) ** 2
        + ((10.0 - 40.0) * 1e-6) ** 2
        + ((20.0 - (-10.0)) * 1e-6) ** 2
    )
    expected_mV = 10e-6 / (4.0 * np.pi * 0.2 * r) * 1e3

    got = np.asarray(precompute_extracellular_potential_mV(axon, np.asarray([0.0])))[0]

    assert np.allclose(got, expected_mV, rtol=1e-6, atol=1e-6)


def test_prepare_solver_runtime_precomputes_extracellular_step_potentials():
    axon = AxonInstance(
        Axon(
            layout=Layout.single_uniform(
                Section(
                    "axon",
                    membrane=axs.membranes.Passive(Rm=1e4, EL=-70.0),
                    diameter=units.Q_(1.0, "micrometer"),
                ),
                length=units.Q_(300.0, "micrometer"),
                compartments=11,
            ),
            v_init=-70.0 * axs.mV,
        )
    )
    axon.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((axon.n_compartments,), 1e8, dtype=float),
        xg_S_per_cm2=np.full((axon.n_compartments,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((axon.n_compartments,), 0.01, dtype=float),
        use_extracellular=True,
    )
    electrode = PointSourceElectrode(
        x=150e-6 * axs.m,
        y=100e-6 * axs.m,
        z=1000.0 * axs.um,
    )
    stim = Stimulus.pulse(start=0.2 * axs.ms, duration=0.1 * axs.ms, amplitude=10e-6)
    _attach_point_source_stimulation(axon, electrode, stim)

    runtime = prepare_solver_runtime(
        axon,
        tsim_ms=1.0,
        dt_ms=0.1,
        include_extracellular=True,
    )

    vext_mid = runtime.stimulation.extracellular_potential_mid_mV
    vext_initial_previous = runtime.stimulation.extracellular_potential_initial_previous_mV
    assert vext_mid is not None
    assert vext_initial_previous is not None
    assert vext_mid.shape == (runtime.grid.Nt, axon.n_compartments)
    assert vext_initial_previous.shape == (axon.n_compartments,)
    assert np.allclose(np.asarray(vext_mid[0]), np.asarray(axon.extracellular_potential_mV(0.05)))
    assert np.allclose(np.asarray(vext_initial_previous), np.asarray(axon.extracellular_potential_mV(-0.05)))
    assert np.allclose(np.asarray(vext_mid[2]), np.asarray(axon.extracellular_potential_mV(0.25)))
