from __future__ import annotations

import numpy as np
import pytest

import axonscope as axs
from axonscope import AxonSimulation
from axonscope.axons import Axon, Layout, Section
from axonscope.axons import HodgkinHuxley
from axonscope.channel_models import RateTableConfig
from axonscope.channel_models.passive import PassiveICM
from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    IntracellularContext,
    PointSourceElectrode,
)
from axonscope.solvers.runtime import (
    precompute_extracellular_potential_mV,
    prepare_simulation_grid,
    prepare_solver_runtime,
)
from axonscope.solvers import SolverOptions
from axonscope.stimulation import Stimulus
from axonscope.stimulation.runtime import (
    CompiledExtracellularContexts,
    CompiledIntracellularContexts,
    compile_extracellular_contexts,
    compile_intracellular_contexts,
)
from axonscope.utils import units


class _UnsupportedIntracellularContext(IntracellularContext):
    pass


def _context(electrode: PointSourceElectrode, stimulus: Stimulus, *, sigma=0.2):
    return AnalyticalExtracellularContext(electrodes=[electrode.with_stimulus(stimulus)], sigma=sigma)


def test_prepare_simulation_grid_ends_at_requested_duration():
    grid = prepare_simulation_grid(tsim_ms=1.0, dt_ms=0.2, dtype_local=np.float32)

    assert grid.Nt == 5
    np.testing.assert_allclose(np.asarray(grid.t_vec_ms), [0.2, 0.4, 0.6, 0.8, 1.0])


def test_prepare_simulation_grid_rejects_partial_final_step():
    with pytest.raises(ValueError, match="integer multiple"):
        prepare_simulation_grid(tsim_ms=1.0, dt_ms=0.3, dtype_local=np.float32)


def test_prepare_solver_runtime_collects_membrane_cable_and_stimulus_arrays():
    axon = AxonSimulation(HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC))
    axon.add_current_clamp(position_um=150.0,
        current=Stimulus.pulse(start=0.2, duration=0.1, amplitude=1.0),
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


def test_compile_intracellular_contexts_returns_callable_collection():
    axon = AxonSimulation(
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    axon.add_current_clamp(
        position_um=150.0,
        current=Stimulus.pulse(start=0.2, duration=0.1, amplitude=1.0),
    )

    compiled = compile_intracellular_contexts(axon)

    assert isinstance(compiled, CompiledIntracellularContexts)
    assert compiled.n_compartments == axon.n_compartments
    assert np.asarray(compiled(0.25)).max() > 0.0


def test_intracellular_runtime_rejects_unknown_context_with_clear_error():
    axon = AxonSimulation(
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


def test_compile_extracellular_contexts_returns_callable_collection():
    axon = AxonSimulation(
        HodgkinHuxley(
            length=300.0 * axs.um,
            diameter=0.5 * axs.um,
            compartments=11,
            celsius=6.3 * axs.degC,
        )
    )
    electrode = PointSourceElectrode(
        x0_m=150e-6,
        y0_m=100e-6,
    )
    stim = Stimulus.pulse(start=0.2, duration=0.1, amplitude=10e-6)
    axon.add_extracellular_context(context=_context(electrode, stim), replace=True)

    compiled = compile_extracellular_contexts(axon)

    assert isinstance(compiled, CompiledExtracellularContexts)
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
    axon = AxonSimulation(HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC))
    electrode = PointSourceElectrode(
        x0_m=150e-6,
        y0_m=100e-6,
    )
    stim = Stimulus.pulse(start=0.2, duration=0.1, amplitude=10e-6)
    axon.add_extracellular_context(context=_context(electrode, stim), replace=True)

    t_ms = np.asarray([0.1, 0.25, 0.5], dtype=float)
    sampled = np.asarray(precompute_extracellular_potential_mV(axon, t_ms))

    assert sampled.shape == (3, axon.n_compartments)
    assert np.allclose(sampled[0], np.asarray(axon.extracellular_potential_mV(0.1)))
    assert np.allclose(sampled[1], np.asarray(axon.extracellular_potential_mV(0.25)))
    assert np.allclose(sampled[2], np.asarray(axon.extracellular_potential_mV(0.5)))


def test_extracellular_potential_uses_global_axon_position_for_point_source():
    axon = AxonSimulation(HodgkinHuxley(length=300.0 * axs.um, diameter=0.5 * axs.um, compartments=11, celsius=6.3 * axs.degC))
    axon.set_position(x_offset_um=25.0, y_um=40.0, z_um=-10.0)
    electrode = PointSourceElectrode(
        x_um=150.0,
        y_um=10.0,
        z_um=20.0,
    )
    stim = Stimulus.constant(10e-6, start=0.0)
    axon.add_extracellular_context(context=_context(electrode, stim), replace=True)

    x_m = (
        np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float) + 25.0
    ) * 1e-6
    r = np.sqrt(
        (x_m - 150e-6) ** 2
        + ((10.0 - 40.0) * 1e-6) ** 2
        + ((20.0 - (-10.0)) * 1e-6) ** 2
    )
    expected_mV = 10e-6 / (4.0 * np.pi * 0.2 * r) * 1e3

    got = np.asarray(precompute_extracellular_potential_mV(axon, np.asarray([0.0])))[0]

    assert np.allclose(got, expected_mV, rtol=1e-6, atol=1e-6)


def test_prepare_solver_runtime_precomputes_extracellular_step_potentials():
    axon = AxonSimulation(
        Axon(
            layout=Layout.single_uniform(
                Section(
                    "axon",
                    membrane=PassiveICM(Rm=1e4, EL=-70.0),
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
        x0_m=150e-6,
        y0_m=100e-6,
    )
    stim = Stimulus.pulse(start=0.2, duration=0.1, amplitude=10e-6)
    axon.add_extracellular_context(context=_context(electrode, stim), replace=True)

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
