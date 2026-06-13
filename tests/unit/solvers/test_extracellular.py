from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

import axonscope as axs
from axonscope import AxonSimulation
from axonscope.axons import Axon, Layout, Section
from axonscope.axons.myelinated import MRG
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.channel_models.passive import PassiveICM
from axonscope.stimulation import AnalyticalElectrode, AnalyticalExtracellularContext, PointSourceElectrode
from axonscope.stimulation import Stimulus
from axonscope.solvers.crank_nicholson import (
    CrankNicholson,
)
from axonscope.solvers.experimental import (
    CrankNicholsonVStimForcing,
    CrankNicholson_unoptimized,
)
from axonscope.utils import units
from axonscope.solvers.common import simulation_step_count
from axonscope.solvers.kernels import DoubleCableKernel
from axonscope.solvers.runtime import prepare_solver_runtime


ALL_SOLVERS = [
    CrankNicholson_unoptimized(),
    CrankNicholson(),
]


def _passive_axon(*, L: float, d: float, Nx: int, v_init=None) -> AxonSimulation:
    if v_init is None:
        v_init = -70.0 * axs.mV
    return AxonSimulation(
        Axon(
            layout=Layout.single_uniform(
                Section(
                    "axon",
                    membrane=PassiveICM(Rm=1e4, EL=-70.0),
                    diameter=units.Q_(d, "micrometer"),
                    Ra=units.Q_(100.0, "ohm * centimeter"),
                    Cm=units.Q_(1.0, "microfarad / centimeter ** 2"),
                ),
                length=units.Q_(L, "micrometer"),
                compartments=Nx,
            ),
            v_init=v_init,
        )
    )


class _UniformFieldElectrode(AnalyticalElectrode):
    def __init__(self, footprint_v_per_a: float) -> None:
        self.footprint_v_per_a = float(footprint_v_per_a)

    def footprint(self, x_positions_m, *, sigma_S_m):
        return np.full(np.asarray(x_positions_m, dtype=float).shape, self.footprint_v_per_a, dtype=float)


def _context(electrode: AnalyticalElectrode, stimulus: Stimulus, *, sigma=0.3):
    return AnalyticalExtracellularContext(electrodes=[electrode.with_stimulus(stimulus)], sigma=sigma)


def test_add_extracellular_context_requires_context():
    ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6)
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)

    ax.add_extracellular_context(context=_context(electrode, stim))

    assert ax.use_extracellular is True
    vext = np.asarray(ax.extracellular_potential_mV(0.31))
    assert np.max(np.abs(vext)) > 0.0


def test_add_extracellular_context_accepts_pre_attached_electrode():
    ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6)
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)

    ax.add_extracellular_context(context=_context(electrode, stim))

    assert ax.use_extracellular is True
    vext = np.asarray(ax.extracellular_potential_mV(0.31))
    assert np.max(np.abs(vext)) > 0.0


def test_extracellular_context_accumulates_multiple_electrodes():
    ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    x0_m = 200e-6
    y0_m = 100e-6
    z0_m = 100e-6
    sigma = 0.3
    t_probe = 0.31

    e1 = PointSourceElectrode(x0_m=x0_m, y0_m=y0_m, z0_m=z0_m)
    s1 = Stimulus.constant(-10e-6, start=0.0)
    e2 = PointSourceElectrode(x0_m=x0_m, y0_m=y0_m, z0_m=z0_m)
    s2 = Stimulus.constant(-15e-6, start=0.0)

    x_m = np.asarray(ax.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    r = np.sqrt((x_m - x0_m) ** 2 + y0_m**2 + z0_m**2)
    fp = 1.0 / (4.0 * np.pi * sigma * np.maximum(r, 1e-12))
    expected_mV = (
        s1.evaluate([t_probe])[0]
        + s2.evaluate([t_probe])[0]
    ) * fp * 1e3

    ax.add_extracellular_context(
        context=AnalyticalExtracellularContext(
            electrodes=[
                e1.with_stimulus(s1),
                e2.with_stimulus(s2),
            ],
            sigma=sigma,
        ),
        replace=True,
    )
    got_mV = np.asarray(ax.extracellular_potential_mV(t_probe))

    assert np.allclose(got_mV, expected_mV, rtol=1e-6, atol=1e-6)


def test_add_extracellular_context_rejects_second_context_without_replace():
    ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    e1 = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6)
    e2 = PointSourceElectrode(x0_m=200e-6, y0_m=150e-6, z0_m=100e-6)
    stimulus = Stimulus.constant(-10e-6, start=0.0)

    ax.add_extracellular_context(context=_context(e1, stimulus))

    with pytest.raises(ValueError, match="one extracellular context"):
        ax.add_extracellular_context(context=_context(e2, stimulus))


def test_myelinated_vext_matches_analytic_point_source():
    ax = AxonSimulation(MRG(diameter=10.0 * axs.um, nodes=7))
    x0_um = float(ax.length / 2.0)
    sigma = 0.2
    amp_A = -80e-6

    electrode = PointSourceElectrode(
        x0_m=x0_um * 1e-6,
        y0_m=100e-6,
        z0_m=0.0,
    )
    ax.add_extracellular_context(
        context=_context(electrode, Stimulus.constant(amp_A, start=0.0), sigma=sigma),
        replace=True,
    )

    x_m = np.asarray(ax.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    r = np.sqrt((x_m - x0_um * 1e-6) ** 2 + (100e-6) ** 2)
    expected_mV = amp_A / (4.0 * np.pi * sigma * np.maximum(r, 1e-12)) * 1e3
    got_mV = np.asarray(ax.extracellular_potential_mV(0.5), dtype=float)

    assert np.allclose(got_mV, expected_mV, rtol=1e-6, atol=1e-6)


def test_myelinated_extracellular_stimulus_has_nonzero_effect():
    dt = 0.005
    tsim = 2.0

    ax_on = AxonSimulation(MRG(diameter=10.0 * axs.um, nodes=7))
    x0_um = float(ax_on.length / 2.0)
    electrode = PointSourceElectrode(x0_m=x0_um * 1e-6, y0_m=100e-6, z0_m=0.0)
    stim_on = Stimulus.biphasic(
        start=0.6,
        cathodic_amplitude=80e-6,
        cathodic_duration=0.08,
        anodic_amplitude=20e-6,
        interphase=0.04,
    )
    ax_on.add_extracellular_context(context=_context(electrode, stim_on, sigma=0.2), replace=True)

    ax_off = AxonSimulation(MRG(diameter=10.0 * axs.um, nodes=7))
    x0_off_um = float(ax_off.length / 2.0)
    electrode_off = PointSourceElectrode(x0_m=x0_off_um * 1e-6, y0_m=100e-6, z0_m=0.0)
    stim_off = Stimulus.constant(0.0, start=0.0)
    ax_off.add_extracellular_context(context=_context(electrode_off, stim_off, sigma=0.2), replace=True)

    solver = CrankNicholson()
    vm_on = np.asarray(solver.solve(ax_on, tsim=tsim, dt=dt).Vm)
    vm_off = np.asarray(solver.solve(ax_off, tsim=tsim, dt=dt).Vm)

    max_delta = float(np.max(np.abs(vm_on - vm_off)))
    assert max_delta > 1.0


def test_uniform_constant_vext_with_matching_veinit_does_not_charge_xc():
    ax = _passive_axon(L=100.0, d=1.0, Nx=5, v_init=-70.0 * axs.mV)
    ax.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((ax.n_compartments,), 1e9, dtype=float),
        xg_S_per_cm2=np.full((ax.n_compartments,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((ax.n_compartments,), 0.1, dtype=float),
        use_extracellular=True,
        Veinit=50.0,
    )
    ax.add_extracellular_context(
        context=_context(_UniformFieldElectrode(1000.0), Stimulus.constant(50e-6, start=0.0)),
        replace=True,
    )

    res = CrankNicholson().solve(ax, tsim=0.5, dt=0.01)

    np.testing.assert_allclose(np.asarray(res.Vm), -70.0, rtol=0.0, atol=2e-3)


def test_public_solver_uses_vstim_forcing_for_single_cable_extracellular():
    def build_axon() -> AxonSimulation:
        ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
        ax.add_current_clamp(position_um=200.0,
            current=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
        )
        ax.set_extracellular_layer(Veinit=20.0)
        ax.add_extracellular_context(
            context=_context(_UniformFieldElectrode(1000.0), Stimulus.constant(20e-6, start=0.0)),
            replace=True,
        )
        return ax

    forced = CrankNicholsonVStimForcing().solve(build_axon(), tsim=1.0, dt=0.01)
    reference = CrankNicholson().solve(build_axon(), tsim=1.0, dt=0.01)

    np.testing.assert_allclose(np.asarray(forced.Vm), np.asarray(reference.Vm), atol=0.0, rtol=0.0)


def test_public_vstim_default_is_close_to_double_cable_for_unmyelinated_nrv_defaults():
    def build_axon() -> AxonSimulation:
        ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
        electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6)
        stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
        ax.add_extracellular_context(context=_context(electrode, stim), replace=True)
        ax.add_current_clamp(position_um=200.0,
            current=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
        )
        return ax

    ax_ref = build_axon()
    runtime = prepare_solver_runtime(
        ax_ref,
        tsim_ms=1.2,
        dt_ms=0.01,
        include_extracellular=True,
        include_area=True,
        precompute_intracellular=True,
    )
    reference = DoubleCableKernel(runtime=runtime, Veinit_mV=float(ax_ref.Veinit)).run()
    forced = CrankNicholson().solve(build_axon(), tsim=1.2, dt=0.01)

    np.testing.assert_allclose(np.asarray(forced.Vm), np.asarray(reference.Vm), atol=5e-1, rtol=0.0)


def test_single_cable_vstim_default_uses_inline_forcing():
    ax = _passive_axon(L=100.0, d=1.0, Nx=11, v_init=-70.0 * axs.mV)
    ax.add_extracellular_context(
        context=_context(_UniformFieldElectrode(1000.0), Stimulus.constant(20e-6, start=0.0)),
        replace=True,
    )

    res = CrankNicholson().solve(ax, tsim=0.1, dt=0.01)

    assert res.Vm.shape == (10, ax.n_compartments)
    assert np.isfinite(np.asarray(res.Vm)).all()


def test_vstim_forcing_rejects_double_cable_axons():
    ax = MRG(diameter=10.0 * axs.um, nodes=5)

    with pytest.raises(ValueError, match="single-cable solver"):
        CrankNicholsonVStimForcing().solve(ax, tsim=1.0, dt=0.01)


def test_myelinated_uses_inline_double_cable_solver():
    ax = AxonSimulation(MRG(diameter=10.0 * axs.um, nodes=5))
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.x_nodes_um[center_node])
    ax.add_current_clamp(position_um=pos_um,
        current=Stimulus.pulse(start=0.5, duration=0.05, amplitude=1.0),
    )

    res = CrankNicholson().solve(ax, tsim=1.0, dt=0.01)

    assert res.Vm.shape[1] == ax.n_compartments
    assert np.isfinite(np.asarray(res.Vm)).all()


def test_double_cable_kernel_matches_public_solver_path():
    ax = AxonSimulation(MRG(diameter=10.0 * axs.um, nodes=5))
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.x_nodes_um[center_node])
    ax.add_current_clamp(position_um=pos_um,
        current=Stimulus.pulse(start=0.5, duration=0.05, amplitude=1.0),
    )

    runtime = prepare_solver_runtime(
        ax,
        tsim_ms=1.0,
        dt_ms=0.01,
        include_extracellular=True,
        include_area=True,
    )
    direct = DoubleCableKernel(
        runtime=runtime,
        Veinit_mV=float(getattr(ax, "Veinit", 0.0)),
    ).run()
    public = CrankNicholson().solve(ax, tsim=1.0, dt=0.01)

    np.testing.assert_allclose(np.asarray(direct.t), np.asarray(public.t), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(direct.Vm), np.asarray(public.Vm), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("solver", ALL_SOLVERS, ids=lambda s: s.__class__.__name__)
def test_all_solvers_run_myelinated_with_extracellular(solver):
    ax = AxonSimulation(MRG(diameter=10.0 * axs.um, nodes=5))
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.x_nodes_um[center_node])
    ax.add_current_clamp(position_um=pos_um,
        current=Stimulus.pulse(start=0.5, duration=0.05, amplitude=1.5),
    )

    tsim = 4.0
    dt = 0.01
    res = solver.solve(ax, tsim=tsim, dt=dt)

    assert res.Vm.shape[0] == simulation_step_count(tsim, dt)
    assert res.Vm.shape[1] == ax.n_compartments
    vm = np.asarray(res.Vm)
    assert np.isfinite(vm).all()

    center_trace = vm[:, int(ax.node_indices[center_node])]
    peaks, _ = find_peaks(center_trace, height=0.0, distance=max(1, int(0.5 / dt)))
    # Anti-regression: this protocol should produce one AP only on center node.
    assert len(peaks) == 1
    late_mask = np.asarray(res.t) >= 2.0
    assert float(np.max(center_trace[late_mask])) < 0.0


@pytest.mark.parametrize("solver", ALL_SOLVERS, ids=lambda s: s.__class__.__name__)
def test_all_solvers_run_unmyelinated_with_extracellular_and_vext(solver):
    ax = AxonSimulation(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))

    xraxial = np.full((ax.n_compartments,), 1e8, dtype=float)
    xg = np.full((ax.n_compartments,), 1e-3, dtype=float)
    xc = np.full((ax.n_compartments,), 0.01, dtype=float)
    ax.set_extracellular_layer(
        xraxial_MOhm_per_cm=xraxial,
        xg_S_per_cm2=xg,
        xc_uF_per_cm2=xc,
        use_extracellular=True,
        Veinit=0.0,
    )

    electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6)
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    ax.add_extracellular_context(context=_context(electrode, stim), replace=True)

    ax.add_current_clamp(position_um=200.0,
        current=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    res = solver.solve(ax, tsim=1.2, dt=0.01)

    assert res.Vm.shape[0] == simulation_step_count(1.2, 0.01)
    assert res.Vm.shape[1] == ax.n_compartments
    assert np.isfinite(np.asarray(res.Vm)).all()
