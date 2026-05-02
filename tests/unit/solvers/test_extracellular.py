from __future__ import annotations

import importlib
import numpy as np
import pytest
from scipy.signal import find_peaks

from axonscope.axons.base import AxonBase
from axonscope.axons.myelinated import MRG
from axonscope.axons.unmyelinated import HodgkinHuxley
from axonscope.channel_models.passive import PassiveICM
from axonscope.electrodes import Electrode, PointSourceElectrode
from axonscope.stimulus import Stimulus
from axonscope.stimulus_eval import evaluate_stimulus_numpy
from axonscope.solvers.euler import Euler
from axonscope.solvers.crank_nicholson import (
    CrankNicholson,
)
from axonscope.solvers.experimental import (
    CrankNicholsonVStimForcing,
    CrankNicholson_unoptimized,
    CrankNicholsonSemiImplicit,
    CrankNicholsonImplicit,
)
from axonscope.solvers.kernels import DoubleCableKernel
from axonscope.solvers.runtime import prepare_solver_runtime


ALL_SOLVERS = [
    CrankNicholson_unoptimized(),
    CrankNicholson(),
    CrankNicholsonSemiImplicit(),
    CrankNicholsonImplicit(n_newton=3),
]


class _UniformFieldElectrode(Electrode):
    def __init__(self, footprint_v_per_a: float) -> None:
        self.footprint_v_per_a = float(footprint_v_per_a)

    def footprint(self, x_positions_m):
        return np.full(np.asarray(x_positions_m, dtype=float).shape, self.footprint_v_per_a, dtype=float)


def test_add_extracellular_context_with_electrode_and_stimulus():
    ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)
    electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6, sigma_S_m=0.3)
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)

    ax.add_extracellular_context(electrode, stim)

    assert ax.use_extracellular is True
    vext = np.asarray(ax.extracellular_potential_mV(0.31))
    assert np.max(np.abs(vext)) > 0.0


def test_add_extracellular_context_accumulates_multiple_contexts():
    ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)
    x0_m = 200e-6
    y0_m = 100e-6
    z0_m = 100e-6
    sigma = 0.3
    t_probe = 0.31

    e1 = PointSourceElectrode(x0_m=x0_m, y0_m=y0_m, z0_m=z0_m, sigma_S_m=sigma)
    s1 = Stimulus.constant(-10e-6, start=0.0)
    e2 = PointSourceElectrode(x0_m=x0_m, y0_m=y0_m, z0_m=z0_m, sigma_S_m=sigma)
    s2 = Stimulus.constant(-15e-6, start=0.0)

    x_m = np.asarray(ax.x, dtype=float) * 1e-6
    r = np.sqrt((x_m - x0_m) ** 2 + y0_m**2 + z0_m**2)
    fp = 1.0 / (4.0 * np.pi * sigma * np.maximum(r, 1e-12))
    expected_mV = (
        evaluate_stimulus_numpy(s1, [t_probe])[0]
        + evaluate_stimulus_numpy(s2, [t_probe])[0]
    ) * fp * 1e3

    ax.add_extracellular_context(e1, s1, replace=True)
    ax.add_extracellular_context(e2, s2, replace=False)
    got_mV = np.asarray(ax.extracellular_potential_mV(t_probe))

    assert np.allclose(got_mV, expected_mV, rtol=1e-6, atol=1e-6)


def test_myelinated_vext_matches_analytic_point_source():
    ax = MRG(d=10.0, nodes=7)
    x0_um = float(ax.L / 2.0)
    sigma = 0.2
    amp_A = -80e-6

    electrode = PointSourceElectrode(
        x0_m=x0_um * 1e-6,
        y0_m=100e-6,
        z0_m=0.0,
        sigma_S_m=sigma,
    )
    ax.add_extracellular_context(electrode, Stimulus.constant(amp_A, start=0.0), replace=True)

    x_m = np.asarray(ax.x, dtype=float) * 1e-6
    r = np.sqrt((x_m - x0_um * 1e-6) ** 2 + (100e-6) ** 2)
    expected_mV = amp_A / (4.0 * np.pi * sigma * np.maximum(r, 1e-12)) * 1e3
    got_mV = np.asarray(ax.extracellular_potential_mV(0.5), dtype=float)

    assert np.allclose(got_mV, expected_mV, rtol=1e-6, atol=1e-6)


def test_myelinated_extracellular_stimulus_has_nonzero_effect():
    dt = 0.005
    tsim = 2.0

    ax_on = MRG(d=10.0, nodes=7)
    x0_um = float(ax_on.L / 2.0)
    electrode = PointSourceElectrode(x0_m=x0_um * 1e-6, y0_m=100e-6, z0_m=0.0, sigma_S_m=0.2)
    stim_on = Stimulus.biphasic(
        start=0.6,
        cathodic_amplitude=80e-6,
        cathodic_duration=0.08,
        anodic_amplitude=20e-6,
        interphase=0.04,
    )
    ax_on.add_extracellular_context(electrode, stim_on, replace=True)

    ax_off = MRG(d=10.0, nodes=7)
    x0_off_um = float(ax_off.L / 2.0)
    electrode_off = PointSourceElectrode(x0_m=x0_off_um * 1e-6, y0_m=100e-6, z0_m=0.0, sigma_S_m=0.2)
    stim_off = Stimulus.constant(0.0, start=0.0)
    ax_off.add_extracellular_context(electrode_off, stim_off, replace=True)

    solver = CrankNicholson()
    vm_on = np.asarray(solver.solve(ax_on, tsim=tsim, dt=dt).Vm)
    vm_off = np.asarray(solver.solve(ax_off, tsim=tsim, dt=dt).Vm)

    max_delta = float(np.max(np.abs(vm_on - vm_off)))
    assert max_delta > 1.0


def test_uniform_constant_vext_with_matching_veinit_does_not_charge_xc():
    ax = AxonBase(
        PassiveICM(Rm=1e4, EL=-70.0),
        d=1.0,
        Nx=5,
        L=100.0,
        Ra=100.0,
        Cm=1.0,
        Vinit=-70.0,
    )
    ax.set_extracellular_layer(
        xraxial_MOhm_per_cm=np.full((ax.Nx,), 1e9, dtype=float),
        xg_S_per_cm2=np.full((ax.Nx,), 1e-3, dtype=float),
        xc_uF_per_cm2=np.full((ax.Nx,), 0.1, dtype=float),
        use_extracellular=True,
        Veinit=50.0,
    )
    ax.add_extracellular_context(
        _UniformFieldElectrode(1000.0),
        Stimulus.constant(50e-6, start=0.0),
        replace=True,
    )

    res = CrankNicholson().solve(ax, tsim=0.5, dt=0.01)

    np.testing.assert_allclose(np.asarray(res.Vm), -70.0, rtol=0.0, atol=2e-3)


def test_public_solver_uses_vstim_forcing_for_single_cable_extracellular():
    def build_axon() -> HodgkinHuxley:
        ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)
        ax.insert_I_Clamp(
            position=200.0,
            stimulus=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
        )
        ax.set_extracellular_layer(Veinit=20.0)
        ax.add_extracellular_context(
            _UniformFieldElectrode(1000.0),
            Stimulus.constant(20e-6, start=0.0),
            replace=True,
        )
        return ax

    forced = CrankNicholsonVStimForcing().solve(build_axon(), tsim=1.0, dt=0.01)
    reference = CrankNicholson().solve(build_axon(), tsim=1.0, dt=0.01)

    np.testing.assert_allclose(np.asarray(forced.Vm), np.asarray(reference.Vm), atol=0.0, rtol=0.0)


def test_public_vstim_default_is_close_to_double_cable_for_unmyelinated_nrv_defaults():
    def build_axon() -> HodgkinHuxley:
        ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)
        electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6, sigma_S_m=0.3)
        stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
        ax.add_extracellular_context(electrode, stim, replace=True)
        ax.insert_I_Clamp(
            position=200.0,
            stimulus=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
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


def test_single_cable_vstim_default_bypasses_generic_extracellular_solver(monkeypatch):
    cn_mod = importlib.import_module("axonscope.solvers.crank_nicholson")

    def _fail_if_generic(*args, **kwargs):
        raise AssertionError("Single-cable extracellular default should use Vstim forcing.")

    monkeypatch.setattr(cn_mod, "_solve_with_extracellular_generic", _fail_if_generic)

    ax = AxonBase(
        PassiveICM(Rm=1e4, EL=-70.0),
        d=1.0,
        Nx=11,
        L=100.0,
        Ra=100.0,
        Cm=1.0,
        Vinit=-70.0,
    )
    ax.add_extracellular_context(
        _UniformFieldElectrode(1000.0),
        Stimulus.constant(20e-6, start=0.0),
        replace=True,
    )

    res = CrankNicholson().solve(ax, tsim=0.1, dt=0.01)

    assert res.Vm.shape == (10, ax.Nx)
    assert np.isfinite(np.asarray(res.Vm)).all()


def test_vstim_forcing_rejects_double_cable_axons():
    ax = MRG(d=10.0, nodes=5)

    with pytest.raises(ValueError, match="single-cable solver"):
        CrankNicholsonVStimForcing().solve(ax, tsim=1.0, dt=0.01)


def test_myelinated_prefers_inline_extracellular_solver(monkeypatch):
    cn_mod = importlib.import_module("axonscope.solvers.crank_nicholson")

    def _fail_if_generic(*args, **kwargs):
        raise AssertionError("MRG should bypass the generic extracellular helper.")

    monkeypatch.setattr(cn_mod, "_solve_with_extracellular_generic", _fail_if_generic)

    ax = MRG(d=10.0, nodes=5)
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.x[int(ax.node_indices[center_node])])
    ax.insert_I_Clamp(
        position=pos_um,
        stimulus=Stimulus.pulse(start=0.5, duration=0.05, amplitude=1.0),
    )

    res = CrankNicholson().solve(ax, tsim=1.0, dt=0.01)

    assert res.Vm.shape[1] == ax.Nx
    assert np.isfinite(np.asarray(res.Vm)).all()


def test_double_cable_kernel_matches_public_solver_path():
    ax = MRG(d=10.0, nodes=5)
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.x[int(ax.node_indices[center_node])])
    ax.insert_I_Clamp(
        position=pos_um,
        stimulus=Stimulus.pulse(start=0.5, duration=0.05, amplitude=1.0),
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


def test_euler_raises_on_extracellular():
    ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)
    ax.set_extracellular_layer(use_extracellular=True, Veinit=0.0)
    with pytest.raises(NotImplementedError, match="does not support extracellular coupling"):
        Euler().solve(ax, tsim=0.5, dt=0.01)


@pytest.mark.parametrize("solver", ALL_SOLVERS, ids=lambda s: s.__class__.__name__)
def test_all_solvers_run_myelinated_with_extracellular(solver):
    ax = MRG(d=10.0, nodes=5)
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.x[int(ax.node_indices[center_node])])
    ax.insert_I_Clamp(
        position=pos_um,
        stimulus=Stimulus.pulse(start=0.5, duration=0.05, amplitude=1.5),
    )

    tsim = 4.0
    dt = 0.01
    res = solver.solve(ax, tsim=tsim, dt=dt)

    assert res.Vm.shape[0] == int(np.ceil(tsim / dt))
    assert res.Vm.shape[1] == ax.Nx
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
    ax = HodgkinHuxley(L=400.0, d=0.5, Nx=41)

    xraxial = np.full((ax.Nx,), 1e8, dtype=float)
    xg = np.full((ax.Nx,), 1e-3, dtype=float)
    xc = np.full((ax.Nx,), 0.01, dtype=float)
    ax.set_extracellular_layer(
        xraxial_MOhm_per_cm=xraxial,
        xg_S_per_cm2=xg,
        xc_uF_per_cm2=xc,
        use_extracellular=True,
        Veinit=0.0,
    )

    electrode = PointSourceElectrode(x0_m=200e-6, y0_m=100e-6, z0_m=100e-6, sigma_S_m=0.3)
    stim = Stimulus.pulse(start=0.3, amplitude=20e-6, duration=0.1, baseline=0.0)
    ax.add_extracellular_context(electrode, stim, replace=True)

    ax.insert_I_Clamp(
        position=200.0,
        stimulus=Stimulus.pulse(start=0.4, duration=0.05, amplitude=0.8),
    )
    res = solver.solve(ax, tsim=1.2, dt=0.01)

    assert res.Vm.shape[0] == int(np.ceil(1.2 / 0.01))
    assert res.Vm.shape[1] == ax.Nx
    assert np.isfinite(np.asarray(res.Vm)).all()
