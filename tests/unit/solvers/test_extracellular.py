from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

import axonfleet as axs
from axonfleet import AxonInstance
from axonfleet.axons import Axon, Layout, Section
from axonfleet.axons.myelinated import MRG
from axonfleet.axons.unmyelinated import HodgkinHuxley
from axonfleet.runtime.jax.inputs.extracellular import build_vstim_midpoint_batch
from axonfleet.analytical import PointSourceElectrode
from axonfleet.stimulation import Stimulus
from axonfleet.utils import units
from axonfleet.runtime.timebase import simulation_step_count


def _run_public_batch(axon: AxonInstance, *, tsim: float, dt: float, recording=None):
    return axs.AxonSimulation(
        axon,
        duration=tsim,
        dt=dt,
        recording=recording,
    ).run().single


def _passive_axon(*, L: float, d: float, Nx: int, v_init=None) -> AxonInstance:
    if v_init is None:
        v_init = -70.0 * axs.mV
    return AxonInstance(
        Axon(
            layout=Layout.single_uniform(
                Section(
                    "axon",
                    membrane=axs.membranes.Passive(Rm=1e4, EL=-70.0),
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


def _uniform_drive(
    axon: AxonInstance,
    *,
    footprint_v_per_a: float,
    stimulus: Stimulus,
    drive_id: str,
) -> axs.ExtracellularDrive:
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    return axs.ExtracellularDrive(
        id=axs.DriveId(drive_id),
        footprint=axs.ExtracellularFootprint.shared(
            values=np.full((axon.n_compartments,), float(footprint_v_per_a)),
            positions=positions,
        ),
        stimulus=stimulus,
    )


def _uniform_stimulation(
    axon: AxonInstance,
    *,
    footprint_v_per_a: float,
    stimulus: Stimulus,
    drive_id: str = "uniform",
) -> axs.ExtracellularStimulation:
    return axs.ExtracellularStimulation(
        [
            _uniform_drive(
                axon,
                footprint_v_per_a=footprint_v_per_a,
                stimulus=stimulus,
                drive_id=drive_id,
            )
        ]
    )


def _attach_uniform_stimulation(
    axon: AxonInstance,
    *,
    footprint_v_per_a: float,
    stimulus: Stimulus,
    replace: bool = True,
) -> None:
    axon.add_extracellular_stimulation(
        stimulation=_uniform_stimulation(
            axon,
            footprint_v_per_a=footprint_v_per_a,
            stimulus=stimulus,
        ),
        replace=replace,
    )


def _point_source_m(x_m: float, y_m: float, z_m: float) -> PointSourceElectrode:
    return PointSourceElectrode(x=x_m * axs.m, y=y_m * axs.m, z=z_m * axs.m)


def _attach_point_source_stimulation(
    axon: AxonInstance,
    electrode: PointSourceElectrode,
    stimulus: Stimulus,
    *,
    sigma=0.3 * axs.S_per_m,
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


def test_add_extracellular_stimulation_requires_typed_stimulation():
    ax = AxonInstance(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    stim = Stimulus.pulse(start=0.3 * axs.ms, amplitude=20e-6, duration=0.1 * axs.ms, baseline=0.0)

    with pytest.raises(TypeError, match="ExtracellularStimulation"):
        ax.add_extracellular_stimulation(stimulation=stim)

    ax.add_extracellular_stimulation(
        stimulation=_uniform_stimulation(
            ax,
            footprint_v_per_a=10.0,
            stimulus=stim,
        )
    )

    assert ax.use_extracellular is True
    vext = np.asarray(
        ax.extracellular_stimulation.evaluate([0.31], voltage_unit="millivolt")[0]
    )
    assert np.max(np.abs(vext)) > 0.0


def test_extracellular_stimulation_accumulates_multiple_drives():
    ax = AxonInstance(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    t_probe = 0.31

    s1 = Stimulus.constant(-10e-6, start=0.0 * axs.ms)
    s2 = Stimulus.constant(-15e-6, start=0.0 * axs.ms)

    expected_mV = (
        s1.evaluate([t_probe])[0] * 2.0
        + s2.evaluate([t_probe])[0] * 3.0
    ) * 1e3

    ax.add_extracellular_stimulation(
        stimulation=axs.ExtracellularStimulation(
            [
                _uniform_drive(
                    ax,
                    footprint_v_per_a=2.0,
                    stimulus=s1,
                    drive_id="first",
                ),
                _uniform_drive(
                    ax,
                    footprint_v_per_a=3.0,
                    stimulus=s2,
                    drive_id="second",
                ),
            ]
        ),
        replace=True,
    )
    got_mV = np.asarray(
        ax.extracellular_stimulation.evaluate([t_probe], voltage_unit="millivolt")[0]
    )

    assert np.allclose(got_mV, expected_mV, rtol=1e-6, atol=1e-6)


def test_point_source_vstim_footprint_cache_keeps_stimulus_amplitude_live():
    ax = AxonInstance(
        HodgkinHuxley(length=100.0 * axs.um, diameter=0.5 * axs.um, compartments=11)
    )
    electrode = PointSourceElectrode(
        x=50.0 * axs.um,
        z=100.0 * axs.um,
    )
    first_stimulus = (
        Stimulus.pulse(start=0.0 * axs.ms, duration=0.1 * axs.ms, amplitude=10.0 * axs.uA)
    )
    first_stimulation = axs.analytical.point_source_stimulation(
        electrode,
        ax.layout.position_values(unit=axs.um) * axs.um,
        stimulus=first_stimulus,
        sigma=0.3 * axs.S_per_m,
    )

    first = np.asarray(
        build_vstim_midpoint_batch(
            ax,
            [(first_stimulation,), (first_stimulation,)],
            tsim_ms=0.1,
            dt_ms=0.05,
            dtype_local=np.float32,
        )
    )
    second_stimulus = (
        Stimulus.pulse(start=0.0 * axs.ms, duration=0.1 * axs.ms, amplitude=20.0 * axs.uA)
    )
    second_stimulation = axs.analytical.point_source_stimulation(
        electrode,
        ax.layout.position_values(unit=axs.um) * axs.um,
        stimulus=second_stimulus,
        sigma=0.3 * axs.S_per_m,
    )
    second = np.asarray(
        build_vstim_midpoint_batch(
            ax,
            [(second_stimulation,), (second_stimulation,)],
            tsim_ms=0.1,
            dt_ms=0.05,
            dtype_local=np.float32,
        )
    )

    np.testing.assert_allclose(second, 2.0 * first, rtol=1e-6, atol=1e-6)


def test_add_extracellular_stimulation_rejects_second_without_replace():
    ax = AxonInstance(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
    stimulus = Stimulus.constant(-10e-6, start=0.0 * axs.ms)

    _attach_uniform_stimulation(ax, footprint_v_per_a=2.0, stimulus=stimulus)

    with pytest.raises(ValueError, match="one extracellular stimulation"):
        _attach_uniform_stimulation(
            ax,
            footprint_v_per_a=3.0,
            stimulus=stimulus,
            replace=False,
        )


def test_myelinated_vext_matches_analytic_point_source():
    ax = AxonInstance(MRG(diameter=10.0 * axs.um, nodes=7))
    x0_um = float(ax.length / 2.0)
    sigma_S_m = 0.2
    sigma = sigma_S_m * axs.S_per_m
    amp_A = -80e-6

    electrode = _point_source_m(x0_um * 1e-6, 100e-6, 0.0)
    _attach_point_source_stimulation(
        ax,
        electrode,
        Stimulus.constant(amp_A, start=0.0 * axs.ms),
        sigma=sigma,
    )

    x_m = np.asarray(ax.layout.position_values(unit="micrometer"), dtype=float) * 1e-6
    r = np.sqrt((x_m - x0_um * 1e-6) ** 2 + (100e-6) ** 2)
    expected_mV = amp_A / (4.0 * np.pi * sigma_S_m * np.maximum(r, 1e-12)) * 1e3
    got_mV = np.asarray(
        ax.extracellular_stimulation.evaluate([0.5], voltage_unit="millivolt")[0],
        dtype=float,
    )

    assert np.allclose(got_mV, expected_mV, rtol=1e-6, atol=1e-6)


def test_myelinated_extracellular_stimulus_has_nonzero_effect():
    dt = 0.005
    tsim = 2.0

    ax_on = AxonInstance(MRG(diameter=10.0 * axs.um, nodes=7))
    x0_um = float(ax_on.length / 2.0)
    electrode = _point_source_m(x0_um * 1e-6, 100e-6, 0.0)
    stim_on = Stimulus.biphasic(
        start=0.6 * axs.ms,
        cathodic_amplitude=80e-6,
        cathodic_duration=0.08 * axs.ms,
        anodic_amplitude=20e-6,
        interphase=0.04 * axs.ms,
    )
    _attach_point_source_stimulation(
        ax_on,
        electrode,
        stim_on,
        sigma=0.2 * axs.S_per_m,
    )

    ax_off = AxonInstance(MRG(diameter=10.0 * axs.um, nodes=7))
    x0_off_um = float(ax_off.length / 2.0)
    electrode_off = _point_source_m(x0_off_um * 1e-6, 100e-6, 0.0)
    stim_off = Stimulus.constant(0.0, start=0.0 * axs.ms)
    _attach_point_source_stimulation(
        ax_off,
        electrode_off,
        stim_off,
        sigma=0.2 * axs.S_per_m,
    )

    vm_on = np.asarray(_run_public_batch(ax_on, tsim=tsim, dt=dt).Vm)
    vm_off = np.asarray(_run_public_batch(ax_off, tsim=tsim, dt=dt).Vm)

    max_delta = float(np.max(np.abs(vm_on - vm_off)))
    assert max_delta > 1.0


def test_public_batch_path_uses_vstim_forcing_for_single_cable_extracellular():
    def build_axon(*, with_extracellular: bool = True) -> AxonInstance:
        ax = AxonInstance(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))
        ax.add_current_clamp(position=200.0 * axs.um,
            current=Stimulus.pulse(start=0.4 * axs.ms, duration=0.05 * axs.ms, amplitude=0.8),
        )
        if with_extracellular:
            _attach_uniform_stimulation(
                ax,
                footprint_v_per_a=1000.0,
                stimulus=Stimulus.constant(20e-6, start=0.0 * axs.ms),
            )
        return ax

    forced = _run_public_batch(build_axon(with_extracellular=True), tsim=1.0, dt=0.01)
    control = _run_public_batch(build_axon(with_extracellular=False), tsim=1.0, dt=0.01)

    assert forced.diagnostics["dispatch_method"] == "batch-single-cable"
    assert np.isfinite(np.asarray(forced.Vm)).all()
    assert float(np.max(np.abs(np.asarray(forced.Vm) - np.asarray(control.Vm)))) > 1e-6


def test_single_cable_vstim_default_uses_inline_forcing():
    ax = _passive_axon(L=100.0, d=1.0, Nx=11, v_init=-70.0 * axs.mV)
    _attach_uniform_stimulation(
        ax,
        footprint_v_per_a=1000.0,
        stimulus=Stimulus.constant(20e-6, start=0.0 * axs.ms),
    )

    res = _run_public_batch(ax, tsim=0.1, dt=0.01)

    assert res.Vm.shape == (10, ax.n_compartments)
    assert np.isfinite(np.asarray(res.Vm)).all()


def test_myelinated_uses_inline_double_cable_solver():
    ax = AxonInstance(MRG(diameter=10.0 * axs.um, nodes=5))
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.node_position_values(unit=axs.um)[center_node])
    ax.add_current_clamp(position=pos_um * axs.um,
        current=Stimulus.pulse(start=0.5 * axs.ms, duration=0.05 * axs.ms, amplitude=1.0),
    )

    res = _run_public_batch(ax, tsim=1.0, dt=0.01)

    assert res.Vm.shape[1] == ax.n_compartments
    assert np.isfinite(np.asarray(res.Vm)).all()


def test_public_batch_path_runs_myelinated_with_extracellular():
    ax = AxonInstance(MRG(diameter=10.0 * axs.um, nodes=5))
    center_node = int(ax.node_indices.shape[0] // 2)
    pos_um = float(ax.node_position_values(unit=axs.um)[center_node])
    ax.add_current_clamp(position=pos_um * axs.um,
        current=Stimulus.pulse(start=0.5 * axs.ms, duration=0.05 * axs.ms, amplitude=1.5),
    )

    tsim = 4.0
    dt = 0.01
    res = _run_public_batch(ax, tsim=tsim, dt=dt)

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


def test_public_batch_path_runs_unmyelinated_with_extracellular_and_vext():
    ax = AxonInstance(HodgkinHuxley(length=400.0 * axs.um, diameter=0.5 * axs.um, compartments=41))

    electrode = _point_source_m(200e-6, 100e-6, 100e-6)
    stim = Stimulus.pulse(start=0.3 * axs.ms, amplitude=20e-6, duration=0.1 * axs.ms, baseline=0.0)
    _attach_point_source_stimulation(ax, electrode, stim)

    ax.add_current_clamp(position=200.0 * axs.um,
        current=Stimulus.pulse(start=0.4 * axs.ms, duration=0.05 * axs.ms, amplitude=0.8),
    )
    res = _run_public_batch(ax, tsim=1.2, dt=0.01)

    assert res.Vm.shape[0] == simulation_step_count(1.2, 0.01)
    assert res.Vm.shape[1] == ax.n_compartments
    assert np.isfinite(np.asarray(res.Vm)).all()
