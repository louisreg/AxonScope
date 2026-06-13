import numpy as np
import pytest

import axonscope as axs
from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    ExtracellularContext,
    NRVExtracellularContext,
)
from axonscope.stimulation import PointSourceElectrode, Stimulus
from axonscope.stimulation.runtime import (
    CompiledExtracellularContext,
    compile_extracellular_context,
)


def _context(electrode: PointSourceElectrode, stimulus: Stimulus, *, sigma=0.3):
    return AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=sigma,
    )


class _ConstantFootprintContext(ExtracellularContext):
    def footprint_for_electrode(
        self,
        electrode,
        x_positions_m,
        *,
        axon_y_um=0.0,
        axon_z_um=0.0,
    ):
        return np.full_like(np.asarray(x_positions_m, dtype=float), 3.0)


def test_point_source_footprint_matches_analytical_formula():
    x = np.array([0.0, 1.0e-3, 2.0e-3])
    electrode = PointSourceElectrode(x0_m=1.0e-3, y0_m=0.0, z0_m=1.0e-3)
    ctx = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3,
    )

    fp = ctx.footprint_for_electrode(electrode, x)

    r = np.sqrt((x - 1.0e-3) ** 2 + (1.0e-3) ** 2)
    expected = 1.0 / (4.0 * np.pi * 0.3 * r)
    assert np.allclose(fp, expected)


def test_point_source_public_um_coordinates_match_si_aliases():
    x_m = np.array([0.0, 1.0e-3, 2.0e-3])
    by_um = PointSourceElectrode(x_um=1000.0, z_um=1000.0)
    by_m = PointSourceElectrode(x0_m=1.0e-3, z0_m=1.0e-3)
    ctx = AnalyticalExtracellularContext(
        electrodes=[by_um.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3,
    )

    assert np.allclose(ctx.footprint_for_electrode(by_um, x_m), ctx.footprint_for_electrode(by_m, x_m))


def test_point_source_footprint_is_symmetric_around_electrode():
    x = np.array([-1.0e-3, 0.0, 1.0e-3])
    electrode = PointSourceElectrode(x0_m=0.0, z0_m=1.0e-3)
    ctx = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3,
    )

    fp = ctx.footprint_for_electrode(electrode, x)

    assert np.isclose(fp[0], fp[2])
    assert fp[1] > fp[0]


def test_point_source_min_distance_avoids_singularity():
    x = np.array([0.0])
    electrode = PointSourceElectrode(x0_m=0.0, y0_m=0.0, z0_m=0.0, min_distance_m=1.0e-6)
    ctx = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3,
    )

    fp = ctx.footprint_for_electrode(electrode, x)

    expected = 1.0 / (4.0 * np.pi * 0.3 * 1.0e-6)
    assert np.isfinite(fp[0])
    assert np.isclose(fp[0], expected)


def test_with_stimulus_returns_stimulated_copy():
    electrode = PointSourceElectrode(x0_m=0.0, z0_m=1.0e-3)
    stim = Stimulus.pulse(start=1.0, amplitude=1.0e-6, duration=1.0)

    returned = electrode.with_stimulus(stim)

    assert returned is not electrode
    assert electrode.stimulus is None
    assert returned.stimulus.y_unit == "ampere"
    assert np.allclose(returned.stimulus.y, stim.y)


def test_analytical_context_normalizes_sigma_units():
    electrode = PointSourceElectrode(x_um=0.0, stimulus=Stimulus.constant(0.0))
    ctx = AnalyticalExtracellularContext(electrodes=[electrode], sigma=0.3 * axs.S_per_m)

    assert np.isclose(ctx.sigma_S_m, 0.3)


def test_extracellular_context_evaluate_shape():
    x = np.linspace(0.0, 1.0e-3, 5)
    t = np.linspace(0.0, 3.0, 7)
    electrode = PointSourceElectrode(x0_m=0.5e-3, z0_m=1.0e-3)
    stim = Stimulus.pulse(start=1.0, amplitude=2.0e-6, duration=1.0)
    extra = _context(electrode, stim)

    Vext = extra.evaluate(x, t, position_unit="meter")

    assert Vext.shape == (len(t), len(x))


def test_extracellular_context_evaluate_values():
    x = np.array([0.0, 1.0e-3])
    t = np.array([0.5, 1.5, 2.5])
    electrode = PointSourceElectrode(x0_m=0.0, z0_m=1.0e-3)
    stim = Stimulus.pulse(start=1.0, amplitude=2.0e-6, duration=1.0)
    extra = _context(electrode, stim)

    Vext = extra.evaluate(x, t, position_unit="meter")
    fp = extra.footprint_for_electrode(electrode, x)

    assert np.allclose(Vext[0], 0.0)
    assert np.allclose(Vext[1], 2.0e-6 * fp)
    assert np.allclose(Vext[2], 0.0)


def test_context_evaluate_uses_attached_stimulus_with_units():
    x = np.array([0.0, 1000.0]) * axs.um
    electrode = PointSourceElectrode(x_um=0.0 * axs.um, z_um=1000.0 * axs.um)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0 * axs.uA, duration=1.0 * axs.ms)
    extra = _context(electrode, stim, sigma=0.3 * axs.S_per_m)

    vext_mV = extra.evaluate(x, np.array([1.5]) * axs.ms, voltage_unit=axs.mV)

    expected_mV = 2.0e-6 * extra.footprint_for_electrode(electrode, np.array([0.0, 1.0e-3])) * 1e3
    assert np.allclose(vext_mV[0], expected_mV)


def test_footprint_and_activation_helpers_convert_units():
    x = np.linspace(0.0, 1000.0, 5) * axs.um
    electrode = PointSourceElectrode(x_um=0.0 * axs.um, z_um=1000.0 * axs.um)
    extra = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3 * axs.S_per_m,
    )

    footprint_mV_per_uA = extra.footprint_per_current(
        electrode,
        x,
        voltage_unit=axs.mV,
        current_unit=axs.uA,
    )
    activation = extra.activation_function(
        electrode,
        x,
        voltage_unit=axs.mV,
        current_unit=axs.uA,
    )

    x_m = np.linspace(0.0, 1.0e-3, 5)
    assert np.allclose(footprint_mV_per_uA, extra.footprint_for_electrode(electrode, x_m) * 1e-3)
    assert activation.shape == footprint_mV_per_uA.shape
    assert np.isfinite(activation).all()


def test_context_plot_helpers_smoke():
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1000.0, 25) * axs.um
    t = np.linspace(0.0, 3.0, 50) * axs.ms
    electrode = PointSourceElectrode(x_um=0.0 * axs.um, z_um=1000.0 * axs.um)
    extra = _context(
        electrode,
        Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0 * axs.uA, duration=1.0 * axs.ms),
        sigma=0.3 * axs.S_per_m,
    )

    fig, axes = plt.subplots(1, 3)
    try:
        assert extra.plot_footprint(x, ax=axes[0]) is axes[0]
        assert extra.plot_evaluation(x, t, ax=axes[1]) is axes[1]
        assert extra.plot_activation_function(x, ax=axes[2]) is axes[2]
        assert len(axes[0].lines) >= 1
        assert len(axes[1].images) == 1
        assert len(axes[2].lines) >= 1
    finally:
        plt.close(fig)


def test_compile_returns_jax_ready_object():
    x = np.linspace(0.0, 1.0e-3, 5)
    electrode = PointSourceElectrode(x0_m=0.5e-3, z0_m=1.0e-3)
    stim = Stimulus.pulse(start=1.0, amplitude=1.0e-6, duration=1.0)
    extra = _context(electrode, stim)

    compiled = compile_extracellular_context(extra, x)

    assert isinstance(compiled, CompiledExtracellularContext)
    assert compiled.electrodes[0].footprint_V_per_A.shape == (len(x),)


def test_compiled_extracellular_stimulus_matches_numpy():
    x = np.linspace(0.0, 1.0e-3, 5)
    electrode = PointSourceElectrode(x0_m=0.5e-3, z0_m=1.0e-3)
    stim = Stimulus.pulse(start=1.0, amplitude=2.0e-6, duration=1.0)
    extra = _context(electrode, stim)
    compiled = compile_extracellular_context(extra, x)

    expected = extra.evaluate(x, np.array([1.5]), position_unit="meter")[0]
    got = np.asarray(compiled(1.5))

    assert np.allclose(got, expected)


def test_extracellular_runtime_accepts_context_contract_without_analytical_subclass():
    x = np.linspace(0.0, 1.0e-3, 5)
    electrode = PointSourceElectrode(x0_m=0.0, z0_m=1.0e-3).with_stimulus(
        Stimulus.constant(2.0e-6)
    )
    extra = _ConstantFootprintContext(electrodes=[electrode])

    expected = extra.evaluate(x, np.array([0.0]), position_unit="meter")[0]
    compiled = compile_extracellular_context(extra, x)
    got = np.asarray(compiled(0.0))

    assert np.allclose(expected, 6.0e-6)
    assert np.allclose(got, expected)


def test_point_source_units_scale_linearly_with_current():
    x = np.array([1.0e-3])
    electrode = PointSourceElectrode(x0_m=0.0, z0_m=1.0e-3)

    extra_1 = _context(electrode, Stimulus.constant(1.0e-6))
    extra_2 = _context(electrode, Stimulus.constant(2.0e-6))

    V1 = extra_1.evaluate(x, np.array([0.0]), position_unit="meter")[0, 0]
    V2 = extra_2.evaluate(x, np.array([0.0]), position_unit="meter")[0, 0]

    assert np.isclose(V2, 2.0 * V1)


def test_nrv_extracellular_context_is_declared_but_not_implemented():
    electrode = PointSourceElectrode(x_um=0.0, stimulus=Stimulus.constant(0.0))
    ctx = NRVExtracellularContext(
        electrodes=[electrode],
        medium="endoneurium_bhadra",
        fem_model={"mesh": "future"},
        metadata={"case": "smoke"},
    )

    copied = ctx.with_electrodes([electrode])

    assert ctx.backend == "nrv"
    assert ctx.medium == "endoneurium_bhadra"
    assert ctx.fem_model == {"mesh": "future"}
    assert ctx.metadata["case"] == "smoke"
    assert copied.medium == ctx.medium
    assert copied.fem_model == ctx.fem_model
    assert copied.metadata["case"] == "smoke"

    with pytest.raises(NotImplementedError, match="not implemented"):
        ctx.footprint_for_electrode(electrode, np.array([0.0]))
