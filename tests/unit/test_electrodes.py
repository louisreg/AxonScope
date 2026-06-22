import numpy as np
import pytest

import axonscope as axs
from axonscope.stimulation import (
    AnalyticalExtracellularContext,
    ExtracellularContext,
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularPotential,
    ExtracellularStimulation,
    NRVExtracellularContext,
)
from axonscope.stimulation import PointSourceElectrode, Stimulus
from axonscope.stimulation.runtime import (
    CompiledExtracellularContext,
    compile_extracellular_context,
)


def _context(electrode: PointSourceElectrode, stimulus: Stimulus, *, sigma=0.3 * axs.S_per_m):
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
    electrode = PointSourceElectrode(x=1.0e-3 * axs.m, y=0.0 * axs.m, z=1.0e-3 * axs.m)
    ctx = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3 * axs.S_per_m,
    )

    fp = ctx.footprint_for_electrode(electrode, x)

    r = np.sqrt((x - 1.0e-3) ** 2 + (1.0e-3) ** 2)
    expected = 1.0 / (4.0 * np.pi * 0.3 * r)
    assert np.allclose(fp, expected)


def test_point_source_position_units_normalize_to_common_geometry():
    x_m = np.array([0.0, 1.0e-3, 2.0e-3])
    by_um = PointSourceElectrode(x=1000.0 * axs.um, z=1000.0 * axs.um)
    by_m = PointSourceElectrode(x=1.0e-3 * axs.m, z=1.0e-3 * axs.m)
    ctx = AnalyticalExtracellularContext(
        electrodes=[by_um.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3 * axs.S_per_m,
    )

    assert np.allclose(ctx.footprint_for_electrode(by_um, x_m), ctx.footprint_for_electrode(by_m, x_m))


def test_point_source_rejects_plain_coordinate_values():
    with pytest.raises(TypeError, match="x must include units compatible with length"):
        PointSourceElectrode(x=0.0, z=1000.0 * axs.um)
    with pytest.raises(TypeError, match="z must include units compatible with length"):
        PointSourceElectrode(x=0.0 * axs.um, z=1000.0)
    with pytest.raises(TypeError, match="min_distance must include units compatible with length"):
        PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um, min_distance=1e-3)


def test_point_source_footprint_is_symmetric_around_electrode():
    x = np.array([-1.0e-3, 0.0, 1.0e-3])
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m)
    ctx = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3 * axs.S_per_m,
    )

    fp = ctx.footprint_for_electrode(electrode, x)

    assert np.isclose(fp[0], fp[2])
    assert fp[1] > fp[0]


def test_point_source_min_distance_avoids_singularity():
    x = np.array([0.0])
    electrode = PointSourceElectrode(
        x=0.0 * axs.m,
        y=0.0 * axs.m,
        z=0.0 * axs.m,
        min_distance=1.0e-6 * axs.m,
    )
    ctx = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0))],
        sigma=0.3 * axs.S_per_m,
    )

    fp = ctx.footprint_for_electrode(electrode, x)

    expected = 1.0 / (4.0 * np.pi * 0.3 * 1.0e-6)
    assert np.isfinite(fp[0])
    assert np.isclose(fp[0], expected)


def test_with_stimulus_returns_stimulated_copy():
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=1.0e-6, duration=1.0 * axs.ms)

    returned = electrode.with_stimulus(stim)

    assert returned is not electrode
    assert electrode.stimulus is None
    assert returned.stimulus.y_unit == "ampere"
    assert np.allclose(returned.stimulus.y, stim.y)


def test_set_stimulus_updates_electrode_in_place():
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m)
    first = Stimulus.pulse(start=1.0 * axs.ms, amplitude=1.0e-6, duration=1.0 * axs.ms)
    second = Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0e-6, duration=1.0 * axs.ms)

    electrode.set_stimulus(first)
    assert electrode.stimulus is not None
    np.testing.assert_allclose(electrode.stimulus.y, first.y)

    electrode.set_stimulus(second)
    assert electrode.stimulus is not None
    np.testing.assert_allclose(electrode.stimulus.y, second.y)


def test_analytical_context_normalizes_sigma_units():
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um, stimulus=Stimulus.constant(0.0))
    ctx = AnalyticalExtracellularContext(electrodes=[electrode], sigma=0.3 * axs.S_per_m)

    assert np.isclose(ctx.sigma_S_m, 0.3)


def test_analytical_context_rejects_plain_sigma():
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um, stimulus=Stimulus.constant(0.0))

    with pytest.raises(TypeError, match="sigma must include units compatible with conductivity"):
        AnalyticalExtracellularContext(electrodes=[electrode], sigma=0.3)


def test_extracellular_context_evaluate_shape():
    x = np.linspace(0.0, 1.0e-3, 5)
    t = np.linspace(0.0, 3.0, 7)
    electrode = PointSourceElectrode(x=0.5e-3 * axs.m, z=1.0e-3 * axs.m)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0e-6, duration=1.0 * axs.ms)
    extra = _context(electrode, stim)

    Vext = extra.evaluate(x, t, position_unit="meter")

    assert Vext.shape == (len(t), len(x))


def test_extracellular_context_evaluate_values():
    x = np.array([0.0, 1.0e-3])
    t = np.array([0.5, 1.5, 2.5])
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0e-6, duration=1.0 * axs.ms)
    extra = _context(electrode, stim)

    Vext = extra.evaluate(x, t, position_unit="meter")
    fp = extra.footprint_for_electrode(electrode, x)

    assert np.allclose(Vext[0], 0.0)
    assert np.allclose(Vext[1], 2.0e-6 * fp)
    assert np.allclose(Vext[2], 0.0)


def test_context_evaluate_uses_attached_stimulus_with_units():
    x = np.array([0.0, 1000.0]) * axs.um
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0 * axs.uA, duration=1.0 * axs.ms)
    extra = _context(electrode, stim, sigma=0.3 * axs.S_per_m)

    vext_mV = extra.evaluate(x, np.array([1.5]) * axs.ms, voltage_unit=axs.mV)

    expected_mV = 2.0e-6 * extra.footprint_for_electrode(electrode, np.array([0.0, 1.0e-3])) * 1e3
    assert np.allclose(vext_mV[0], expected_mV)


def test_footprint_and_activation_helpers_convert_units():
    x = np.linspace(0.0, 1000.0, 5) * axs.um
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um)
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


def test_analytical_context_builds_static_extracellular_footprint():
    positions = np.linspace(0.0, 1000.0, 5) * axs.um
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um)
    context = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0 * axs.uA))],
        sigma=0.3 * axs.S_per_m,
    )

    footprint = context.build_footprint(
        electrode,
        positions,
        source_id="center-electrode",
    )

    assert isinstance(footprint, ExtracellularFootprint)
    assert footprint.source_id == "center-electrode"
    assert footprint.shared_across_axons
    assert np.allclose(footprint.position_values(unit=axs.um), np.linspace(0.0, 1000.0, 5))
    assert np.allclose(
        footprint.values_for_axon(),
        context.footprint_for_electrode(electrode, np.linspace(0.0, 1.0e-3, 5)),
    )


def test_point_source_builds_same_footprint_as_context_builder():
    positions = np.linspace(0.0, 1000.0, 5) * axs.um
    electrode = PointSourceElectrode(x=500.0 * axs.um, z=1000.0 * axs.um)
    context = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(Stimulus.constant(0.0 * axs.uA))],
        sigma=0.3 * axs.S_per_m,
    )

    direct = electrode.build_footprint(positions, sigma=0.3 * axs.S_per_m)
    through_context = context.build_footprint(electrode, positions)

    assert np.allclose(direct.values_for_axon(), through_context.values_for_axon())


def test_local_point_source_context_matches_transverse_offset_calculation():
    positions_m = np.linspace(0.0, 1.0e-3, 5)
    stimulus = Stimulus.constant(0.0 * axs.uA)
    electrode = PointSourceElectrode(
        x=500.0 * axs.um,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    context = AnalyticalExtracellularContext(
        electrodes=[electrode.with_stimulus(stimulus)],
        sigma=0.3 * axs.S_per_m,
    )

    legacy = context.footprint_for_electrode(
        context.electrodes[0],
        positions_m,
        axon_y_um=20.0,
        axon_z_um=-40.0,
    )
    local = axs.analytical.local_point_source_context(
        electrode,
        stimulus=stimulus,
        sigma=0.3 * axs.S_per_m,
        axon_y=20.0 * axs.um,
        axon_z=-40.0 * axs.um,
    )
    shifted = local.footprint_for_electrode(local.electrodes[0], positions_m)

    np.testing.assert_allclose(shifted, legacy)


def test_extracellular_drive_and_stimulation_evaluate_factorized_sum():
    positions = np.array([0.0, 500.0, 1000.0]) * axs.um
    footprint_a = ExtracellularFootprint.shared(
        values=np.array([1.0, 2.0, 3.0]),
        positions=positions,
    )
    footprint_b = ExtracellularFootprint.shared(
        values=np.array([10.0, 20.0, 30.0]),
        positions=positions,
    )
    drive_a = ExtracellularDrive(
        id=axs.DriveId("cathode"),
        footprint=footprint_a,
        stimulus=Stimulus.constant(2.0 * axs.uA),
    )
    drive_b = ExtracellularDrive(
        id=axs.DriveId("anode"),
        footprint=footprint_b,
        stimulus=Stimulus.constant(1.0 * axs.uA),
    )
    stimulation = ExtracellularStimulation([drive_a, drive_b])

    got_mV = stimulation.evaluate(np.array([0.0]) * axs.ms, voltage_unit=axs.mV)[0]
    expected_mV = 1e3 * (2.0e-6 * np.array([1.0, 2.0, 3.0]) + 1.0e-6 * np.array([10.0, 20.0, 30.0]))

    assert stimulation.names == (axs.DriveId("cathode"), axs.DriveId("anode"))
    assert stimulation[axs.DriveId("cathode")] is drive_a
    assert np.allclose(got_mV, expected_mV)


def test_extracellular_drive_rejects_raw_string_identifier():
    footprint = ExtracellularFootprint.shared(
        values=np.array([1.0]),
        positions=np.array([0.0]) * axs.um,
    )

    with pytest.raises(TypeError, match="DriveId"):
        ExtracellularDrive(
            id="source",
            footprint=footprint,
            stimulus=Stimulus.constant(0.0 * axs.uA),
        )


def test_extracellular_stimulation_materializes_dense_potential_explicitly():
    positions = np.array([0.0, 500.0]) * axs.um
    footprint = ExtracellularFootprint.shared(values=np.array([1.0, 2.0]), positions=positions)
    drive = ExtracellularDrive(
        id=axs.DriveId("source"),
        footprint=footprint,
        stimulus=Stimulus.constant(3.0 * axs.uA),
    )
    stimulation = ExtracellularStimulation([drive])

    potential = stimulation.potential(np.array([0.0, 1.0]) * axs.ms, voltage_unit=axs.mV)

    assert isinstance(potential, ExtracellularPotential)
    assert potential.value_values(voltage_unit=axs.mV).shape == (2, 2)
    assert np.allclose(potential.value_values(voltage_unit=axs.mV)[0], [0.003, 0.006])


def test_context_plot_helpers_smoke():
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1000.0, 25) * axs.um
    t = np.linspace(0.0, 3.0, 50) * axs.ms
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um)
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
    electrode = PointSourceElectrode(x=0.5e-3 * axs.m, z=1.0e-3 * axs.m)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=1.0e-6, duration=1.0 * axs.ms)
    extra = _context(electrode, stim)

    compiled = compile_extracellular_context(extra, x)

    assert isinstance(compiled, CompiledExtracellularContext)
    assert compiled.electrodes[0].footprint_V_per_A.shape == (len(x),)


def test_compiled_extracellular_stimulus_matches_numpy():
    x = np.linspace(0.0, 1.0e-3, 5)
    electrode = PointSourceElectrode(x=0.5e-3 * axs.m, z=1.0e-3 * axs.m)
    stim = Stimulus.pulse(start=1.0 * axs.ms, amplitude=2.0e-6, duration=1.0 * axs.ms)
    extra = _context(electrode, stim)
    compiled = compile_extracellular_context(extra, x)

    expected = extra.evaluate(x, np.array([1.5]), position_unit="meter")[0]
    got = np.asarray(compiled(1.5))

    assert np.allclose(got, expected)


def test_extracellular_runtime_accepts_context_contract_without_analytical_subclass():
    x = np.linspace(0.0, 1.0e-3, 5)
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m).with_stimulus(
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
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m)

    extra_1 = _context(electrode, Stimulus.constant(1.0e-6))
    extra_2 = _context(electrode, Stimulus.constant(2.0e-6))

    V1 = extra_1.evaluate(x, np.array([0.0]), position_unit="meter")[0, 0]
    V2 = extra_2.evaluate(x, np.array([0.0]), position_unit="meter")[0, 0]

    assert np.isclose(V2, 2.0 * V1)


def test_nrv_extracellular_context_is_declared_but_not_implemented():
    electrode = PointSourceElectrode(x=0.0 * axs.um, z=1000.0 * axs.um, stimulus=Stimulus.constant(0.0))
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
