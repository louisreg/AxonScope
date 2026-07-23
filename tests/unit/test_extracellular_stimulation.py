import numpy as np
import pytest

import axonfleet as axs
from axonfleet.analytical import PointSourceElectrode
from axonfleet.stimulation import (
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularPotential,
    ExtracellularStimulation,
    Stimulus,
)


def test_point_source_footprint_matches_analytical_formula():
    x = np.array([0.0, 1.0e-3, 2.0e-3])
    electrode = PointSourceElectrode(x=1.0e-3 * axs.m, y=0.0 * axs.m, z=1.0e-3 * axs.m)

    fp = electrode.footprint_for_axon(x, sigma_S_m=0.3)

    r = np.sqrt((x - 1.0e-3) ** 2 + (1.0e-3) ** 2)
    expected = 1.0 / (4.0 * np.pi * 0.3 * r)
    assert np.allclose(fp, expected)


def test_point_source_position_units_normalize_to_common_geometry():
    x_m = np.array([0.0, 1.0e-3, 2.0e-3])
    by_um = PointSourceElectrode(x=1000.0 * axs.um, z=1000.0 * axs.um)
    by_m = PointSourceElectrode(x=1.0e-3 * axs.m, z=1.0e-3 * axs.m)

    assert np.allclose(
        by_um.footprint_for_axon(x_m, sigma_S_m=0.3),
        by_m.footprint_for_axon(x_m, sigma_S_m=0.3),
    )


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

    fp = electrode.footprint_for_axon(x, sigma_S_m=0.3)

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

    fp = electrode.footprint_for_axon(x, sigma_S_m=0.3)

    expected = 1.0 / (4.0 * np.pi * 0.3 * 1.0e-6)
    assert np.isfinite(fp[0])
    assert np.isclose(fp[0], expected)


def test_point_source_footprint_offsets_match_transverse_calculation():
    positions_m = np.linspace(0.0, 1.0e-3, 5)
    electrode = PointSourceElectrode(
        x=500.0 * axs.um,
        y=0.0 * axs.um,
        z=0.0 * axs.um,
    )
    footprint = axs.analytical.point_source_footprint(
        electrode,
        positions_m * axs.m,
        sigma=0.3 * axs.S_per_m,
        axon_y=20.0 * axs.um,
        axon_z=-40.0 * axs.um,
    )

    r = np.sqrt(
        (positions_m - 500.0e-6) ** 2
        + ((0.0 - 20.0) * 1e-6) ** 2
        + ((0.0 - (-40.0)) * 1e-6) ** 2
    )
    expected = 1.0 / (4.0 * np.pi * 0.3 * r)
    np.testing.assert_allclose(footprint.values_for_axon(), expected)


def test_point_source_stimulation_attaches_as_typed_stimulation():
    axon = axs.axons.HodgkinHuxley(
        length=100.0 * axs.um,
        diameter=0.5 * axs.um,
        compartments=5,
        celsius=6.3 * axs.degC,
    )
    positions = axon.layout.position_values(unit=axs.um) * axs.um
    electrode = PointSourceElectrode(x=50.0 * axs.um, z=100.0 * axs.um)
    stimulation = axs.analytical.point_source_stimulation(
        electrode,
        positions,
        sigma=0.3 * axs.S_per_m,
        stimulus=Stimulus.constant(1.0 * axs.uA),
    )

    sim = axs.AxonInstance(axon)
    sim.add_extracellular_stimulation(stimulation=stimulation)

    assert sim.extracellular_stimulation is stimulation
    assert not hasattr(sim, "extracellular_stimulations")
    got = stimulation.evaluate(np.asarray([0.0]) * axs.ms, voltage_unit=axs.mV)[0]
    expected = stimulation.evaluate(np.asarray([0.0]) * axs.ms, voltage_unit=axs.mV)[0]
    np.testing.assert_allclose(got, expected)


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
    expected_mV = 1e3 * (
        2.0e-6 * np.array([1.0, 2.0, 3.0])
        + 1.0e-6 * np.array([10.0, 20.0, 30.0])
    )

    assert stimulation.names == (axs.DriveId("cathode"), axs.DriveId("anode"))
    assert stimulation[axs.DriveId("cathode")] is drive_a
    np.testing.assert_allclose(got_mV, expected_mV)


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


def test_extracellular_stimulation_rejects_duplicate_or_incompatible_drives():
    positions = np.array([0.0, 500.0]) * axs.um
    shifted_positions = np.array([0.0, 250.0, 500.0]) * axs.um
    drive = ExtracellularDrive(
        id=axs.DriveId("source"),
        footprint=ExtracellularFootprint.shared(values=np.array([1.0, 2.0]), positions=positions),
        stimulus=Stimulus.constant(1.0 * axs.uA),
    )
    duplicate = ExtracellularDrive(
        id=axs.DriveId("source"),
        footprint=drive.footprint,
        stimulus=Stimulus.constant(2.0 * axs.uA),
    )
    incompatible = ExtracellularDrive(
        id=axs.DriveId("other"),
        footprint=ExtracellularFootprint.shared(
            values=np.array([1.0, 1.5, 2.0]),
            positions=shifted_positions,
        ),
        stimulus=Stimulus.constant(2.0 * axs.uA),
    )

    with pytest.raises(ValueError, match="ids must be unique"):
        ExtracellularStimulation([drive, duplicate])
    with pytest.raises(ValueError, match="incompatible position supports"):
        ExtracellularStimulation([drive, incompatible])


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
    np.testing.assert_allclose(potential.value_values(voltage_unit=axs.mV)[0], [0.003, 0.006])


def test_extracellular_public_plot_helpers():
    import matplotlib.pyplot as plt

    positions = np.array([0.0, 500.0]) * axs.um
    footprint = ExtracellularFootprint.shared(
        values=np.array([1.0, 2.0]),
        positions=positions,
        source_id="source",
    )
    drive = ExtracellularDrive(
        id=axs.DriveId("source"),
        footprint=footprint,
        stimulus=Stimulus.constant(3.0 * axs.uA),
    )
    stimulation = ExtracellularStimulation([drive])
    potential = stimulation.potential(np.array([0.0, 1.0]) * axs.ms, voltage_unit=axs.mV)

    _, axes = plt.subplots(1, 3)
    footprint.plot(ax=axes[0], position_unit=axs.um, voltage_unit=axs.mV, current_unit=axs.uA)
    stimulation.plot_footprints(
        ax=axes[1],
        position_unit=axs.um,
        voltage_unit=axs.mV,
        current_unit=axs.uA,
    )
    potential.plot(ax=axes[2], time_unit=axs.ms, position_unit=axs.um, voltage_unit=axs.mV)

    assert len(axes[0].lines) == 1
    assert len(axes[1].lines) == 1
    assert len(axes[2].images) == 1
    plt.close("all")


def test_point_source_units_scale_linearly_with_current():
    x = np.array([1.0e-3]) * axs.m
    electrode = PointSourceElectrode(x=0.0 * axs.m, z=1.0e-3 * axs.m)

    extra_1 = axs.analytical.point_source_stimulation(
        electrode,
        x,
        sigma=0.3 * axs.S_per_m,
        stimulus=Stimulus.constant(1.0e-6),
    )
    extra_2 = axs.analytical.point_source_stimulation(
        electrode,
        x,
        sigma=0.3 * axs.S_per_m,
        stimulus=Stimulus.constant(2.0e-6),
    )

    v1 = extra_1.evaluate(np.array([0.0]) * axs.ms)[0, 0]
    v2 = extra_2.evaluate(np.array([0.0]) * axs.ms)[0, 0]

    assert np.isclose(v2, 2.0 * v1)
