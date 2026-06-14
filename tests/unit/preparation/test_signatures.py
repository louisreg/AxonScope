from __future__ import annotations

import numpy as np
import pytest

import axonscope as axs
from axonscope.preparation import (
    array_signature,
    drive_signature,
    extracellular_stimulation_signature,
    footprint_signature,
    stimulus_signature,
)


def test_array_signature_tracks_shape_dtype_and_content():
    first = array_signature(np.asarray([1.0, 2.0], dtype=np.float32))
    same = array_signature(np.asarray([1.0, 2.0], dtype=np.float32))
    changed_dtype = array_signature(np.asarray([1.0, 2.0], dtype=np.float64))
    changed_value = array_signature(np.asarray([1.0, 2.1], dtype=np.float32))

    assert first == same
    assert first.shape == (2,)
    assert first.dtype == "float32"
    assert first != changed_dtype
    assert first != changed_value


def test_footprint_signature_is_stable_for_equal_scientific_inputs():
    positions = np.array([0.0, 500.0, 1000.0]) * axs.um
    footprint_a = axs.ExtracellularFootprint.shared(
        values=np.array([1.0, 2.0, 3.0]),
        positions=positions,
        source_id="electrode",
    )
    footprint_b = axs.ExtracellularFootprint.shared(
        values=np.array([1.0, 2.0, 3.0]),
        positions=positions,
        source_id="electrode",
    )
    footprint_c = axs.ExtracellularFootprint.shared(
        values=np.array([1.0, 2.0, 3.5]),
        positions=positions,
        source_id="electrode",
    )

    assert footprint_signature(footprint_a) == footprint_signature(footprint_b)
    assert footprint_signature(footprint_a) != footprint_signature(footprint_c)
    assert footprint_signature(footprint_a).source_id == "electrode"


def test_stimulation_signature_tracks_drive_id_stimulus_and_order():
    positions = np.array([0.0, 500.0]) * axs.um
    footprint = axs.ExtracellularFootprint.shared(
        values=np.array([1.0, 2.0]),
        positions=positions,
    )
    stimulus_a = axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.2 * axs.ms,
        amplitude=10.0 * axs.uA,
    )
    stimulus_b = axs.Stimulus.pulse(
        start=1.0 * axs.ms,
        duration=0.2 * axs.ms,
        amplitude=20.0 * axs.uA,
    )
    drive_a = axs.ExtracellularDrive(
        id=axs.DriveId("a"),
        footprint=footprint,
        stimulus=stimulus_a,
    )
    drive_b = axs.ExtracellularDrive(
        id=axs.DriveId("b"),
        footprint=footprint,
        stimulus=stimulus_b,
    )

    signature_ab = extracellular_stimulation_signature(
        axs.ExtracellularStimulation([drive_a, drive_b])
    )
    signature_ba = extracellular_stimulation_signature(
        axs.ExtracellularStimulation([drive_b, drive_a])
    )

    assert drive_signature(drive_a).id == axs.DriveId("a")
    assert stimulus_signature(stimulus_a) != stimulus_signature(stimulus_b)
    assert signature_ab.drives[0].id == axs.DriveId("a")
    assert signature_ab.drives[1].id == axs.DriveId("b")
    assert signature_ab != signature_ba


def test_signature_helpers_reject_wrong_objects():
    with pytest.raises(TypeError, match="Stimulus"):
        stimulus_signature(object())
    with pytest.raises(TypeError, match="ExtracellularFootprint"):
        footprint_signature(object())
    with pytest.raises(TypeError, match="ExtracellularDrive"):
        drive_signature(object())
    with pytest.raises(TypeError, match="ExtracellularStimulation"):
        extracellular_stimulation_signature(object())
