"""Stable signatures for future reusable prepared simulation cohorts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from axonscope.identifiers import AxonId, DriveId
from axonscope.stimulation.extracellular import (
    ExtracellularDrive,
    ExtracellularFootprint,
    ExtracellularStimulation,
)
from axonscope.stimulation.stimuli import Stimulus


@dataclass(frozen=True)
class ArraySignature:
    """Shape, dtype, and content digest for a preparation input array."""

    shape: tuple[int, ...]
    dtype: str
    digest: str


@dataclass(frozen=True)
class StimulusSignature:
    """Preparation signature for a temporal waveform."""

    t_ms: ArraySignature
    y: ArraySignature
    mode: str
    y_unit: str | None


@dataclass(frozen=True)
class FootprintSignature:
    """Preparation signature for a static extracellular footprint."""

    shared_across_axons: bool
    axon_ids: tuple[AxonId, ...] | None
    positions_um: ArraySignature
    values_V_per_A: ArraySignature
    interpolation: str
    source_id: str | None
    reference: str


@dataclass(frozen=True)
class DriveSignature:
    """Preparation signature for one extracellular drive."""

    id: DriveId
    footprint: FootprintSignature
    stimulus: StimulusSignature


@dataclass(frozen=True)
class ExtracellularStimulationSignature:
    """Preparation signature for a complete extracellular drive set."""

    drives: tuple[DriveSignature, ...]


def array_signature(values: Any) -> ArraySignature:
    """Return a deterministic signature for array-like preparation data."""

    arr = np.asarray(values)
    digest = _array_digest(arr)
    return ArraySignature(
        shape=tuple(int(dim) for dim in arr.shape),
        dtype=str(arr.dtype),
        digest=digest,
    )


def stimulus_signature(stimulus: Stimulus) -> StimulusSignature:
    """Return the reusable-preparation signature for `stimulus`."""

    if not isinstance(stimulus, Stimulus):
        raise TypeError("stimulus must be an axonscope.stimulation.Stimulus.")
    return StimulusSignature(
        t_ms=array_signature(stimulus.t),
        y=array_signature(stimulus.y),
        mode=stimulus.mode,
        y_unit=stimulus.y_unit,
    )


def footprint_signature(footprint: ExtracellularFootprint) -> FootprintSignature:
    """Return the reusable-preparation signature for `footprint`."""

    if not isinstance(footprint, ExtracellularFootprint):
        raise TypeError("footprint must be an ExtracellularFootprint.")
    return FootprintSignature(
        shared_across_axons=footprint.shared_across_axons,
        axon_ids=None if footprint.axon_ids is None else tuple(footprint.axon_ids),
        positions_um=array_signature(footprint.positions_um),
        values_V_per_A=array_signature(footprint.values_V_per_A),
        interpolation=footprint.interpolation,
        source_id=footprint.source_id,
        reference=footprint.reference,
    )


def drive_signature(drive: ExtracellularDrive) -> DriveSignature:
    """Return the reusable-preparation signature for one extracellular drive."""

    if not isinstance(drive, ExtracellularDrive):
        raise TypeError("drive must be an ExtracellularDrive.")
    return DriveSignature(
        id=drive.id,
        footprint=footprint_signature(drive.footprint),
        stimulus=stimulus_signature(drive.stimulus),
    )


def extracellular_stimulation_signature(
    stimulation: ExtracellularStimulation,
) -> ExtracellularStimulationSignature:
    """Return a deterministic signature for a complete stimulation definition."""

    if not isinstance(stimulation, ExtracellularStimulation):
        raise TypeError("stimulation must be an ExtracellularStimulation.")
    return ExtracellularStimulationSignature(
        drives=tuple(drive_signature(drive) for drive in stimulation.drives)
    )


def _array_digest(values: np.ndarray) -> str:
    arr = np.asarray(values)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(tuple(int(dim) for dim in arr.shape)).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    if arr.dtype.hasobject:
        payload = json.dumps(arr.tolist(), sort_keys=True, default=repr).encode("utf-8")
        hasher.update(payload)
    else:
        contiguous = np.ascontiguousarray(arr)
        hasher.update(contiguous.view(np.uint8))
    return hasher.hexdigest()


__all__ = [
    "ArraySignature",
    "DriveSignature",
    "ExtracellularStimulationSignature",
    "FootprintSignature",
    "StimulusSignature",
    "array_signature",
    "drive_signature",
    "extracellular_stimulation_signature",
    "footprint_signature",
    "stimulus_signature",
]
