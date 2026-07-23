"""Host-side row helpers for dispatcher batch preparation."""

from __future__ import annotations

from typing import Sequence

from axonfleet.axon_instance import AxonInstance
from axonfleet.axons.axon import Axon
from axonfleet.stimulation import ExtracellularStimulation


def extracellular_stimulation_rows(
    axons: Sequence[Axon | AxonInstance],
) -> tuple[tuple[ExtracellularStimulation, ...], ...]:
    """Return one enabled extracellular-stimulation row per axon."""

    return tuple(
        (axon.extracellular_stimulation,)
        if bool(getattr(axon, "use_extracellular", False))
        and getattr(axon, "extracellular_stimulation", None) is not None
        else ()
        for axon in axons
    )
