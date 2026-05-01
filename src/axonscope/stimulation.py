from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axonscope.stimulus import Stimulus

if TYPE_CHECKING:
    from axonscope.electrodes import Electrode


@dataclass(frozen=True)
class IntracellularCurrentClamp:
    """Descriptive intracellular current clamp attached to an axon position."""

    position_um: float
    stimulus: Stimulus


@dataclass(frozen=True)
class ExtracellularContext:
    """Descriptive pairing between an electrode and a temporal stimulus."""

    electrode: "Electrode"
    stimulus: Stimulus
