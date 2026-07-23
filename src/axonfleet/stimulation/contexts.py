"""Physical stimulation contexts attached to simulation protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axonfleet.stimulation.stimuli import Stimulus
from axonfleet.utils import units


@dataclass(frozen=True, kw_only=True, init=False)
class IntracellularCurrentClamp:
    """Descriptive intracellular current clamp attached to an axon position.

    `position` must carry length units and is stored internally as
    `position_um`. Plain waveform amplitudes are interpreted as nanoamperes.
    Pint quantities are converted to nanoamperes at construction time.
    """

    position_um: Any
    """Axial clamp position in micrometers after normalization."""

    current: Stimulus
    """Current waveform normalized to nanoamperes."""

    def __init__(self, *, position: Any, current: Stimulus) -> None:
        """Create an intracellular current clamp at axial `position`."""

        object.__setattr__(self, "position_um", position)
        object.__setattr__(self, "current", current)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate the current waveform and normalize public units."""

        if not isinstance(self.current, Stimulus):
            raise TypeError("current must be an axonfleet.stimulation.Stimulus.")
        object.__setattr__(
            self,
            "position_um",
            units.require_length_um(self.position_um, name="position"),
        )
        object.__setattr__(self, "current", self.current.as_unit("nanoampere"))


__all__ = ["IntracellularCurrentClamp"]
