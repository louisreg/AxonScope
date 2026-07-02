"""Physical stimulation contexts attached to simulation protocols."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

from axonscope.stimulation.stimuli import Stimulus
from axonscope.utils import units


@dataclass(frozen=True, kw_only=True)
class IntracellularContext(ABC):
    """Base class for intracellular stimulation descriptions.

    Intracellular contexts describe current injection protocols attached to an
    axon. Runtime compilers lower concrete subclasses to solver arrays and
    functions; solver kernels do not inspect these public objects directly.
    """


@dataclass(frozen=True, kw_only=True, init=False)
class IntracellularCurrentClamp(IntracellularContext):
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
            raise TypeError("current must be an axonscope.stimulation.Stimulus.")
        object.__setattr__(
            self,
            "position_um",
            units.require_length_um(self.position_um, name="position"),
        )
        object.__setattr__(self, "current", self.current.as_unit("nanoampere"))


__all__ = ["IntracellularContext", "IntracellularCurrentClamp"]
