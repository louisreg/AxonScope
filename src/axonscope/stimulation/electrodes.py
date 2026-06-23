"""Extracellular electrode source descriptions.

Electrodes describe current sources with attached temporal stimuli. Concrete
analytical source geometries live in helper modules such as
`axonscope.analytical`; stimulation core code only depends on the generic
electrode and footprint contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from axonscope.stimulation.stimuli import ArrayLike, Stimulus


def _normalize_stimulus(stimulus: Stimulus) -> Stimulus:
    """Validate an electrode stimulus and express amplitudes in amperes."""

    if not isinstance(stimulus, Stimulus):
        raise TypeError("stimulus must be an axonscope.stimulation.Stimulus.")
    return stimulus.as_unit("ampere")


@dataclass(frozen=True)
class Electrode(ABC):
    """Base class for positioned extracellular current sources.

    An electrode can own an attached stimulus, but it does not know how the
    surrounding medium maps its current to Vext. Context objects provide that
    medium-specific mapping.
    """

    stimulus: Stimulus | None = field(default=None, init=False, compare=False)
    """Attached current waveform normalized to amperes, or `None`."""

    def set_stimulus(self, stimulus: Stimulus) -> None:
        """Attach `stimulus` to this electrode in place.

        Use this when a simulation keeps the same electrode geometry and only
        changes the driven current between runs, for example during threshold
        searches. Plain stimulus amplitudes are interpreted as amperes for
        extracellular stimulation.
        """

        object.__setattr__(self, "stimulus", _normalize_stimulus(stimulus))

    def with_stimulus(self, stimulus: Stimulus) -> "Electrode":
        """Return a copy of this electrode with `stimulus` attached.

        Plain stimulus amplitudes are interpreted as amperes for extracellular
        stimulation. This is the preferred method when reusing the same
        electrode geometry across multiple simulations with different stimuli.
        """

        electrode = copy.copy(self)
        object.__setattr__(electrode, "stimulus", _normalize_stimulus(stimulus))
        return electrode

    def with_scaled_stimulus(self, scale: float) -> "Electrode":
        """Return a copy whose attached stimulus amplitude is scaled."""

        stimulus = getattr(self, "stimulus", None)
        if stimulus is None:
            raise ValueError("Attach a stimulus before scaling an electrode.")
        return self.with_stimulus(stimulus.scaled(scale))


@dataclass(frozen=True)
class AnalyticalElectrode(Electrode, ABC):
    """Base class for electrodes with an analytic footprint formula."""

    @abstractmethod
    def footprint(
        self,
        x_positions_m: ArrayLike,
        *,
        sigma_S_m: float,
    ) -> np.ndarray:
        """Return V/A footprint samples for an axon at y=z=0.

        Parameters
        ----------
        x_positions_m:
            Axial sample positions in meters.
        sigma_S_m:
            Conductivity of the homogeneous medium in S/m.
        """
        raise NotImplementedError

    def footprint_for_axon(
        self,
        x_positions_m: ArrayLike,
        *,
        sigma_S_m: float,
        axon_y_um: Any = 0.0,
        axon_z_um: Any = 0.0,
    ) -> np.ndarray:
        """Return V/A footprint samples for one axon-local evaluation."""

        return self.footprint(x_positions_m, sigma_S_m=sigma_S_m)
