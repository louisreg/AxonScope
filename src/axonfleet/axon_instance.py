"""Concrete axon instance used by simulation execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from axonfleet.axons.axon import Axon
from axonfleet.stimulation import (
    ExtracellularStimulation,
    IntracellularCurrentClamp,
    Stimulus,
)
from axonfleet.utils import units


_EXTRACELLULAR_TOPOLOGY_REVISION = 0
_SIMULATION_STRUCTURE_REVISION = 0


def extracellular_topology_revision() -> int:
    """Return the process-wide revision for attached stimulation objects."""

    return _EXTRACELLULAR_TOPOLOGY_REVISION


def simulation_structure_revision() -> int:
    """Return the process-wide revision for dispatch-relevant instance state."""

    return _SIMULATION_STRUCTURE_REVISION


def _bump_simulation_structure_revision() -> None:
    global _SIMULATION_STRUCTURE_REVISION
    _SIMULATION_STRUCTURE_REVISION += 1


def _bump_extracellular_topology_revision() -> None:
    global _EXTRACELLULAR_TOPOLOGY_REVISION
    _EXTRACELLULAR_TOPOLOGY_REVISION += 1
    _bump_simulation_structure_revision()


class AxonInstance:
    """One concrete occurrence of a descriptive axon.

    `Axon` objects remain purely descriptive: geometry, layout, cable
    formulation, and membrane models. `AxonInstance` binds one descriptive axon
    to local stimulation contexts for a concrete run. It does not own
    anatomical/world placement; external geometry should be converted to
    sampled footprints/drives before attachment.

    Parameters
    ----------
    axon:
        Descriptive axon model.
    """

    def __init__(
        self,
        axon: Axon,
    ) -> None:
        if not isinstance(axon, Axon):
            raise TypeError("axon must be an axonfleet.axons.Axon instance.")
        self.axon = axon
        self.intracellular_contexts: list[IntracellularCurrentClamp] = []
        self.extracellular_stimulation: ExtracellularStimulation | None = None

    @property
    def dtype(self) -> np.dtype:
        """Numerical dtype derived from the wrapped axon's membrane layout."""

        return np.dtype(self.axon.layout.sections[0].membrane.dtype)

    def __getattr__(self, name: str) -> Any:
        """Delegate descriptive axon attributes to the wrapped axon."""

        return getattr(self.axon, name)

    @property
    def use_extracellular(self) -> bool:
        """Whether the solver should include extracellular handling."""

        return (
            self.axon.resolved_formulation == "double-cable"
            or self.extracellular_stimulation is not None
        )

    def add_intracellular_context(
        self,
        *,
        context: IntracellularCurrentClamp,
    ) -> None:
        """Attach an intracellular stimulation context.

        Parameters
        ----------
        context:
            Point current injection to attach to this axon instance.
        """

        if not isinstance(context, IntracellularCurrentClamp):
            raise TypeError(
                "context must be an axonfleet.stimulation.IntracellularCurrentClamp."
            )
        self.intracellular_contexts.append(context)
        _bump_simulation_structure_revision()

    def add_current_clamp(
        self,
        *,
        position: Any,
        current: Stimulus,
    ) -> None:
        """Convenience wrapper for adding an `IntracellularCurrentClamp`.

        `position` must carry length units. Plain stimulus amplitudes are
        interpreted as nanoamperes.
        """

        self.add_intracellular_context(
            context=IntracellularCurrentClamp(position=position, current=current)
        )

    def add_extracellular_stimulation(
        self,
        *,
        stimulation: ExtracellularStimulation,
        replace: bool = False,
    ) -> None:
        """Attach sampled extracellular stimulation.

        This is the canonical high-level extracellular path: helpers or
        external tools build `ExtracellularFootprint`/`ExtracellularDrive`
        objects, then the simulation receives the resulting immutable
        `ExtracellularStimulation`.
        """

        if not isinstance(stimulation, ExtracellularStimulation):
            raise TypeError(
                "stimulation must be an axonfleet.stimulation.ExtracellularStimulation."
            )
        if self.extracellular_stimulation is not None and not replace:
            raise ValueError(
                "AxonInstance accepts one extracellular stimulation. "
                "Use ExtracellularStimulation([...]) for multiple drives, "
                "or pass replace=True."
            )
        if self.extracellular_stimulation is not stimulation:
            self.extracellular_stimulation = stimulation
            _bump_extracellular_topology_revision()

def as_axon_instance(value: Axon | AxonInstance) -> AxonInstance:
    """Return `value` as an `AxonInstance`.

    Passing a pure `Axon` creates a no-stimulation instance around it. This
    keeps low-level solver calls convenient while preserving `Axon` as a pure
    descriptive object.
    """

    if isinstance(value, AxonInstance):
        return value
    if isinstance(value, Axon):
        return AxonInstance(value)
    raise TypeError(f"expected Axon or AxonInstance, got {type(value)!r}.")


__all__ = ["AxonInstance", "as_axon_instance"]
