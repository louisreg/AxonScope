from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from axonscope.axon_instance import AxonInstance
from axonscope.axons.axon import Axon
from axonscope.results import SimResult


class Solver(ABC):
    """Abstract base class for temporal axon solvers.

    Solver implementations consume ``AxonInstance`` protocols, or pure
    descriptive ``Axon`` objects that are wrapped as no-stimulation protocols.
    Direct solver calls interpret plain time values as milliseconds and also
    accept Pint-like time quantities. Public recording objects are handled by
    the higher-level simulation facade.
    """

    @abstractmethod
    def solve(
        self,
        axon: Axon | AxonInstance,
        tsim: Any | None = None,
        dt: Any | None = None,
        record_diagnostics: bool = False,
        record_observables: bool = False,
    ) -> SimResult:
        """Run a simulation and return voltage traces plus metadata."""

        raise NotImplementedError
