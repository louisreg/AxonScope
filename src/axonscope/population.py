"""Public axon population container."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import overload

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.axons.axon import Axon


AxonPopulationInput = Axon | AxonInstance | Iterable[Axon | AxonInstance]


class AxonPopulation(Sequence[AxonInstance]):
    """Typed collection of concrete axon instances.

    `AxonPopulation` is the public object for cohorts. It stores
    `AxonInstance` rows, preserves input order, and treats a one-row cohort as
    the smallest population case. Passing a pure `Axon` creates a default
    no-stimulation instance around it, matching the public simulation wrappers.
    """

    def __init__(
        self,
        axons: AxonPopulationInput,
        *,
        name: str | None = None,
    ) -> None:
        self._items = _normalize_population_items(axons)
        self.name = name

    @classmethod
    def single(
        cls,
        axon: Axon | AxonInstance,
        *,
        name: str | None = None,
    ) -> "AxonPopulation":
        """Build a one-row population."""

        return cls((axon,), name=name)

    @classmethod
    def from_axons(
        cls,
        axons: Iterable[Axon | AxonInstance],
        *,
        name: str | None = None,
    ) -> "AxonPopulation":
        """Build a population from an iterable of axons or instances."""

        return cls(axons, name=name)

    @overload
    def __getitem__(self, index: int) -> AxonInstance:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[AxonInstance, ...]:
        ...

    def __getitem__(self, index: int | slice) -> AxonInstance | tuple[AxonInstance, ...]:
        return self._items[index]

    def __iter__(self) -> Iterator[AxonInstance]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def axons(self) -> tuple[Axon, ...]:
        """Return the descriptive axons in population order."""

        return tuple(item.axon for item in self._items)

    @property
    def instances(self) -> tuple[AxonInstance, ...]:
        """Return concrete instances in population order."""

        return self._items

    @property
    def is_single(self) -> bool:
        """Return whether the population contains exactly one instance."""

        return len(self._items) == 1

    @property
    def is_empty(self) -> bool:
        """Return whether the population has no instances."""

        return len(self._items) == 0

    def __repr__(self) -> str:
        label = f", name={self.name!r}" if self.name is not None else ""
        return f"AxonPopulation(n={len(self)}{label})"


def _normalize_population_items(axons: AxonPopulationInput) -> tuple[AxonInstance, ...]:
    """Normalize public population input into concrete instances."""

    if isinstance(axons, (Axon, AxonInstance)):
        candidates = (axons,)
    else:
        try:
            candidates = tuple(axons)
        except TypeError as exc:
            raise TypeError(
                "AxonPopulation expects an Axon, AxonInstance, or iterable of "
                "Axon/AxonInstance objects."
            ) from exc

    if not candidates:
        raise ValueError("AxonPopulation requires at least one axon or instance.")

    items: list[AxonInstance] = []
    invalid: list[str] = []
    for index, value in enumerate(candidates):
        try:
            items.append(as_axon_instance(value))
        except TypeError:
            invalid.append(f"{index}: {type(value).__name__}")
    if invalid:
        detail = ", ".join(invalid)
        raise TypeError(f"AxonPopulation received invalid entries: {detail}.")
    return tuple(items)


__all__ = ["AxonPopulation", "AxonPopulationInput"]
