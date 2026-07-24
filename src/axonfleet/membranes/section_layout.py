"""Named membrane assignment for anatomical section templates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from axonfleet.membranes.model import require_membrane_description


def _section_key(value: str) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True)
class SectionLayout:
    """Assign descriptive membrane models to named anatomical cable sections."""

    sections: Mapping[str, object]

    def __init__(self, **sections: Any) -> None:
        if not sections:
            raise ValueError("SectionLayout requires at least one section membrane.")
        normalized = {
            _section_key(name): require_membrane_description(model)
            for name, model in sections.items()
        }
        object.__setattr__(self, "sections", MappingProxyType(normalized))

    def membrane_for(self, section: str) -> object:
        """Return the membrane assigned to `section`."""

        key = _section_key(section)
        try:
            return self.sections[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.sections))
            raise KeyError(f"unknown section {section!r}; available sections: {available}") from exc


__all__ = ["SectionLayout"]
