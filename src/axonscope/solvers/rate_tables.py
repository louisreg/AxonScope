"""Public solver-side rate-table configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateTableConfig:
    """Voltage grid used to tabulate membrane alpha/beta rate constants."""

    v_min_mV: float = -120.0
    v_max_mV: float = 80.0
    step_mV: float = 0.05
    clamp: bool = True

    def validate(self) -> None:
        if self.step_mV <= 0:
            raise ValueError(f"step_mV must be positive, got {self.step_mV}.")
        if self.v_max_mV <= self.v_min_mV:
            raise ValueError(
                "v_max_mV must be greater than v_min_mV, "
                f"got {self.v_min_mV}..{self.v_max_mV}."
            )


__all__ = ["RateTableConfig"]
