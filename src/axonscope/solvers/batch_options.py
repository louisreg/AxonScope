from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class BatchRecording:
    """Vm recording policy for population batch solvers."""

    mode: Literal["full", "center", "probes", "indices"] = "full"
    values: tuple[int, ...] = ()
    probe_count: int = 8

    @classmethod
    def full(cls) -> "BatchRecording":
        return cls(mode="full")

    @classmethod
    def center(cls) -> "BatchRecording":
        return cls(mode="center")

    @classmethod
    def probes(cls, count: int = 8) -> "BatchRecording":
        return cls(mode="probes", probe_count=int(count))

    @classmethod
    def indices(cls, values: Sequence[int]) -> "BatchRecording":
        return cls(mode="indices", values=tuple(int(value) for value in values))

    @classmethod
    def from_mode(cls, mode: str, *, probe_count: int = 8) -> "BatchRecording":
        if mode == "full":
            return cls.full()
        if mode == "center":
            return cls.center()
        if mode == "probes":
            return cls.probes(probe_count)
        raise ValueError(f"unknown recording mode: {mode!r}.")

    def __post_init__(self) -> None:
        if self.mode not in {"full", "center", "probes", "indices"}:
            raise ValueError(f"unknown recording mode: {self.mode!r}.")
        if self.probe_count < 1:
            raise ValueError("probe_count must be >= 1.")
        if self.mode == "indices" and not self.values:
            raise ValueError("indices recording requires at least one index.")

    @property
    def label(self) -> str:
        return self.mode

    @property
    def is_full(self) -> bool:
        return self.mode == "full"

    def indices_for(self, nx: int) -> np.ndarray | None:
        if nx < 1:
            raise ValueError("nx must be >= 1.")
        if self.mode == "full":
            return None
        if self.mode == "center":
            return np.asarray([nx // 2], dtype=np.int32)
        if self.mode == "probes":
            count = min(int(self.probe_count), nx)
            return np.unique(np.linspace(0, nx - 1, count, dtype=np.int32))

        indices = np.asarray(self.values, dtype=np.int32)
        if np.any(indices < 0) or np.any(indices >= nx):
            raise ValueError(f"recording indices must be within [0, {nx}), got {indices}.")
        return indices

    def width_for(self, nx: int) -> int:
        indices = self.indices_for(nx)
        return nx if indices is None else int(len(indices))


@dataclass(frozen=True)
class BatchOptions:
    """Execution options for population batch solvers."""

    recording: BatchRecording = field(default_factory=BatchRecording.full)
    time_chunk_steps: int | None = None

    @classmethod
    def full(cls) -> "BatchOptions":
        return cls(recording=BatchRecording.full())

    @classmethod
    def center(cls, *, time_chunk_steps: int | None = None) -> "BatchOptions":
        return cls(recording=BatchRecording.center(), time_chunk_steps=time_chunk_steps)

    @classmethod
    def probes(
        cls,
        count: int = 8,
        *,
        time_chunk_steps: int | None = None,
    ) -> "BatchOptions":
        return cls(recording=BatchRecording.probes(count), time_chunk_steps=time_chunk_steps)

    def __post_init__(self) -> None:
        if not isinstance(self.recording, BatchRecording):
            raise TypeError("recording must be a BatchRecording.")
        if self.time_chunk_steps is not None and int(self.time_chunk_steps) < 1:
            raise ValueError("time_chunk_steps must be >= 1.")


__all__ = ["BatchOptions", "BatchRecording"]
