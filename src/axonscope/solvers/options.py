from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

BatchRecordingMode = Literal["full", "center", "probes", "indices", "none"]
DEFAULT_OBSERVER_TIME_CHUNK_STEPS = 128


@dataclass(frozen=True)
class SolverOptions:
    """Reserved numerical options consumed by runtime preparation.

    These options belong to solver construction/execution, not to dispatch
    planning. Dispatchers may forward this object unchanged when they decide
    whether a group should run through scalar or batch kernels.
    """


@dataclass(frozen=True)
class BatchRecording:
    """Vm storage policy used by batch solver kernels.

    The policy is intentionally solver-side: it controls how much of the
    computed Vm tensor the batch kernels retain. Public recording requests are
    translated into this smaller numerical policy before execution.
    """

    mode: BatchRecordingMode = "full"
    values: tuple[int, ...] = ()
    probe_count: int = 8

    @classmethod
    def full(cls) -> "BatchRecording":
        """Record all compartments."""

        return cls(mode="full")

    @classmethod
    def center(cls) -> "BatchRecording":
        """Record only the center compartment."""

        return cls(mode="center")

    @classmethod
    def probes(cls, count: int = 8) -> "BatchRecording":
        """Record evenly spaced compartment probes."""

        return cls(mode="probes", probe_count=int(count))

    @classmethod
    def indices(cls, values: Sequence[int]) -> "BatchRecording":
        """Record explicit compartment indices."""

        return cls(mode="indices", values=tuple(int(value) for value in values))

    @classmethod
    def none(cls) -> "BatchRecording":
        """Record no Vm trace."""

        return cls(mode="none")

    @classmethod
    def from_mode(cls, mode: str, *, probe_count: int = 8) -> "BatchRecording":
        """Build a recording policy from a simple mode string."""

        if mode == "full":
            return cls.full()
        if mode == "center":
            return cls.center()
        if mode == "probes":
            return cls.probes(probe_count)
        if mode == "none":
            return cls.none()
        raise ValueError(f"unknown recording mode: {mode!r}.")

    def __post_init__(self) -> None:
        if self.mode not in {"full", "center", "probes", "indices", "none"}:
            raise ValueError(f"unknown recording mode: {self.mode!r}.")
        if int(self.probe_count) < 1:
            raise ValueError("probe_count must be >= 1.")
        if self.mode == "indices" and not self.values:
            raise ValueError("indices recording requires at least one index.")

    @property
    def label(self) -> str:
        """Short label suitable for benchmark output."""

        return self.mode

    @property
    def is_full(self) -> bool:
        """Whether the policy retains all compartments."""

        return self.mode == "full"

    def indices_for(self, nx: int) -> np.ndarray | None:
        """Return retained compartment indices, or ``None`` for full Vm."""

        if nx < 1:
            raise ValueError("nx must be >= 1.")
        if self.mode == "none":
            return np.asarray([], dtype=np.int32)
        if self.mode == "full":
            return None
        if self.mode == "center":
            return np.asarray([nx // 2], dtype=np.int32)
        if self.mode == "probes":
            count = min(int(self.probe_count), nx)
            return np.unique(np.linspace(0, nx - 1, count, dtype=np.int32))

        indices = np.asarray(self.values, dtype=np.int32)
        if np.any(indices < 0) or np.any(indices >= nx):
            raise ValueError(
                f"recording indices must be within [0, {nx}), got {indices}."
            )
        return indices

    def width_for(self, nx: int) -> int:
        """Return the number of retained Vm columns for ``nx`` compartments."""

        indices = self.indices_for(nx)
        return nx if indices is None else int(len(indices))


@dataclass(frozen=True)
class BatchOptions:
    """Execution options consumed by batch solver kernels."""

    recording: BatchRecording = field(default_factory=BatchRecording.full)
    time_chunk_steps: int | None = None

    @classmethod
    def full(
        cls,
        *,
        time_chunk_steps: int | None = None,
    ) -> "BatchOptions":
        """Record full Vm, optionally chunking the time loop."""

        return cls(
            recording=BatchRecording.full(),
            time_chunk_steps=time_chunk_steps,
        )

    @classmethod
    def center(
        cls,
        *,
        time_chunk_steps: int | None = None,
    ) -> "BatchOptions":
        """Record the center compartment only."""

        return cls(
            recording=BatchRecording.center(),
            time_chunk_steps=time_chunk_steps,
        )

    @classmethod
    def probes(
        cls,
        count: int = 8,
        *,
        time_chunk_steps: int | None = None,
    ) -> "BatchOptions":
        """Record evenly spaced compartment probes."""

        return cls(
            recording=BatchRecording.probes(count),
            time_chunk_steps=time_chunk_steps,
        )

    @classmethod
    def none(
        cls,
        *,
        time_chunk_steps: int | None = DEFAULT_OBSERVER_TIME_CHUNK_STEPS,
    ) -> "BatchOptions":
        """Record no Vm trace, typically for solver-side observer runs.

        Observer-only runs default to a stable, VmRaster word-aligned time chunk
        to reduce first-call JAX recompilation across duration sweeps while the
        runtime writes into one full-duration packed VmRaster state. Pass
        ``time_chunk_steps=None`` explicitly to force one unchunked scan.
        """

        return cls(
            recording=BatchRecording.none(),
            time_chunk_steps=time_chunk_steps,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.recording, BatchRecording):
            raise TypeError("recording must be a BatchRecording.")
        if self.time_chunk_steps is not None and int(self.time_chunk_steps) < 1:
            raise ValueError("time_chunk_steps must be >= 1.")


__all__ = [
    "BatchOptions",
    "BatchRecording",
    "BatchRecordingMode",
    "DEFAULT_OBSERVER_TIME_CHUNK_STEPS",
    "SolverOptions",
]
