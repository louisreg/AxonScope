"""Internal JAX output-plan descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from axonscope.solvers.options import BatchOptions, BatchRecording

OutputSink = Literal["vm", "none", "vm_raster"]


@dataclass(frozen=True)
class OutputPlan:
    """Backend-local output and chunking plan.

    Public recording requests are lowered to ``BatchOptions`` at the orchestration
    boundary. The JAX backend then normalizes those options into this internal
    plan so full Vm, sampled Vm, and observer-only VmRaster routes share one
    output descriptor.
    """

    recording: BatchRecording
    time_chunk_steps: int | None
    sink: OutputSink

    @classmethod
    def from_batch_options(
        cls,
        options: BatchOptions,
        *,
        observers: tuple[object, ...] | None,
    ) -> "OutputPlan":
        if options.recording.mode == "none":
            sink: OutputSink = "vm_raster" if observers else "none"
        else:
            sink = "vm"
        return cls(
            recording=options.recording,
            time_chunk_steps=options.time_chunk_steps,
            sink=sink,
        )

    def to_batch_options(self) -> BatchOptions:
        """Return the legacy structural object expected by result assembly."""

        return BatchOptions(
            recording=self.recording,
            time_chunk_steps=self.time_chunk_steps,
        )


__all__ = ["OutputPlan", "OutputSink"]
