"""Performance and memory-estimation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from axonscope.axon_instance import AxonInstance, as_axon_instance
from axonscope.axons.axon import Axon
from axonscope.population import AxonPopulation
from axonscope.recording import Recording
from axonscope.solvers import BatchOptions
from axonscope.solvers.common import simulation_step_count
from axonscope.utils import units


class Runtime(Enum):
    """Runtime target used by performance planning."""

    AUTO = "auto"
    NUMPY = "numpy"
    JAX = "jax"


DeviceKind = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True)
class Device:
    """Structured runtime device request."""

    kind: DeviceKind
    index: int | None = None

    @classmethod
    def auto(cls) -> "Device":
        """Let AxonScope or the backend choose a device."""

        return cls("auto")

    @classmethod
    def cpu(cls) -> "Device":
        """Request CPU execution."""

        return cls("cpu")

    @classmethod
    def gpu(cls, index: int = 0) -> "Device":
        """Request one GPU device by index."""

        return cls("gpu", int(index))

    def __post_init__(self) -> None:
        if self.kind not in {"auto", "cpu", "gpu"}:
            raise ValueError("Device kind must be 'auto', 'cpu', or 'gpu'.")
        if self.kind == "gpu":
            if self.index is None or int(self.index) < 0:
                raise ValueError("GPU device index must be >= 0.")
            object.__setattr__(self, "index", int(self.index))
        elif self.index is not None:
            raise ValueError("Only GPU devices accept an index.")


@dataclass(frozen=True)
class PrecisionPolicy:
    """Dtype policy used by estimators and future runtime lowering."""

    state_dtype: str
    solver_dtype: str
    accumulation_dtype: str

    @classmethod
    def float32(cls) -> "PrecisionPolicy":
        """Use float32 for state, solver inputs, and reductions."""

        return cls("float32", "float32", "float32")

    @classmethod
    def float64(cls) -> "PrecisionPolicy":
        """Use float64 for state, solver inputs, and reductions."""

        return cls("float64", "float64", "float64")

    @classmethod
    def mixed(
        cls,
        *,
        state_dtype: Any = "float32",
        solver_dtype: Any = "float32",
        accumulation_dtype: Any = "float64",
    ) -> "PrecisionPolicy":
        """Build an explicit mixed-precision policy."""

        return cls(
            _dtype_name(state_dtype),
            _dtype_name(solver_dtype),
            _dtype_name(accumulation_dtype),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dtype", _dtype_name(self.state_dtype))
        object.__setattr__(self, "solver_dtype", _dtype_name(self.solver_dtype))
        object.__setattr__(
            self,
            "accumulation_dtype",
            _dtype_name(self.accumulation_dtype),
        )


@dataclass(frozen=True)
class MemoryEstimateItem:
    """One array-like contribution to a simulation memory estimate."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    role: str
    retained: bool
    note: str = ""

    @property
    def mib(self) -> float:
        """Size in MiB."""

        return self.bytes / (1024**2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "bytes": self.bytes,
            "mib": self.mib,
            "role": self.role,
            "retained": self.retained,
            "note": self.note,
        }


@dataclass(frozen=True)
class SimulationEstimate:
    """Memory and workload estimate for one public simulation definition."""

    axon_count: int
    step_count: int
    max_compartments: int
    duration_ms: float
    dt_ms: float
    runtime: Runtime
    device: Device
    precision: PrecisionPolicy
    recording_width_max: int
    items: tuple[MemoryEstimateItem, ...]
    warnings: tuple[str, ...]
    recommendations: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def total_bytes(self) -> int:
        """Sum all estimated array contributions."""

        return sum(item.bytes for item in self.items)

    @property
    def total_mib(self) -> float:
        """Total estimated size in MiB."""

        return self.total_bytes / (1024**2)

    @property
    def retained_bytes(self) -> int:
        """Bytes expected to remain visible in public outputs."""

        return sum(item.bytes for item in self.items if item.retained)

    @property
    def retained_mib(self) -> float:
        """Retained public output size in MiB."""

        return self.retained_bytes / (1024**2)

    def item(self, name: str) -> MemoryEstimateItem:
        """Return one estimate row by name."""

        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable estimate."""

        return {
            "axon_count": self.axon_count,
            "step_count": self.step_count,
            "max_compartments": self.max_compartments,
            "duration_ms": self.duration_ms,
            "dt_ms": self.dt_ms,
            "runtime": self.runtime.value,
            "device": {"kind": self.device.kind, "index": self.device.index},
            "precision": {
                "state_dtype": self.precision.state_dtype,
                "solver_dtype": self.precision.solver_dtype,
                "accumulation_dtype": self.precision.accumulation_dtype,
            },
            "recording_width_max": self.recording_width_max,
            "total_bytes": self.total_bytes,
            "total_mib": self.total_mib,
            "retained_bytes": self.retained_bytes,
            "retained_mib": self.retained_mib,
            "items": [item.to_dict() for item in self.items],
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }

    def format(self) -> str:
        """Format a compact human-readable report."""

        lines = [
            "AxonScope simulation estimate",
            (
                f"  axons={self.axon_count}, Nt={self.step_count}, "
                f"max Nx={self.max_compartments}, dtype={self.precision.solver_dtype}"
            ),
            (
                f"  total={self.total_mib:.3f} MiB, "
                f"retained={self.retained_mib:.3f} MiB"
            ),
            "  arrays:",
        ]
        for item in self.items:
            shape = "x".join(str(dim) for dim in item.shape) or "scalar"
            kept = "retained" if item.retained else "temporary"
            lines.append(
                f"    {item.name:36s} {item.mib:9.3f} MiB  "
                f"{shape:14s} {kept:9s} {item.role}"
            )
        if self.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {warning}" for warning in self.warnings)
        if self.recommendations:
            lines.append("  recommendations:")
            lines.extend(f"    - {recommendation}" for recommendation in self.recommendations)
        return "\n".join(lines)


def estimate_simulation(
    axons: Axon | AxonInstance | AxonPopulation | Iterable[Axon | AxonInstance],
    *,
    duration: Any,
    dt: Any,
    recording: Recording | None = None,
    batch_options: BatchOptions | None = None,
    observers: Sequence[Any] | None = None,
    population_lifecycle: bool | None = None,
    runtime: Runtime = Runtime.AUTO,
    device: Device | None = None,
    precision: PrecisionPolicy | None = None,
    memory_budget_bytes: int | None = None,
) -> SimulationEstimate:
    """Estimate memory pressure for a public simulation definition."""

    population = axons if isinstance(axons, AxonPopulation) else AxonPopulation(axons)
    instances = tuple(population.instances)
    duration_ms = units.to_ms(duration)
    dt_ms = units.to_ms(dt)
    step_count = simulation_step_count(duration_ms, dt_ms)
    resolved_precision = precision or _precision_from_instances(instances)
    resolved_runtime = _coerce_runtime(runtime)
    resolved_device = Device.auto() if device is None else device
    dtype = np.dtype(resolved_precision.solver_dtype)
    state_dtype = np.dtype(resolved_precision.state_dtype)
    max_nx = max(int(instance.axon.n_compartments) for instance in instances)
    axon_count = len(instances)
    is_population_run = (
        not population.is_single
        if population_lifecycle is None
        else bool(population_lifecycle)
    )
    recording_widths = _recording_widths(
        instances,
        recording=recording,
        batch_options=batch_options,
        is_population=is_population_run,
    )
    recording_width_max = max(recording_widths)

    items: list[MemoryEstimateItem] = []
    items.append(
        _item(
            "state.vm",
            (axon_count, max_nx),
            state_dtype,
            role="solver_state",
            retained=False,
            note="current membrane voltage state",
        )
    )
    items.append(
        _item(
            "inputs.positions",
            (axon_count, max_nx),
            dtype,
            role="preparation",
            retained=False,
            note="batched intrinsic positions",
        )
    )
    items.append(
        _item(
            "inputs.intracellular_current_density",
            (axon_count, step_count, max_nx),
            dtype,
            role="kernel_input",
            retained=False,
            note="current batch backend materializes Iinj[B,Nt,Nx]",
        )
    )

    context_count = _context_count(instances)
    electrode_rows = _electrode_row_count(instances)
    stimulus_count = _unique_stimulus_count(instances)
    if context_count > 0:
        items.append(
            _item(
                "inputs.extracellular_potential_mid",
                (axon_count, step_count, max_nx),
                dtype,
                role="kernel_input",
                retained=False,
                note="current batch backend materializes Vstim[B,Nt,Nx]",
            )
        )
        if electrode_rows:
            items.append(
                _item(
                    "footprints.factorized_rows",
                    (electrode_rows, max_nx),
                    dtype,
                    role="factorized_reference",
                    retained=False,
                    note="spatial footprint storage without the time axis",
                )
            )
    if stimulus_count:
        items.append(
            _item(
                "stimuli.sampled_waveforms",
                (stimulus_count, step_count),
                dtype,
                role="factorized_reference",
                retained=False,
                note="temporal stimulus samples without the spatial axis",
            )
        )

    items.append(
        _item(
            "outputs.recorded_vm",
            (axon_count, step_count, recording_width_max),
            dtype,
            role="public_output",
            retained=True,
            note="retained Vm according to the current recording policy",
        )
    )

    warnings, recommendations = _estimate_guidance(
        items,
        context_count=context_count,
        max_nx=max_nx,
        recording_width_max=recording_width_max,
        memory_budget_bytes=memory_budget_bytes,
        observers=observers,
    )

    return SimulationEstimate(
        axon_count=axon_count,
        step_count=step_count,
        max_compartments=max_nx,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        runtime=resolved_runtime,
        device=resolved_device,
        precision=resolved_precision,
        recording_width_max=recording_width_max,
        items=tuple(items),
        warnings=warnings,
        recommendations=recommendations,
        metadata={
            "context_count": context_count,
            "electrode_rows": electrode_rows,
            "unique_stimulus_count": stimulus_count,
            "recording_policy": _recording_label(recording, batch_options),
            "population_lifecycle": is_population_run,
        },
    )


def _item(
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    role: str,
    retained: bool,
    note: str = "",
) -> MemoryEstimateItem:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return MemoryEstimateItem(
        name=name,
        shape=shape,
        dtype=str(dtype),
        bytes=size * int(dtype.itemsize),
        role=role,
        retained=retained,
        note=note,
    )


def _coerce_runtime(value: Runtime) -> Runtime:
    if isinstance(value, Runtime):
        return value
    raise TypeError("runtime must be an axonscope.Runtime value.")


def _dtype_name(value: Any) -> str:
    return str(np.dtype(value))


def _precision_from_instances(instances: Sequence[AxonInstance]) -> PrecisionPolicy:
    itemsize = max(np.dtype(instance.dtype).itemsize for instance in instances)
    dtype = "float64" if itemsize > np.dtype("float32").itemsize else "float32"
    return PrecisionPolicy(dtype, dtype, dtype)


def _recording_widths(
    instances: Sequence[AxonInstance],
    *,
    recording: Recording | None,
    batch_options: BatchOptions | None,
    is_population: bool,
) -> tuple[int, ...]:
    if is_population:
        if recording is not None:
            policy = recording.to_batch_options().recording
        elif batch_options is not None:
            policy = batch_options.recording
        else:
            policy = BatchOptions().recording
        return tuple(policy.width_for(int(instance.axon.n_compartments)) for instance in instances)
    return tuple(int(instance.axon.n_compartments) for instance in instances)


def _context_rows(instances: Sequence[AxonInstance]) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for instance in instances:
        rows.append(tuple(getattr(instance, "extracellular_contexts", ())))
    return tuple(rows)


def _context_count(instances: Sequence[AxonInstance]) -> int:
    return sum(len(row) for row in _context_rows(instances))


def _electrode_row_count(instances: Sequence[AxonInstance]) -> int:
    total = 0
    for row in _context_rows(instances):
        for context in row:
            total += len(tuple(getattr(context, "electrodes", ())))
    return total


def _unique_stimulus_count(instances: Sequence[AxonInstance]) -> int:
    seen: set[int] = set()
    for instance in instances:
        for context in getattr(instance, "intracellular_contexts", ()):
            stimulus = getattr(context, "current", None)
            if stimulus is not None:
                seen.add(id(stimulus))
        for row in getattr(instance, "extracellular_contexts", ()):
            for electrode in getattr(row, "electrodes", ()):
                stimulus = getattr(electrode, "stimulus", None)
                if stimulus is not None:
                    seen.add(id(stimulus))
    return len(seen)


def _estimate_guidance(
    items: Sequence[MemoryEstimateItem],
    *,
    context_count: int,
    max_nx: int,
    recording_width_max: int,
    memory_budget_bytes: int | None,
    observers: Sequence[Any] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    recommendations: list[str] = []
    item_map = {item.name: item for item in items}

    dense_vext = item_map.get("inputs.extracellular_potential_mid")
    factorized = item_map.get("footprints.factorized_rows")
    if dense_vext is not None:
        warnings.append(
            "Current batch extracellular lowering may materialize dense "
            f"Vstim[B,Nt,Nx] ({dense_vext.mib:.3f} MiB for this run)."
        )
        if factorized is not None and dense_vext.bytes > max(1, 8 * factorized.bytes):
            recommendations.append(
                "Keep Phase 7.5 focused on in-kernel observer/drive reductions: "
                f"factorized footprints are {factorized.mib:.3f} MiB before the time axis."
            )

    output = item_map.get("outputs.recorded_vm")
    if output is not None and recording_width_max >= max_nx:
        warnings.append(
            "Full Vm recording retains every compartment; large pools should prefer "
            "center/probe recordings or future observer-only runs."
        )
    if observers:
        recommendations.append(
            "Solver-side observers are not wired yet; Phase 7.5 should lower these "
            "specs into compact per-step kernel state."
        )
    if memory_budget_bytes is not None:
        total = sum(item.bytes for item in items)
        if total > int(memory_budget_bytes):
            warnings.append(
                f"Estimated arrays exceed memory_budget_bytes={int(memory_budget_bytes)} "
                f"({total} bytes estimated)."
            )

    if context_count == 0:
        recommendations.append(
            "No extracellular contexts detected; use this as the zero-field baseline "
            "when comparing CPU/GPU hotpath traces."
        )
    return tuple(warnings), tuple(recommendations)


def _recording_label(recording: Recording | None, batch_options: BatchOptions | None) -> str:
    if recording is not None:
        return recording.spatial.value
    if batch_options is not None:
        return batch_options.recording.label
    return "full"


__all__ = [
    "Device",
    "MemoryEstimateItem",
    "PrecisionPolicy",
    "Runtime",
    "SimulationEstimate",
    "estimate_simulation",
]
