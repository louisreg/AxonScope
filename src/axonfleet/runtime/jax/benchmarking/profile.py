"""JAX runtime support for host-side benchmark, estimate, and inspection tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from axonfleet.runtime.jax.inputs.lowering import (
    PlannedInputLowering,
    plan_input_lowering,
)
from axonfleet.solvers.options import BatchOptions


def benchmark_plan_input_lowering(
    *,
    group_mode: str,
    axons: Sequence[Any],
    stimulation_rows: Sequence[tuple[Any, ...]],
    kernel_options: BatchOptions,
    observers: tuple[Any, ...] | None,
) -> PlannedInputLowering:
    """Return runtime input-lowering formats without materializing arrays."""

    return plan_input_lowering(
        group_mode=group_mode,
        axons=axons,
        stimulation_rows=stimulation_rows,
        kernel_options=kernel_options,
        observers=observers,
    )


def benchmark_membrane_output_names(
    model: Any,
    method_name: str,
) -> tuple[str, ...]:
    """Return membrane output names, compiling the JAX model if needed."""

    method = getattr(model, method_name, None)
    if not callable(method):
        from axonfleet.runtime.jax.membranes.compile import compile_membrane_model

        model = compile_membrane_model(model)
        method = getattr(model, method_name, None)
    if not callable(method):
        return ()
    return tuple(str(name) for name in method())


@dataclass
class JaxBenchmarkProfile:
    """Handle for one active JAX profiler trace."""

    output_dir: Path
    create_perfetto_link: bool = False
    create_perfetto_trace: bool = False
    active: bool = False

    def start(self) -> "JaxBenchmarkProfile":
        import jax

        self.output_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(
            str(self.output_dir),
            create_perfetto_link=self.create_perfetto_link,
            create_perfetto_trace=self.create_perfetto_trace,
        )
        self.active = True
        return self

    def stop(self) -> dict[str, Any]:
        if not self.active:
            return {
                "enabled": True,
                "runtime": "jax",
                "output": str(self.output_dir),
                "stopped": False,
            }
        import jax

        jax.profiler.stop_trace()
        self.active = False
        return {
            "enabled": True,
            "runtime": "jax",
            "output": str(self.output_dir),
            "stopped": True,
            "view_hint": f"tensorboard --logdir {self.output_dir}",
        }


def benchmark_profile_start(
    output_dir: str | Path,
    *,
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
) -> JaxBenchmarkProfile:
    """Start a JAX profiler trace for benchmark instrumentation."""

    return JaxBenchmarkProfile(
        output_dir=Path(output_dir),
        create_perfetto_link=bool(create_perfetto_link),
        create_perfetto_trace=bool(create_perfetto_trace),
    ).start()


def benchmark_save_device_memory_profile(output_path: str | Path) -> dict[str, Any]:
    """Write a JAX device-memory profile in pprof format."""

    import jax

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jax.profiler.save_device_memory_profile(str(path))
    return {
        "enabled": True,
        "runtime": "jax",
        "path": str(path),
        "format": "pprof",
        "view_hint": f"pprof --web {path}",
    }
