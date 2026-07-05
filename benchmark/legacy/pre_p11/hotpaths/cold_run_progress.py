"""Show progress, inspection, estimates, and hotpath timing for one run shape.

Run:
    python benchmark/hotpaths/cold_run_progress.py

Cold JAX calls can spend most of their first execution compiling. This benchmark
probe uses the same `AxonSimulation` workflow for planning (`inspect()`),
memory/output sizing (`estimate()`), progress reporting (`progress=True`), and
hotpath benchmark spans (`axs.benchmark(...)`). The goal is to understand what
the run is doing, not to publish a reproducible performance comparison.
"""

from __future__ import annotations

import tempfile

from rich.console import Console
from rich.table import Table

import axonscope as axs


def _build_simulation(
    *,
    policy: axs.ExecutionPolicy,
    progress: bool | str = False,
) -> axs.AxonSimulation:
    """Build the fixed-shape pool used by inspect, estimate, and run."""

    # The two rows differ only by stimulus amplitude, so the dispatcher can use
    # the public batch route and the second execution sees the same JAX shape.
    simulations: list[axs.AxonInstance] = []
    for amplitude in (0.35, 0.55):
        axon = axs.axons.HodgkinHuxley(
            length=80.0 * axs.um,
            diameter=0.7 * axs.um,
            compartments=9,
            celsius=6.3 * axs.degC,
        )
        simulation = axs.AxonInstance(axon)
        simulation.add_current_clamp(
            position=40.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.05 * axs.ms,
                duration=0.05 * axs.ms,
                amplitude=amplitude * axs.nA,
            ),
        )
        simulations.append(simulation)

    return axs.AxonSimulation(
        simulations,
        duration=0.20 * axs.ms,
        dt=0.05 * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        execution_policy=policy,
        progress=progress,
    )


def _span_duration_ms(report: axs.BenchmarkReport, name: str) -> float | None:
    for event in report.events:
        if event.name == name:
            return event.duration_ms
    return None


def _duration_text(value_ms: float | None) -> str:
    return "n/a" if value_ms is None else f"{value_ms / 1000.0:.3f} s"


def main() -> None:
    console = Console()

    # Step 1: keep runtime choices explicit. Inspection, estimates, progress,
    # and benchmark metadata will then describe the same backend/device/precision.
    policy = axs.ExecutionPolicy(
        runtime=axs.Runtime.JAX,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )
    planned = _build_simulation(policy=policy)

    # Step 2: dry-run the execution shape before launching the kernel. This
    # answers "which route will run?" and "what output/memory footprint should I
    # expect?" without making a cold JAX call.
    console.rule("Inspection")
    planned.inspect().print()

    console.rule("Estimate")
    planned.estimate().print()

    # Step 3: benchmark the first and second run with named spans while progress
    # reports the live dispatch/backend stages. Use `progress="plain"` instead
    # of `True` when capturing logs in a non-interactive environment.
    with tempfile.TemporaryDirectory(prefix="axonscope-progress-") as output_dir:
        with axs.benchmark(
            output_dir,
            print_summary=False,
            save=False,
            sync_device=True,
        ) as benchmark_session:
            console.rule("Cold call")
            with benchmark_session.span("hotpaths.cold_run"):
                cold = _build_simulation(policy=policy, progress=True).run()

            console.rule("Warm call")
            with benchmark_session.span("hotpaths.warm_run"):
                warm = _build_simulation(policy=policy, progress=True).run()

            report = benchmark_session.report()

    # Step 4: keep the user-facing summary compact. The full benchmark report is
    # printed below so the same run also exposes dispatch, lowering, and kernel
    # hotpaths.
    cold_elapsed = _span_duration_ms(report, "hotpaths.cold_run")
    warm_elapsed = _span_duration_ms(report, "hotpaths.warm_run")

    summary = Table(title="Cold Versus Warm Hotpath Probe")
    summary.add_column("run")
    summary.add_column("elapsed", justify="right")
    summary.add_column("Vm shape")
    summary.add_row("cold", _duration_text(cold_elapsed), str(cold[0].Vm.shape))
    summary.add_row("warm", _duration_text(warm_elapsed), str(warm[0].Vm.shape))
    console.print(summary)
    console.rule("Benchmark hotpaths")
    console.print(report.format())


if __name__ == "__main__":
    main()
