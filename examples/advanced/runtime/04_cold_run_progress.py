"""Show human-readable progress during a cold solver run.

Run:
    python examples/advanced/runtime/04_cold_run_progress.py

Cold JAX calls can spend most of their first execution compiling. The
`progress` option reports the active dispatch/backend step so a first call does
not look stalled, while keeping the report wired to the structured dispatcher
events used by inspection and benchmarking.
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

import axonscope as axs


def main() -> None:
    console = Console()

    # Step 1: keep runtime choices explicit. This makes the progress report and
    # any later inspection report describe the same backend/device/precision.
    policy = axs.ExecutionPolicy(
        runtime=axs.Runtime.JAX,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )

    # Step 2: build a tiny homogeneous pool so the dispatcher takes the public
    # batch route. The two rows differ only by stimulus amplitude, so they share
    # the same geometry/runtime signature.
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

    recording = axs.Recording.center(axs.signals.Vm)

    # Step 3: the first call may compile. `progress=True` uses the Rich live
    # display; use `progress="plain"` instead when capturing text logs.
    console.rule("First call")
    first_start = time.perf_counter()
    first = axs.simulate_pool(
        simulations,
        duration=0.20 * axs.ms,
        dt=0.05 * axs.ms,
        recording=recording,
        execution_policy=policy,
        progress=True,
    )
    first_elapsed = time.perf_counter() - first_start

    # Step 4: running the same shape again should use the warm path. The same
    # progress events appear, but the "compiling JAX kernel if needed" step
    # should be much shorter once JAX has cached the compiled executable.
    console.rule("Warm call")
    warm_start = time.perf_counter()
    warm = axs.simulate_pool(
        simulations,
        duration=0.20 * axs.ms,
        dt=0.05 * axs.ms,
        recording=recording,
        execution_policy=policy,
        progress=True,
    )
    warm_elapsed = time.perf_counter() - warm_start

    summary = Table(title="Cold Versus Warm Runtime")
    summary.add_column("run")
    summary.add_column("elapsed", justify="right")
    summary.add_column("Vm shape")
    summary.add_row("first", f"{first_elapsed:.3f} s", str(first[0].Vm.shape))
    summary.add_row("warm", f"{warm_elapsed:.3f} s", str(warm[0].Vm.shape))
    console.print(summary)


if __name__ == "__main__":
    main()
