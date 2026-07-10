"""Choose typed single-cable and double-cable solver policies.

Run:
    python examples/advanced/runtime/04_solver_policy.py

Solver selection is part of `ExecutionPolicy`, next to runtime, device, and
precision. `Device` selects CPU/GPU; `SolverPolicy` selects the numerical route
by cable family through runtime-specific constructors. `BatchOptions` only
controls output/chunking details.
"""

from __future__ import annotations

import axonscope as axs


def _double_cable_population() -> axs.AxonPopulation:
    rows = []
    for diameter_um in (3.0, 4.0):
        axon = axs.axons.MRG(
            diameter=diameter_um * axs.um,
            nodes=3,
        )
        rows.append(axs.AxonInstance(axon))
    return axs.AxonPopulation(rows)


def _inspection_for(policy: axs.ExecutionPolicy) -> axs.SimulationInspection:
    simulation = axs.AxonSimulation(
        _double_cable_population(),
        duration=0.20 * axs.ms,
        dt=0.02 * axs.ms,
        recording=axs.Recording.probes(axs.signals.Vm, count=3),
        batch_options=axs.BatchOptions.full(time_chunk_steps=10),
        execution_policy=policy,
    )
    return simulation.inspect()


def main() -> None:
    cpu_policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
        solvers=axs.SolverPolicy(
            double_cable=axs.runtime.jax.cpu.DoubleCableSolver.thomas()
        ),
    )
    gpu_policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.gpu(0),
        precision=axs.PrecisionPolicy.float32(),
        solvers=axs.SolverPolicy(
            single_cable=axs.runtime.jax.gpu.SingleCableSolver.jax_tridiagonal(),
            double_cable=axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
                block_b=64
            ),
        ),
    )

    for label, policy in (("CPU", cpu_policy), ("GPU", gpu_policy)):
        report = _inspection_for(policy)
        kernel = report.kernels[0]
        print(f"\n{label} policy")
        print(f"  device: {policy.device.kind}")
        print(f"  precision: {policy.precision.solver_dtype}")
        print(f"  kernel: {kernel.kernel}")
        print(f"  double-cable solver: {kernel.double_cable_block_solver}")


if __name__ == "__main__":
    main()
