"""Inspect solver planning on a heterogeneous pool.

Run:
    python examples/advanced/runtime/03_pipeline_inspection.py

Inspection is the dry-run view of the solver pipeline. This example deliberately
uses a mixed pool so the report has something interesting to show: one strict
single-cable batch, one scalar fallback, and one padded double-cable batch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

import axonscope as axs


def _mib(values: list[int]) -> list[float]:
    return [value / (1024**2) for value in values]


def _observation_slots(report: axs.SimulationInspection) -> list[int]:
    return [
        0 if detail.observation_shape is None else int(np.prod(detail.observation_shape[1:]))
        for detail in report.assembly_details
    ]


def _plot_report_comparison(
    retained_report: axs.SimulationInspection,
    compact_report: axs.SimulationInspection,
) -> None:
    labels = [str(group.group_id) for group in retained_report.dispatch_groups]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    fig.suptitle("retained Vm versus compact observer-only inspection")
    retained_report.plot(ax=axes[0, 0])
    axes[0, 0].set_title("retained Vm route")
    compact_report.plot(ax=axes[0, 1])
    axes[0, 1].set_title("compact observer-only route")

    memory_ax = axes[1, 0]
    memory_ax.bar(
        x - width / 2,
        _mib([item.total_estimated_bytes for item in retained_report.memory]),
        width=width,
        label="retained Vm",
    )
    memory_ax.bar(
        x + width / 2,
        _mib([item.total_estimated_bytes for item in compact_report.memory]),
        width=width,
        label="compact observer-only",
    )
    memory_ax.set_title("estimated working memory")
    memory_ax.set_xticks(x, labels)
    memory_ax.set_xlabel("dispatch group")
    memory_ax.set_ylabel("MiB")
    memory_ax.grid(True, axis="y", alpha=0.3)
    memory_ax.legend(frameon=False, fontsize=8)

    output_ax = axes[1, 1]
    output_ax.bar(
        x - width / 2,
        _mib([item.retained_public_bytes for item in retained_report.memory]),
        width=width,
        label="retained public",
    )
    output_ax.bar(
        x + width / 2,
        _mib([item.retained_public_bytes for item in compact_report.memory]),
        width=width,
        label="compact public",
    )
    output_ax.plot(
        x,
        _observation_slots(compact_report),
        marker="o",
        color="black",
        linewidth=1.6,
        label="compact observation slots",
    )
    output_ax.set_title("public output footprint")
    output_ax.set_xticks(x, labels)
    output_ax.set_xlabel("dispatch group")
    output_ax.set_ylabel("MiB / slots")
    output_ax.grid(True, axis="y", alpha=0.3)
    output_ax.legend(frameon=False, fontsize=8)

    details_fig, details_axes = plt.subplots(
        2,
        4,
        figsize=(15.0, 7.0),
        constrained_layout=True,
    )
    retained_report.plot_details(
        axes=(
            details_axes[0, 0],
            details_axes[0, 1],
            details_axes[1, 0],
            details_axes[1, 1],
        )
    )
    compact_report.plot_details(
        axes=(
            details_axes[0, 2],
            details_axes[0, 3],
            details_axes[1, 2],
            details_axes[1, 3],
        )
    )
    details_fig.suptitle("retained Vm details versus compact observer-only details")


def main() -> None:
    # Step 1: keep runtime policy explicit so inspection reports show the
    # requested backend/device/precision envelope and the double-cable `auto`
    # route can resolve against the effective device.
    policy = axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=axs.Device.cpu(),
        precision=axs.PrecisionPolicy.float32(),
    )
    duration = 1.0 * axs.ms
    dt = 0.01 * axs.ms

    # Step 2: build a mixed retained-Vm pool.
    #
    # Rows 0-1 share one Hodgkin-Huxley geometry and batch strictly. Row 2 is a
    # different single-cable shape and falls back to a scalar route. Rows 3-4 are
    # MRG double-cable axons with different node counts, so they batch with
    # spatial padding.
    clamped_rows: list[axs.AxonInstance] = []
    for amplitude_nA in (0.35, 0.55):
        axon = axs.axons.HodgkinHuxley(
            length=80.0 * axs.um,
            diameter=0.7 * axs.um,
            compartments=7,
            celsius=6.3 * axs.degC,
        )
        simulation = axs.AxonInstance(axon)
        simulation.add_current_clamp(
            position=40.0 * axs.um,
            current=axs.Stimulus.pulse(
                start=0.20 * axs.ms,
                duration=0.20 * axs.ms,
                amplitude=amplitude_nA * axs.nA,
            ),
        )
        clamped_rows.append(simulation)

    scalar_axon = axs.axons.HodgkinHuxley(
        length=120.0 * axs.um,
        diameter=0.7 * axs.um,
        compartments=11,
        celsius=6.3 * axs.degC,
    )
    scalar_row = axs.AxonInstance(scalar_axon)
    scalar_row.add_current_clamp(
        position=60.0 * axs.um,
        current=axs.Stimulus.pulse(
            start=0.20 * axs.ms,
            duration=0.20 * axs.ms,
            amplitude=0.75 * axs.nA,
        ),
    )
    clamped_rows.append(scalar_row)

    for diameter_um, nodes in ((3.0, 3), (4.0, 4)):
        axon = axs.axons.MRG(
            diameter=diameter_um * axs.um,
            nodes=nodes,
        )
        clamped_rows.append(axs.AxonInstance(axon))

    retained_simulation = axs.AxonSimulation(
        axs.AxonPopulation(clamped_rows),
        duration=duration,
        dt=dt,
        recording=axs.Recording.probes(axs.signals.Vm, count=3),
        batch_options=axs.BatchOptions.full(time_chunk_steps=25),
        execution_policy=policy,
    )
    retained_report = retained_simulation.inspect()

    # Step 3: build the same heterogeneous shape as an observer-only sampled
    # extracellular pool. Compatible batch groups can keep Vext factorized and
    # return compact VmRaster observations instead of retained Vm traces.
    stimulus = axs.Stimulus.pulse(
        start=0.20 * axs.ms,
        duration=0.20 * axs.ms,
        amplitude=20.0 * axs.uA,
    )
    electrode = axs.analytical.PointSourceElectrode(
        x=40.0 * axs.um,
        y=0.0 * axs.um,
        z=120.0 * axs.um,
    )
    extracellular_rows: list[axs.AxonInstance] = []
    for compartments, length in ((7, 80.0), (7, 80.0), (11, 120.0)):
        axon = axs.axons.HodgkinHuxley(
            length=length * axs.um,
            diameter=0.7 * axs.um,
            compartments=compartments,
            celsius=6.3 * axs.degC,
        )
        simulation = axs.AxonInstance(axon)
        positions = axon.layout.position_values(unit=axs.um) * axs.um
        simulation.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                positions,
                stimulus=stimulus,
                sigma=0.3 * axs.S_per_m,
            )
        )
        extracellular_rows.append(simulation)

    for diameter_um, nodes in ((3.0, 3), (4.0, 4)):
        axon = axs.axons.MRG(
            diameter=diameter_um * axs.um,
            nodes=nodes,
        )
        simulation = axs.AxonInstance(axon)
        positions = axon.layout.position_values(unit=axs.um) * axs.um
        simulation.add_extracellular_stimulation(
            stimulation=axs.analytical.point_source_stimulation(
                electrode,
                positions,
                stimulus=stimulus,
                sigma=0.3 * axs.S_per_m,
            )
        )
        extracellular_rows.append(simulation)

    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    compact_simulation = axs.AxonSimulation(
        axs.AxonPopulation(extracellular_rows),
        duration=duration,
        dt=dt,
        recording=axs.Recording.none(),
        observers=[activation],
        batch_options=axs.BatchOptions.full(time_chunk_steps=25),
        execution_policy=policy,
    )
    compact_report = compact_simulation.inspect()

    # Step 4: print the full retained report and a compact route comparison. The
    # same group IDs appear in both reports because the geometry mix is the same.
    print("\n=== Retained Vm Inspection ===")
    retained_report.print()

    display = {
        "callable_or_precomputed_per_axon": "callable/precomputed",
        "DispatchCohortResult": "cohort result",
        "factorized_footprint": "factorized footprint",
        "scalar fallback row": "scalar fallback",
    }
    comparison = Table(title="Retained Vm versus compact observer-only")
    for column in (
        "group",
        "kind",
        "rows",
        "retained Vm",
        "compact observer",
        "compact Vext",
        "compact result",
    ):
        comparison.add_column(column, overflow="fold")
    for retained_group, retained_lowering, compact_lowering, compact_assembly in zip(
        retained_report.dispatch_groups,
        retained_report.lowerings,
        compact_report.lowerings,
        compact_report.result_assembly,
        strict=True,
    ):
        comparison.add_row(
            str(retained_group.group_id),
            retained_group.batch_kind,
            str(retained_group.pool_indices),
            str(retained_lowering.retained_vm_width),
            compact_lowering.observer_format,
            display.get(
                compact_lowering.extracellular_format,
                compact_lowering.extracellular_format,
            ),
            display.get(compact_assembly.record_kind, compact_assembly.record_kind),
        )
    Console(width=120).print(comparison)

    # Step 5: compare both inspection records visually. `plot()` is a compact
    # route summary; `plot_details()` expands padding, memory, probes, and public
    # result assembly for each report.
    _plot_report_comparison(retained_report, compact_report)
    plt.show()


if __name__ == "__main__":
    main()
