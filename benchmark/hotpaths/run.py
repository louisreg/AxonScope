"""Run Phase 2.5 hotpath diagnostic workloads.

Examples:
    python benchmark/hotpaths/run.py --list
    python benchmark/hotpaths/run.py --workload all --preset smoke
    python benchmark/hotpaths/run.py --workload all --preset scale
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

import axonscope as axs
from benchmark.hotpaths.catalog import HOTPATH_PRESETS, HOTPATH_WORKLOADS


DEFAULT_OUT_DIR = Path("benchmark/results/hotpaths")


@dataclass(frozen=True)
class PlannedRun:
    """One workload/size combination."""

    workload: str
    size: int


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        default="all",
        choices=("all", *HOTPATH_WORKLOADS),
        help="Workload to run.",
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=tuple(HOTPATH_PRESETS),
        help="Named size preset.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit population sizes. Overrides --preset.",
    )
    parser.add_argument("--compartments", type=int, default=11)
    parser.add_argument("--length-um", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=0.30, help="Duration in ms.")
    parser.add_argument("--dt", type=float, default=0.05, help="Step size in ms.")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=3,
        help="Number of repeated simulations for the footprint_reuse_sweep workload.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=None, help="Optional run directory prefix.")
    parser.add_argument("--list", action="store_true", help="List workloads and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each AxonScope hotpath report.",
    )
    parser.add_argument(
        "--sync-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize JAX arrays at kernel.wait.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print_workloads()
        return
    if args.compartments < 3:
        raise ValueError("--compartments must be >= 3.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0.")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0.")
    if args.sweep_repeats < 1:
        raise ValueError("--sweep-repeats must be >= 1.")

    runs = planned_runs(args.workload, resolve_sizes(args.preset, args.sizes))
    if args.dry_run:
        for run in runs:
            print(f"{run.workload} size={run.size}")
        return

    run_root = make_run_root(args.out_dir, prefix=args.prefix)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "sizes": list(dict.fromkeys(run.size for run in runs)),
        "workloads": list(dict.fromkeys(run.workload for run in runs)),
        "parameters": {
            "compartments": int(args.compartments),
            "length_um": float(args.length_um),
            "duration_ms": float(args.duration),
            "dt_ms": float(args.dt),
            "warmups": int(args.warmups),
            "sweep_repeats": int(args.sweep_repeats),
            "sync_device": bool(args.sync_device),
        },
        "runs": [],
    }

    for run in runs:
        simulations = build_simulations(
            run.workload,
            size=run.size,
            compartments=args.compartments,
            length_um=args.length_um,
            duration_ms=args.duration,
            dt_ms=args.dt,
            sweep_repeats=args.sweep_repeats,
        )
        estimates = [simulation.estimate().to_dict() for simulation in simulations]
        simulation_labels = _simulation_labels(run.workload, len(simulations))
        for _ in range(args.warmups):
            for simulation in simulations:
                simulation.run()

        output_dir = run_root / f"{run.workload}_n{run.size}"
        axs.enable_benchmark(
            output_dir,
            print_summary=False,
            sync_device=bool(args.sync_device),
        )
        try:
            result_batches = tuple(simulation.run() for simulation in simulations)
        finally:
            report = axs.disable_benchmark(print_summary=bool(args.print_summary))

        run_record = {
            "workload": run.workload,
            "size": run.size,
            "simulation_count": len(simulations),
            "output_dir": str(output_dir),
            "simulation_labels": list(simulation_labels),
            "result_count": sum(_result_count(results) for results in result_batches),
            "vm_shapes": _sample_vm_shapes(result_batches),
            "observation_names": _sample_observation_names(result_batches),
            "memory_estimate": estimates[0],
            "memory_estimates": estimates,
            "workload_metadata": _describe_simulations(simulation_labels, simulations),
            "event_count": 0 if report is None else len(report.events),
            "summary": [] if report is None else [row.to_dict() for row in report.summary],
        }
        manifest["runs"].append(run_record)
        print(
            f"{run.workload} size={run.size}: "
            f"{run_record['event_count']} events -> {output_dir}"
        )

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest: {manifest_path}")


def print_workloads() -> None:
    print("Hotpath workloads:")
    for name, workload in HOTPATH_WORKLOADS.items():
        print(f"  {name:28s} {workload.description}")
    print("Presets:")
    for name, sizes in HOTPATH_PRESETS.items():
        joined = ", ".join(str(size) for size in sizes)
        print(f"  {name:28s} sizes={joined}")


def resolve_sizes(preset: str, sizes: Sequence[int] | None) -> tuple[int, ...]:
    if sizes is not None:
        resolved = tuple(int(size) for size in sizes)
    else:
        resolved = HOTPATH_PRESETS[preset]
    if not resolved:
        raise ValueError("at least one size is required.")
    if any(size < 1 for size in resolved):
        raise ValueError("all sizes must be >= 1.")
    return resolved


def planned_runs(workload: str, sizes: Sequence[int]) -> tuple[PlannedRun, ...]:
    workload_names = tuple(HOTPATH_WORKLOADS) if workload == "all" else (workload,)
    return tuple(
        PlannedRun(workload=name, size=int(size))
        for name in workload_names
        for size in sizes
    )


def make_run_root(out_dir: Path, *, prefix: str | None) -> Path:
    stem = prefix or f"hotpaths_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = out_dir / stem
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_simulation(
    workload: str,
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> axs.AxonSimulation:
    return build_simulations(
        workload,
        size=size,
        compartments=compartments,
        length_um=length_um,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        sweep_repeats=1,
    )[0]


def build_simulations(
    workload: str,
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
    sweep_repeats: int = 1,
) -> tuple[axs.AxonSimulation, ...]:
    if workload == "intracellular_only":
        instances = build_intracellular_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "point_source_extracellular":
        instances = build_point_source_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "footprint_reuse_sweep":
        return tuple(
            axs.AxonSimulation(
                axs.AxonPopulation(
                    build_point_source_pool(
                        size=size,
                        compartments=compartments,
                        length_um=length_um,
                        amplitude_uA=20.0 + 2.5 * repeat,
                    )
                ),
                duration=duration_ms * axs.ms,
                dt=dt_ms * axs.ms,
                recording=axs.Recording.center(axs.signals.Vm),
            )
            for repeat in range(int(sweep_repeats))
        )
    elif workload == "observer_only":
        instances = build_intracellular_pool(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
        peak_voltage = axs.analysis.PeakVoltage(target=axs.positions.CENTER)
        activation = axs.analysis.Activation(
            threshold=-80.0 * axs.mV,
            target=axs.positions.CENTER,
        )
        return (
            axs.AxonSimulation(
                axs.AxonPopulation(instances),
                duration=duration_ms * axs.ms,
                dt=dt_ms * axs.ms,
                recording=axs.Recording.none(),
                observers=[peak_voltage, activation],
            ),
        )
    elif workload == "realistic_mixed_population":
        instances = build_realistic_mixed_population(
            size=size,
            compartments=compartments,
            length_um=length_um,
        )
    elif workload == "hotpath_matrix":
        return build_hotpath_matrix(
            size=size,
            compartments=compartments,
            length_um=length_um,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
        )
    else:
        raise ValueError(f"Unknown hotpath workload: {workload!r}.")

    return (
        axs.AxonSimulation(
            axs.AxonPopulation(instances),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
    )


def build_realistic_mixed_population(
    *,
    size: int,
    compartments: int,
    length_um: float,
) -> list[axs.AxonInstance]:
    """Build a deliberately mixed population for Phase 7.6 hotpath probes."""

    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.10 * axs.ms,
        amplitude=18.0 * axs.uA,
    )
    electrode = axs.PointSourceElectrode(
        x=0.5 * length_um * axs.um,
        z=140.0 * axs.um,
        stimulus=stimulus,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[electrode],
        sigma=0.3 * axs.S_per_m,
    )

    diameter_cycle_um = (0.6, 0.8, 1.0, 1.2)
    offsets = np.linspace(-60.0, 60.0, size) if size > 1 else np.asarray([0.0])
    instances: list[axs.AxonInstance] = []
    for index, offset_um in enumerate(offsets):
        diameter_um = diameter_cycle_um[index % len(diameter_cycle_um)]
        row_compartments = max(3, int(compartments) + 2 * (index % 3))
        if index % 2 == 0:
            axon = axs.axons.HodgkinHuxley(
                length=length_um * axs.um,
                diameter=diameter_um * axs.um,
                compartments=row_compartments,
                celsius=6.3 * axs.degC,
            )
        else:
            axon = axs.axons.RattayAberham(
                length=length_um * axs.um,
                diameter=diameter_um * axs.um,
                compartments=row_compartments,
                celsius=37.0 * axs.degC,
            )

        instance = axs.AxonInstance(axon, y=float(offset_um) * axs.um)
        instance.add_current_clamp(
            position=0.5 * length_um * axs.um,
            current=axs.Stimulus.pulse(
                start=0.08 * axs.ms,
                duration=0.12 * axs.ms,
                amplitude=(0.45 + 0.02 * (index % 5)) * axs.nA,
            ),
        )
        if index % 3 == 0:
            instance.add_extracellular_context(context=context)
        instances.append(instance)
    return instances


def build_hotpath_matrix(
    *,
    size: int,
    compartments: int,
    length_um: float,
    duration_ms: float,
    dt_ms: float,
) -> tuple[axs.AxonSimulation, ...]:
    """Return a compact matrix of representative Phase 7.6 hotpath scenarios."""

    peak_voltage = axs.analysis.PeakVoltage(target=axs.positions.CENTER)
    activation = axs.analysis.Activation(
        threshold=-80.0 * axs.mV,
        target=axs.positions.CENTER,
    )
    return (
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.probes(axs.signals.Vm, count=5),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_intracellular_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.none(),
            observers=[peak_voltage, activation],
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_point_source_pool(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
        axs.AxonSimulation(
            axs.AxonPopulation(
                build_realistic_mixed_population(
                    size=size,
                    compartments=compartments,
                    length_um=length_um,
                )
            ),
            duration=duration_ms * axs.ms,
            dt=dt_ms * axs.ms,
            recording=axs.Recording.center(axs.signals.Vm),
        ),
    )


def build_intracellular_pool(
    *,
    size: int,
    compartments: int,
    length_um: float,
) -> list[axs.AxonInstance]:
    axon = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )
    instances = []
    for index in range(size):
        instance = axs.AxonInstance(axon)
        instance.add_current_clamp(
            position=0.5 * length_um * axs.um,
            current=axs.Stimulus.pulse(
                start=0.10 * axs.ms,
                duration=0.10 * axs.ms,
                amplitude=(0.5 + 0.01 * index) * axs.nA,
            ),
        )
        instances.append(instance)
    return instances


def build_point_source_pool(
    *,
    size: int,
    compartments: int,
    length_um: float,
    amplitude_uA: float = 25.0,
) -> list[axs.AxonInstance]:
    axon = axs.axons.HodgkinHuxley(
        length=length_um * axs.um,
        diameter=0.9 * axs.um,
        compartments=compartments,
        celsius=6.3 * axs.degC,
    )
    stimulus = axs.Stimulus.pulse(
        start=0.10 * axs.ms,
        duration=0.10 * axs.ms,
        amplitude=float(amplitude_uA) * axs.uA,
    )
    electrode = axs.PointSourceElectrode(
        x=0.5 * length_um * axs.um,
        z=120.0 * axs.um,
        stimulus=stimulus,
    )
    context = axs.AnalyticalExtracellularContext(
        electrodes=[electrode],
        sigma=0.3 * axs.S_per_m,
    )

    offsets = np.linspace(-40.0, 40.0, size) if size > 1 else np.asarray([0.0])
    instances = []
    for offset_um in offsets:
        instance = axs.AxonInstance(axon, y=float(offset_um) * axs.um)
        instance.add_extracellular_context(context=context)
        instances.append(instance)
    return instances


def _simulation_labels(workload: str, count: int) -> tuple[str, ...]:
    if workload == "hotpath_matrix":
        labels = (
            "homogeneous_center",
            "homogeneous_probes",
            "observer_only_none",
            "point_source_center",
            "realistic_mixed_center",
        )
        return labels[:count]
    if workload == "footprint_reuse_sweep":
        return tuple(f"repeat_{index}" for index in range(count))
    return tuple(workload for _ in range(count))


def _describe_simulations(
    labels: Sequence[str],
    simulations: Sequence[axs.AxonSimulation],
) -> list[dict[str, object]]:
    return [
        _describe_simulation(label, simulation)
        for label, simulation in zip(labels, simulations, strict=True)
    ]


def _describe_simulation(
    label: str,
    simulation: axs.AxonSimulation,
) -> dict[str, object]:
    instances = tuple(simulation.axons)
    diameters_um = [float(instance.axon.diameter) for instance in instances]
    compartments = [int(instance.axon.n_compartments) for instance in instances]
    return {
        "label": label,
        "axon_count": len(instances),
        "model_counts": _count_values(type(instance.axon).__name__ for instance in instances),
        "formulation_counts": _count_values(
            instance.axon.resolved_formulation for instance in instances
        ),
        "diameter_um": _numeric_distribution(diameters_um),
        "compartments": _numeric_distribution(compartments),
        "intracellular_rows": sum(bool(instance.intracellular_contexts) for instance in instances),
        "extracellular_rows": sum(instance.extracellular_context is not None for instance in instances),
        "recording_policy": _recording_policy(simulation.recording),
        "observer_names": [
            str(getattr(observer, "name", type(observer).__name__))
            for observer in (simulation.observers or ())
        ],
    }


def _count_values(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _numeric_distribution(values: Sequence[float | int]) -> dict[str, object]:
    if not values:
        return {"unique": [], "min": None, "max": None}
    unique = sorted({float(value) for value in values})
    formatted_unique: list[float | int] = [
        int(value) if value.is_integer() else value for value in unique[:12]
    ]
    distribution: dict[str, object] = {
        "unique": formatted_unique,
        "min": min(values),
        "max": max(values),
    }
    if len(unique) > 12:
        distribution["unique_truncated"] = True
        distribution["unique_count"] = len(unique)
    return distribution


def _recording_policy(recording: axs.Recording | None) -> dict[str, object]:
    if recording is None:
        return {"spatial": "default", "signals": ["membrane_voltage"]}
    if not recording.voltage:
        return {"spatial": "none", "signals": []}
    return {
        "spatial": recording.spatial.value,
        "signals": [str(signal.id) for signal in recording.signals],
        "probe_count": recording.probe_count,
        "indices": None if recording.record_indices is None else list(recording.record_indices),
    }


def _result_count(results: object) -> int:
    if isinstance(results, axs.AxonSimulationResult):
        return len(results)
    return 1


def _sample_vm_shapes(result_batches: Sequence[object]) -> list[list[int]]:
    shapes: list[list[int]] = []
    for results in result_batches:
        if isinstance(results, axs.AxonSimulationResult):
            for result in results[: max(0, 3 - len(shapes))]:
                try:
                    shapes.append(list(np.asarray(result.Vm).shape))
                except (AttributeError, ValueError):
                    continue
        else:
            try:
                shapes.append(list(np.asarray(results.Vm).shape))  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                continue
        if len(shapes) >= 3:
            break
    return shapes


def _sample_observation_names(result_batches: Sequence[object]) -> list[str]:
    for results in result_batches:
        observations = getattr(results, "observations", None)
        if observations:
            return sorted(str(name) for name in observations)
    return []


if __name__ == "__main__":
    main()
