"""Profile generated HH, Markov, and mixed membranes in full cable runs.

This benchmark deliberately uses the public ``AxonSimulation.run()`` path.
It complements ``membrane_kinetics.py`` by retaining preparation, cable solve,
result assembly, and the production single-/double-cable dispatch routes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import axonscope as axs


_QUICK = {
    "axons": "1,16",
    "single_compartments": 41,
    "nodes": 5,
    "duration_ms": 0.1,
    "repeats": 2,
}
_STANDARD = {
    "axons": "1,128,1024",
    "single_compartments": 101,
    "nodes": 11,
    "duration_ms": 0.5,
    "repeats": 3,
}
_LARGE = {
    "axons": "1024,4096",
    "single_compartments": 201,
    "nodes": 21,
    "duration_ms": 0.5,
    "repeats": 3,
}
_PRESETS = {
    "quick": _QUICK,
    "local_smoke": _QUICK,
    "local_realistic": _STANDARD,
    "cpu_publication": _STANDARD,
    "gpu_smoke": _STANDARD,
    "gpu_trace_smoke": _STANDARD,
    "gpu_realistic": _LARGE,
}


@dataclass(frozen=True)
class Case:
    membrane: str
    cable: str
    layout: str
    n_axons: int

    @property
    def name(self) -> str:
        return f"{self.membrane}_{self.cable}_{self.layout}_n{self.n_axons}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(_PRESETS), default="quick")
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--models", default="hh,nav16,mixed")
    parser.add_argument("--cables", default="single,double")
    parser.add_argument("--double-layouts", default="uniform,node_localized")
    parser.add_argument("--axons")
    parser.add_argument("--single-compartments", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--dt-ms", type=float, default=0.005)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--case-filter")
    parser.add_argument(
        "--memory-trace",
        choices=("off", "rss", "tracemalloc", "device", "all"),
        default="off",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/p18_membrane_temporal_local"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    preset = _PRESETS[args.preset]
    axon_counts = _csv_ints(args.axons or str(preset["axons"]))
    single_compartments = args.single_compartments or int(preset["single_compartments"])
    nodes = args.nodes or int(preset["nodes"])
    duration_ms = args.duration_ms or float(preset["duration_ms"])
    repeats = args.repeats or int(preset["repeats"])
    cases = _cases(
        models=_csv(args.models),
        cables=_csv(args.cables),
        double_layouts=_csv(args.double_layouts),
        axon_counts=axon_counts,
        case_filter=args.case_filter,
    )
    if args.dry_run:
        for case in cases:
            print(case.name)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        print(f"Running {case.name}...")
        rows.append(
            _measure_case(
                case,
                output=args.output / case.name,
                platform=args.platform,
                single_compartments=single_compartments,
                nodes=nodes,
                duration_ms=duration_ms,
                dt_ms=args.dt_ms,
                repeats=repeats,
                memory_trace=args.memory_trace,
            )
        )

    payload = {
        "preset": args.preset,
        "platform": args.platform,
        "duration_ms": duration_ms,
        "dt_ms": args.dt_ms,
        "repeats": repeats,
        "rows": rows,
    }
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"AxonScope membrane temporal results: {args.output}")
    return 0


def _measure_case(
    case: Case,
    *,
    output: Path,
    platform: str,
    single_compartments: int,
    nodes: int,
    duration_ms: float,
    dt_ms: float,
    repeats: int,
    memory_trace: str,
) -> dict[str, object]:
    build_start = perf_counter()
    membrane = _membrane(case.membrane)
    axon = _axon(
        membrane,
        cable=case.cable,
        layout=case.layout,
        single_compartments=single_compartments,
        nodes=nodes,
    )
    population = axs.AxonPopulation(
        (axon for _ in range(case.n_axons)),
        name=case.name,
    )
    simulation = axs.AxonSimulation(
        population,
        duration=duration_ms * axs.ms,
        dt=dt_ms * axs.ms,
        recording=axs.Recording.center(axs.signals.Vm),
        execution_policy=axs.ExecutionPolicy(
            runtime=axs.runtime.jax,
            device=axs.Device.gpu(0) if platform == "gpu" else axs.Device.cpu(),
            precision=axs.PrecisionPolicy.float32(),
        ),
        progress=False,
    )
    build_ms = 1e3 * (perf_counter() - build_start)

    run_ms: list[float] = []
    with axs.benchmark(
        output,
        print_summary=False,
        save=True,
        sync_device=True,
        record_shapes=True,
        memory_trace=memory_trace,
    ) as session:
        session.record_metadata(
            benchmark="membrane_temporal",
            case=case.name,
            membrane=case.membrane,
            cable=case.cable,
            layout=case.layout,
            n_axons=case.n_axons,
        )
        for run_index in range(repeats + 1):
            phase = "cold" if run_index == 0 else "warm"
            start = perf_counter()
            with session.span(
                "membrane_temporal.run",
                phase=phase,
                run_index=run_index,
            ):
                simulation.run()
            run_ms.append(1e3 * (perf_counter() - start))

    state_count = len(membrane.explain().recording_outputs.gates)
    active_compartments = _active_compartments(axon, layout=case.layout)
    dense_state_bytes = case.n_axons * axon.n_compartments * state_count * 4
    compact_state_bytes = case.n_axons * active_compartments * state_count * 4
    return {
        "case": case.name,
        "membrane": case.membrane,
        "cable": case.cable,
        "layout": case.layout,
        "n_axons": case.n_axons,
        "compartments_per_axon": axon.n_compartments,
        "active_compartments_per_axon": active_compartments,
        "evolving_states_per_active_compartment": state_count,
        "dense_evolving_state_bytes": dense_state_bytes,
        "active_site_compact_state_bytes": compact_state_bytes,
        "inactive_state_bytes": dense_state_bytes - compact_state_bytes,
        "build_ms": build_ms,
        "cold_run_ms": run_ms[0],
        "warm_run_ms": run_ms[1:],
        "warm_median_ms": median(run_ms[1:]),
        "run_profiles": _run_profiles(session.events),
    }


def _membrane(name: str):
    if name == "hh":
        return axs.membranes.HodgkinHuxley()
    if name == "nav16":
        return axs.membranes.Composite(
            {"sodium": axs.membranes.Nav16(), "leak": axs.membranes.Passive()}
        )
    if name == "mixed":
        return axs.membranes.Composite(
            {"hh": axs.membranes.HodgkinHuxley(), "nav16": axs.membranes.Nav16()}
        )
    raise ValueError(f"unknown membrane {name!r}; expected hh, nav16, or mixed")


def _axon(
    membrane,
    *,
    cable: str,
    layout: str,
    single_compartments: int,
    nodes: int,
):
    if cable == "single":
        return axs.axons.Unmyelinated(
            membrane=membrane,
            length=1000.0 * axs.um,
            diameter=1.0 * axs.um,
            compartments=single_compartments,
            v_init=-70.0 * axs.mV,
        )
    if cable != "double":
        raise ValueError(f"unknown cable {cable!r}")
    passive = axs.membranes.Passive()
    section_membranes = axs.membranes.SectionLayout(
        node=membrane,
        mysa=membrane if layout == "uniform" else passive,
        flut=membrane if layout == "uniform" else passive,
        stin=membrane if layout == "uniform" else passive,
    )
    return axs.axons.MRG(
        diameter=10.0 * axs.um,
        nodes=nodes,
        membranes=section_membranes,
        v_init=-70.0 * axs.mV,
    )


def _active_compartments(axon, *, layout: str) -> int:
    if layout == "uniform":
        return axon.n_compartments
    return sum(
        element.compartments
        for element in axon.layout.elements
        if element.section.name.lower() == "node"
    )


def _run_profiles(events) -> list[dict[str, object]]:
    selected = (
        "simulation.run_pool",
        "runtime.prepare",
        "kernel.enqueue",
        "kernel.dispatch_jax",
        "kernel.wait",
        "results.to_public",
    )
    by_id = {event.event_id: event for event in events}
    roots = {
        event.event_id: event
        for event in events
        if event.name == "membrane_temporal.run"
    }
    totals = {root_id: {name: 0.0 for name in selected} for root_id in roots}
    for event in events:
        parent_id = event.parent_event_id
        while parent_id is not None and parent_id not in roots:
            parent_id = by_id[parent_id].parent_event_id
        if parent_id in roots and event.name in selected:
            totals[parent_id][event.name] += event.duration_ns / 1e6
    profiles = []
    for root_id, root in roots.items():
        profile = {
            "phase": root.metadata["phase"],
            "run_index": root.metadata["run_index"],
            "total_ms": root.duration_ns / 1e6,
        }
        profile.update({f"{name}_ms": value for name, value in totals[root_id].items()})
        profiles.append(profile)
    return profiles


def _cases(*, models, cables, double_layouts, axon_counts, case_filter):
    cases = []
    for model in models:
        for cable in cables:
            layouts = ("uniform",) if cable == "single" else double_layouts
            for layout in layouts:
                for n_axons in axon_counts:
                    case = Case(model, cable, layout, n_axons)
                    if case_filter is None or case_filter in case.name:
                        cases.append(case)
    return cases


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv(value))


if __name__ == "__main__":
    raise SystemExit(main())
