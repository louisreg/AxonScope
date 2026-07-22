"""Profile generated HH, Markov, and mixed membranes in full cable runs.

This benchmark deliberately uses the public ``AxonSimulation.run()`` path.
It complements ``membrane_kinetics.py`` by retaining preparation, cable solve,
result assembly, and the production single-/double-cable dispatch routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import axonscope as axs
from benchmark.analysis.cache_replay import (
    cache_tree_delta,
    cache_tree_snapshot,
    ratio,
)


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
    parser.add_argument("--models", default="passive,hh,nav16,mixed")
    parser.add_argument("--cables", default="single,double")
    parser.add_argument("--double-layouts", default="uniform,node_localized")
    parser.add_argument("--axons")
    parser.add_argument("--single-compartments", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--dt-ms", type=float, default=0.005)
    parser.add_argument("--v-init-mv", type=float, default=-70.0)
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
    parser.add_argument(
        "--capture-jit-phases",
        action="store_true",
        help="Capture trace/lower/compile/execute phases and HLO for the stateful JIT.",
    )
    parser.add_argument(
        "--compilation-cache-replay",
        action="store_true",
        help="Run miss, exact replay, and dynamic-value replay in fresh processes.",
    )
    parser.add_argument("--cold-only", action="store_true", help=argparse.SUPPRESS)
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
    if args.compilation_cache_replay:
        return _run_compilation_cache_replay(
            args,
            cases=cases,
            single_compartments=single_compartments,
            nodes=nodes,
            duration_ms=duration_ms,
        )
    if args.capture_jit_phases:
        if len(cases) != 1:
            raise SystemExit("--capture-jit-phases requires exactly one selected case.")
        from benchmark.analysis.jax_phase_capture import install_production_jax_captures

        install_production_jax_captures(
            args.output / "jax_phase_capture",
            cables=(cases[0].cable,),
            platform=args.platform,
            route="recording",
        )
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
                v_init_mV=args.v_init_mv,
                repeats=repeats,
                memory_trace=args.memory_trace,
                cold_only=args.cold_only,
            )
        )

    payload = {
        "preset": args.preset,
        "platform": args.platform,
        "duration_ms": duration_ms,
        "dt_ms": args.dt_ms,
        "v_init_mV": args.v_init_mv,
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
    v_init_mV: float,
    repeats: int,
    memory_trace: str,
    cold_only: bool,
) -> dict[str, object]:
    build_start = perf_counter()
    membranes = _membranes(case.membrane)
    axon_templates = tuple(
        _axon(
            membrane,
            cable=case.cable,
            layout=case.layout,
            single_compartments=single_compartments,
            nodes=nodes,
            v_init_mV=v_init_mV,
        )
        for membrane in membranes
    )
    axon = axon_templates[0]
    population = axs.AxonPopulation(
        (
            axon_templates[index % len(axon_templates)]
            for index in range(case.n_axons)
        ),
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
    last_result = None
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
        run_count = 1 if cold_only else repeats + 1
        for run_index in range(run_count):
            phase = "cold" if run_index == 0 else "warm"
            start = perf_counter()
            with session.span(
                "membrane_temporal.run",
                phase=phase,
                run_index=run_index,
            ):
                last_result = simulation.run()
            run_ms.append(1e3 * (perf_counter() - start))

    state_count = len(membranes[0].explain().recording_outputs.gates)
    active_compartments = _active_compartments(axon, layout=case.layout)
    dense_state_bytes = case.n_axons * axon.n_compartments * state_count * 4
    compact_state_bytes = case.n_axons * active_compartments * state_count * 4
    assert last_result is not None
    dispatch_group_ids = {
        int(result.diagnostics["dispatch_group_id"]) for result in last_result
    }
    dispatch_methods = sorted(
        {str(result.diagnostics["dispatch_method"]) for result in last_result}
    )
    result_checksum = _result_checksum(last_result)
    return {
        "case": case.name,
        "membrane": case.membrane,
        "cable": case.cable,
        "layout": case.layout,
        "n_axons": case.n_axons,
        "membrane_parameter_sets": len(membranes),
        "dispatch_group_count": len(dispatch_group_ids),
        "dispatch_methods": dispatch_methods,
        "compartments_per_axon": axon.n_compartments,
        "active_compartments_per_axon": active_compartments,
        "evolving_states_per_active_compartment": state_count,
        "dense_evolving_state_bytes": dense_state_bytes,
        "active_site_compact_state_bytes": compact_state_bytes,
        "inactive_state_bytes": dense_state_bytes - compact_state_bytes,
        "build_ms": build_ms,
        "cold_run_ms": run_ms[0],
        "warm_run_ms": run_ms[1:],
        "warm_median_ms": median(run_ms[1:]) if len(run_ms) > 1 else None,
        "result_checksum": result_checksum,
        "run_profiles": _run_profiles(session.events),
    }


def _result_checksum(result) -> str:
    digest = hashlib.sha256()
    for row in result:
        values = np.ascontiguousarray(row.voltage_values(unit=axs.mV))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _run_compilation_cache_replay(
    args: argparse.Namespace,
    *,
    cases: list[Case],
    single_compartments: int,
    nodes: int,
    duration_ms: float,
) -> int:
    if not cases:
        raise SystemExit("--compilation-cache-replay selected no cases.")

    records = []
    for case in cases:
        case_output = args.output / case.name
        cache_root = case_output / "jax_xla_cache"
        if cache_root.exists() and any(cache_root.iterdir()):
            raise SystemExit(
                "--compilation-cache-replay requires fresh per-case cache roots; "
                f"found data in {cache_root}."
            )
        processes = []
        for label, v_init_mV in (
            ("cache_miss", args.v_init_mv),
            ("cache_replay", args.v_init_mv),
            ("dynamic_v_init_replay", args.v_init_mv + 0.5),
        ):
            child_output = case_output / label
            before = cache_tree_snapshot(cache_root)
            environment = os.environ.copy()
            environment["AXONSCOPE_JAX_COMPILATION_CACHE"] = str(cache_root)
            environment["AXONSCOPE_JAX_CACHE_MIN_COMPILE_TIME_S"] = "0"
            environment["AXONSCOPE_JAX_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
            environment["AXONSCOPE_JAX_PERSISTENT_XLA_CACHES"] = "all"
            environment["JAX_EXPLAIN_CACHE_MISSES"] = "true"
            environment["MPLCONFIGDIR"] = str(child_output / ".matplotlib")
            command = _cache_replay_child_command(
                args,
                case=case,
                output=child_output,
                single_compartments=single_compartments,
                nodes=nodes,
                duration_ms=duration_ms,
                v_init_mV=v_init_mV,
            )
            child_output.mkdir(parents=True, exist_ok=True)
            with (child_output / "process.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"cache replay child {case.name}/{label} failed with "
                    f"exit code {completed.returncode}."
                )
            record = _read_cache_replay_child(
                child_output,
                case=case,
                label=label,
            )
            record["jax_xla_cache"] = cache_tree_delta(
                before,
                cache_tree_snapshot(cache_root),
            )
            processes.append(record)

        miss, replay, dynamic = processes
        exact_result_match = miss["result_checksum"] == replay["result_checksum"]
        stablehlo_match = miss["stablehlo_sha256"] == replay["stablehlo_sha256"]
        dynamic_stablehlo_match = (
            miss["stablehlo_sha256"] == dynamic["stablehlo_sha256"]
        )
        cache_reused = replay["jax_xla_cache"]["new_file_count"] == 0
        accepted = bool(
            exact_result_match
            and stablehlo_match
            and dynamic_stablehlo_match
            and cache_reused
        )
        case_record = {
            "case": case.name,
            "accepted": accepted,
            "exact_result_match": exact_result_match,
            "stablehlo_match": stablehlo_match,
            "dynamic_stablehlo_match": dynamic_stablehlo_match,
            "fresh_process_cache_reused": cache_reused,
            "compile_speedup": ratio(miss["compile_s"], replay["compile_s"]),
            "cold_phase_speedup": ratio(
                miss["total_cold_s"], replay["total_cold_s"]
            ),
            "processes": processes,
        }
        _write_case_hlo_analysis(case_output, case, case_record, miss)
        (case_output / "cache_replay.json").write_text(
            json.dumps(case_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(case_record)

    payload = {
        "platform": args.platform,
        "accepted": all(record["accepted"] for record in records),
        "cases": records,
    }
    (args.output / "compilation_cache_replay.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_cache_replay_report(args.output, payload)
    if not payload["accepted"]:
        raise RuntimeError("Membrane temporal cache replay acceptance gate failed.")
    print(f"Membrane temporal cache replay results: {args.output}")
    return 0


def _cache_replay_child_command(
    args: argparse.Namespace,
    *,
    case: Case,
    output: Path,
    single_compartments: int,
    nodes: int,
    duration_ms: float,
    v_init_mV: float,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        str(args.preset),
        "--platform",
        str(args.platform),
        "--models",
        case.membrane,
        "--cables",
        case.cable,
        "--double-layouts",
        case.layout,
        "--axons",
        str(case.n_axons),
        "--single-compartments",
        str(single_compartments),
        "--nodes",
        str(nodes),
        "--duration-ms",
        str(duration_ms),
        "--dt-ms",
        str(args.dt_ms),
        "--v-init-mv",
        str(v_init_mV),
        "--repeats",
        "1",
        "--memory-trace",
        "off",
        "--output",
        str(output),
        "--capture-jit-phases",
        "--cold-only",
    ]


def _read_cache_replay_child(
    output: Path,
    *,
    case: Case,
    label: str,
) -> dict[str, object]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if len(summary["rows"]) != 1:
        raise RuntimeError(f"{case.name}/{label} did not produce exactly one row.")
    row = summary["rows"][0]
    phase = json.loads(
        (
            output
            / "jax_phase_capture"
            / f"{case.cable}.jit_phases.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "label": label,
        "v_init_mV": float(summary["v_init_mV"]),
        "result_checksum": str(row["result_checksum"]),
        "cold_run_ms": float(row["cold_run_ms"]),
        "trace_s": float(phase["trace_s"]),
        "lower_s": float(phase["lower_s"]),
        "compile_s": float(phase["compile_s"]),
        "first_execution_s": float(phase["first_execution_s"]),
        "total_cold_s": float(phase["total_cold_s"]),
        "stablehlo_sha256": str(phase["stablehlo"]["sha256"]),
        "stablehlo_bytes": int(phase["stablehlo"]["bytes"]),
        "optimized_hlo_sha256": str(phase["optimized_hlo"]["sha256"]),
        "optimized_hlo_bytes": int(phase["optimized_hlo"]["bytes"]),
        "dynamic_shapes": phase["dynamic"],
        "static_signature": phase["static"],
    }


def _write_case_hlo_analysis(
    case_output: Path,
    case: Case,
    case_record: dict[str, object],
    miss: dict[str, object],
) -> None:
    from benchmark.analysis.hlo_fusion_summary import write_hlo_fusion_artifacts

    hlo_file = (
        case_output
        / "cache_miss"
        / "jax_phase_capture"
        / f"{case.cable}.compiled.optimized_hlo.txt"
    )
    write_hlo_fusion_artifacts(
        case_output / "hlo",
        files=(hlo_file,),
        metadata={"case": case_record["case"], "cache_miss": miss},
    )


def _write_cache_replay_report(output: Path, payload: dict[str, object]) -> None:
    lines = [
        "# P18 Membrane Temporal Compilation Cache Replay",
        "",
        "| case | accepted | exact result | StableHLO | dynamic StableHLO | cache reused | compile speedup | cold speedup |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in payload["cases"]:
        lines.append(
            "| {case} | {accepted} | {exact_result_match} | {stablehlo_match} | "
            "{dynamic_stablehlo_match} | {fresh_process_cache_reused} | "
            "{compile_speedup:.2f}x | {cold_phase_speedup:.2f}x |".format(**record)
        )
    lines.extend(["", f"Overall accepted: {payload['accepted']}"])
    (output / "compilation_cache_replay.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _membranes(name: str) -> tuple[object, ...]:
    if name == "passive":
        return (axs.membranes.Passive(),)
    if name == "hh":
        return (axs.membranes.HodgkinHuxley(),)
    if name == "nav16":
        return (
            axs.membranes.Composite(
                {"sodium": axs.membranes.Nav16(), "leak": axs.membranes.Passive()}
            ),
        )
    if name == "nav_isoforms":
        return tuple(
            axs.membranes.Composite(
                {"sodium": model(), "leak": axs.membranes.Passive()}
            )
            for model in (
                axs.membranes.Nav11,
                axs.membranes.Nav12,
                axs.membranes.Nav13,
                axs.membranes.Nav14,
                axs.membranes.Nav15,
                axs.membranes.Nav16,
                axs.membranes.Nav17,
                axs.membranes.Nav18,
                axs.membranes.Nav19,
            )
        )
    if name == "mixed":
        return (
            axs.membranes.Composite(
                {
                    "hh": axs.membranes.HodgkinHuxley(),
                    "nav16": axs.membranes.Nav16(),
                }
            ),
        )
    raise ValueError(
        f"unknown membrane {name!r}; expected passive, hh, nav16, "
        "nav_isoforms, or mixed"
    )


def _axon(
    membrane,
    *,
    cable: str,
    layout: str,
    single_compartments: int,
    nodes: int,
    v_init_mV: float = -70.0,
):
    if cable == "single":
        return axs.axons.Unmyelinated(
            membrane=membrane,
            length=1000.0 * axs.um,
            diameter=1.0 * axs.um,
            compartments=single_compartments,
            v_init=v_init_mV * axs.mV,
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
        v_init=v_init_mV * axs.mV,
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
