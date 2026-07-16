"""Profile startup and execution phases of the basic-08 workload.

This benchmark intentionally mirrors ``examples/basic/08`` without importing
or modifying that public example. Its default scope stops immediately before
``recruitment_sweep`` so large population-construction costs can be measured
without paying for the solver.
"""

from __future__ import annotations

import time

_SCRIPT_START_NS = time.perf_counter_ns()

import argparse
import contextlib
import csv
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "results" / "basic_08_startup"
DEFAULT_MPLCONFIGDIR = REPO_ROOT / "benchmark" / "results" / ".matplotlib-cache"
DEFAULT_AMPLITUDES_UA = (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0)


@dataclass
class _Aggregate:
    count: int = 0
    elapsed_ns: int = 0


class _PhaseRecorder:
    def __init__(self) -> None:
        self._rows: dict[str, _Aggregate] = {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            aggregate = self._rows.setdefault(name, _Aggregate())
            aggregate.count += 1
            aggregate.elapsed_ns += elapsed

    def call(self, name: str, function: Callable[..., Any], *args, **kwargs):
        with self.phase(name):
            return function(*args, **kwargs)

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "phase": name,
                "count": aggregate.count,
                "total_ms": aggregate.elapsed_ns / 1_000_000.0,
                "mean_ms": (
                    aggregate.elapsed_ns / aggregate.count / 1_000_000.0
                    if aggregate.count
                    else 0.0
                ),
            }
            for name, aggregate in self._rows.items()
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fibers-per-family", type=int, default=1000)
    parser.add_argument(
        "--scope",
        choices=("startup", "first-amplitude", "full"),
        default="startup",
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--duration-ms", type=float, default=4.0)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--template-policy",
        choices=("distinct", "shared"),
        default="distinct",
        help="Construct every row or share axons with the same canonical diameter.",
    )
    parser.add_argument(
        "--waveform-update-policy",
        choices=("callback", "typed"),
        default="typed",
        help="Rebuild every row or reuse one typed runtime waveform handle.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fibers_per_family < 1:
        raise SystemExit("--fibers-per-family must be >= 1.")
    if args.duration_ms <= 0.0 or args.dt_ms <= 0.0:
        raise SystemExit("--duration-ms and --dt-ms must be positive.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

    phases = _PhaseRecorder()
    process_start = _SCRIPT_START_NS
    phases._rows["process.python_and_stdlib_bootstrap"] = _Aggregate(
        1,
        time.perf_counter_ns() - process_start,
    )
    with phases.phase("import.cprofile"):
        profiler = _start_profiler(args.profile)
    modules_before = set(sys.modules)
    with phases.phase("import.matplotlib_pyplot"):
        importlib.import_module("matplotlib.pyplot")
    with phases.phase("import.numpy"):
        np = importlib.import_module("numpy")
    with phases.phase("import.axonscope"):
        axs = importlib.import_module("axonscope")
    modules_after = set(sys.modules)
    import_end = time.perf_counter_ns()

    try:
        with phases.phase("startup.workload_total"):
            workload = _build_workload(
                axs,
                np,
                phases,
                fibers_per_family=args.fibers_per_family,
                seed=args.seed,
                template_policy=args.template_policy,
                waveform_update_policy=args.waveform_update_policy,
            )
        startup_end = time.perf_counter_ns()

        curve = None
        if args.scope != "startup":
            values = workload["current_steps"]
            if args.scope == "first-amplitude":
                values = values[:1]
            policy = axs.ExecutionPolicy(
                runtime=axs.runtime.jax,
                device=(
                    axs.Device.gpu(0)
                    if args.platform == "gpu"
                    else axs.Device.cpu()
                ),
                precision=(
                    axs.PrecisionPolicy.float32()
                    if args.platform == "gpu"
                    else None
                ),
            )
            trace_dir = output / "runtime_trace"
            with axs.benchmark(
                trace_dir,
                print_summary=False,
                save=True,
                sync_device=True,
                record_shapes=True,
                memory_trace="rss",
            ):
                with phases.phase("protocol.recruitment_sweep"):
                    curve = axs.protocols.recruitment_sweep(
                        workload["pool"],
                        update=workload["update"],
                        values=values,
                        duration=args.duration_ms * axs.ms,
                        dt=args.dt_ms * axs.ms,
                        criterion=workload["criterion"],
                        recording=axs.Recording.none(),
                        progress=False,
                        solver_progress=False,
                        execution_policy=policy,
                    )
        execution_end = time.perf_counter_ns()
    finally:
        _stop_profiler(profiler, output=output, top_n=args.profile_top)

    phases._rows["process.import_total"] = _Aggregate(1, import_end - process_start)
    phases._rows["process.startup_total"] = _Aggregate(1, startup_end - process_start)
    phases._rows["process.total"] = _Aggregate(1, execution_end - process_start)
    _write_summary(output, phases.rows())
    _write_manifest(
        output,
        args,
        imported_modules=modules_after - modules_before,
        workload=workload,
        curve=curve,
    )
    _print_summary(phases.rows())
    print(f"results: {output}")
    return 0


def _build_workload(
    axs: Any,
    np: Any,
    phases: _PhaseRecorder,
    *,
    fibers_per_family: int,
    seed: int,
    template_policy: str,
    waveform_update_policy: str,
) -> dict[str, Any]:
    with phases.phase("setup.constants_and_stimulus"):
        rng = np.random.default_rng(seed)
        circle_radius = 125.0 * axs.um
        fiber_length = 1500.0 * axs.um
        stim_start = 0.20 * axs.ms
        pulse_width = 0.10 * axs.ms
        sigma = 0.3 * axs.S_per_m
        current_steps = np.asarray(DEFAULT_AMPLITUDES_UA) * axs.uA
        electrode = axs.analytical.PointSourceElectrode(
            x=fiber_length / 2.0,
            y=0.0 * axs.um,
            z=0.0 * axs.um,
            min_distance=5.0 * axs.um,
        )
        zero_current = axs.Stimulus.pulse(
            start=stim_start,
            duration=pulse_width,
            amplitude=0.0 * axs.uA,
        )

    with phases.phase("setup.sample_positions"):
        radius_um = circle_radius.to(axs.um).magnitude
        unmyelinated_angles = rng.uniform(0.0, 2.0 * np.pi, fibers_per_family)
        unmyelinated_radii = radius_um * np.sqrt(
            rng.uniform(0.0, 1.0, fibers_per_family)
        )
        unmyelinated_y = unmyelinated_radii * np.cos(unmyelinated_angles) * axs.um
        unmyelinated_z = unmyelinated_radii * np.sin(unmyelinated_angles) * axs.um
        myelinated_angles = rng.uniform(0.0, 2.0 * np.pi, fibers_per_family)
        myelinated_radii = radius_um * np.sqrt(
            rng.uniform(0.0, 1.0, fibers_per_family)
        )
        myelinated_y = myelinated_radii * np.cos(myelinated_angles) * axs.um
        myelinated_z = myelinated_radii * np.sin(myelinated_angles) * axs.um

    with phases.phase("setup.sample_diameters"):
        unmyelinated_diameters = rng.uniform(0.4, 1.2, fibers_per_family) * axs.um
        myelinated_diameters = (
            rng.choice(np.asarray([7.3, 10.0, 12.8]), size=fibers_per_family)
            * axs.um
        )

    shared_templates = template_policy == "shared"
    unmyelinated_templates: dict[float, tuple[Any, Any]] = {}
    myelinated_templates: dict[float, tuple[Any, Any]] = {}
    if shared_templates:
        # Unit validation and diameter quantization belong to the unique template
        # description, not to every population row that references it.
        with phases.phase("population.canonicalize_template_parameters"):
            unmyelinated_diameter_values = axs.axons.round_axon_diameter_values_um(
                unmyelinated_diameters.to(axs.um).magnitude
            )
            myelinated_diameter_values = axs.axons.round_axon_diameter_values_um(
                myelinated_diameters.to(axs.um).magnitude
            )
    else:
        unmyelinated_diameter_values = unmyelinated_diameters
        myelinated_diameter_values = myelinated_diameters

    pool: list[Any] = []
    with phases.phase("population.unmyelinated_total"):
        for diameter, y, z in zip(
            unmyelinated_diameter_values,
            unmyelinated_y,
            unmyelinated_z,
            strict=True,
        ):
            if shared_templates:
                diameter_key = float(diameter)
                template = unmyelinated_templates.get(diameter_key)
                if template is None:
                    axon = phases.call(
                        "population.unmyelinated.axon",
                        axs.axons.RattayAberham,
                        length=fiber_length,
                        diameter=diameter_key * axs.um,
                        compartments=61,
                        celsius=37.0 * axs.degC,
                    )
                    positions = phases.call(
                        "population.unmyelinated.positions",
                        axon.layout.position_values,
                        unit=axs.um,
                    ) * axs.um
                    template = (axon, positions)
                    unmyelinated_templates[diameter_key] = template
                axon, positions = template
            else:
                axon = phases.call(
                    "population.unmyelinated.axon",
                    axs.axons.RattayAberham,
                    length=fiber_length,
                    diameter=diameter,
                    compartments=61,
                    celsius=37.0 * axs.degC,
                )
                positions = phases.call(
                    "population.unmyelinated.positions",
                    axon.layout.position_values,
                    unit=axs.um,
                ) * axs.um
            extracellular = phases.call(
                "population.unmyelinated.footprint",
                axs.analytical.point_source_stimulation,
                electrode,
                positions,
                sigma=sigma,
                stimulus=zero_current,
                axon_y=y,
                axon_z=z,
            )
            simulation = phases.call(
                "population.unmyelinated.instance",
                axs.AxonInstance,
                axon,
            )
            phases.call(
                "population.unmyelinated.attach",
                simulation.add_extracellular_stimulation,
                stimulation=extracellular,
            )
            pool.append(simulation)

    with phases.phase("population.myelinated_total"):
        for diameter, y, z in zip(
            myelinated_diameter_values,
            myelinated_y,
            myelinated_z,
            strict=True,
        ):
            if shared_templates:
                diameter_key = float(diameter)
                template = myelinated_templates.get(diameter_key)
                if template is None:
                    axon = phases.call(
                        "population.myelinated.axon",
                        axs.axons.MRG,
                        diameter=diameter_key * axs.um,
                        nodes=4,
                        length=fiber_length,
                        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
                    )
                    positions = phases.call(
                        "population.myelinated.positions",
                        axon.layout.position_values,
                        unit=axs.um,
                    ) * axs.um
                    template = (axon, positions)
                    myelinated_templates[diameter_key] = template
                axon, positions = template
            else:
                axon = phases.call(
                    "population.myelinated.axon",
                    axs.axons.MRG,
                    diameter=diameter,
                    nodes=4,
                    length=fiber_length,
                    compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
                )
                positions = phases.call(
                    "population.myelinated.positions",
                    axon.layout.position_values,
                    unit=axs.um,
                ) * axs.um
            extracellular = phases.call(
                "population.myelinated.footprint",
                axs.analytical.point_source_stimulation,
                electrode,
                positions,
                sigma=sigma,
                stimulus=zero_current,
                axon_y=y,
                axon_z=z,
            )
            simulation = phases.call(
                "population.myelinated.instance",
                axs.AxonInstance,
                axon,
            )
            phases.call(
                "population.myelinated.attach",
                simulation.add_extracellular_stimulation,
                stimulation=extracellular,
            )
            pool.append(simulation)

    with phases.phase("protocol.criterion_and_update"):
        criterion = axs.analysis.ActivationCriterion(
            threshold=0.0 * axs.mV,
            blanking=stim_start,
            target=axs.positions.ALL,
        )

        def waveform(current_magnitude: Any) -> Any:
            return axs.Stimulus.pulse(
                start=stim_start,
                duration=pulse_width,
                amplitude=-current_magnitude,
            )

        if waveform_update_policy == "typed":
            update_point_source_current = axs.protocols.ExtracellularWaveformUpdate(
                waveform
            )
        else:
            def update_point_source_current(
                simulation: Any,
                current_magnitude: Any,
            ) -> None:
                stimulation = simulation.extracellular_stimulation
                if stimulation is None:
                    raise ValueError(
                        "simulation has no extracellular stimulation to update."
                    )
                drive = stimulation.drives[0]
                simulation.add_extracellular_stimulation(
                    stimulation=stimulation.replace_drive(
                        drive.id,
                        stimulus=waveform(current_magnitude),
                    ),
                    replace=True,
                )

    population = phases.call("population.index_templates", axs.AxonPopulation, pool)

    return {
        "pool": population,
        "criterion": criterion,
        "update": update_point_source_current,
        "current_steps": current_steps,
        "unique_axon_templates": len(population.axon_templates),
        "unique_unmyelinated_templates": (
            len(unmyelinated_templates) if shared_templates else fibers_per_family
        ),
        "unique_myelinated_templates": (
            len(myelinated_templates) if shared_templates else fibers_per_family
        ),
    }


def _start_profiler(enabled: bool):
    if not enabled:
        return None
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def _stop_profiler(profiler: Any, *, output: Path, top_n: int) -> None:
    if profiler is None:
        return
    import io
    import pstats

    profiler.disable()
    profiler.dump_stats(output / "startup.prof")
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(top_n)
    (output / "profile_top.txt").write_text(stream.getvalue(), encoding="utf-8")


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("phase", "count", "total_ms", "mean_ms"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    output: Path,
    args: argparse.Namespace,
    *,
    imported_modules: set[str],
    workload: dict[str, Any],
    curve: Any,
) -> None:
    payload = {
        "script": "basic_08_startup",
        "source_example": "examples/basic/08_recruitment_curve_population.py",
        "fibers_per_family": args.fibers_per_family,
        "population_size": len(workload["pool"]),
        "scope": args.scope,
        "platform": args.platform,
        "template_policy": args.template_policy,
        "waveform_update_policy": args.waveform_update_policy,
        "unique_axon_templates": workload["unique_axon_templates"],
        "unique_unmyelinated_templates": workload[
            "unique_unmyelinated_templates"
        ],
        "unique_myelinated_templates": workload["unique_myelinated_templates"],
        "duration_ms": args.duration_ms,
        "dt_ms": args.dt_ms,
        "seed": args.seed,
        "profile": bool(args.profile),
        "matplotlib_config_dir": os.environ.get("MPLCONFIGDIR"),
        "imported_module_count": len(imported_modules),
        "imported_modules": sorted(imported_modules),
        "activation_counts": None if curve is None else curve.count.tolist(),
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"{row['phase']:<42} "
            f"{float(row['total_ms']):>12.3f} ms "
            f"({int(row['count'])}x)"
        )


if __name__ == "__main__":
    raise SystemExit(main())
