from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import axonscope as axs
from axonscope.benchmarking import benchmark_span
from axonscope.positions import PositionSelector
from axonscope.results import VM_RASTER_OBSERVATION_KEY, activation_values_from_vm_raster
from axonscope.signals import MEMBRANE_VOLTAGE, Signal
from axonscope.utils import units

from benchmark.workloads.curve_options import (
    case_name,
    resolved_options,
    write_cases_csv,
)


RESULT_FIELDS = (
    "script",
    "case_name",
    "phase",
    "repeat",
    "curve",
    "iteration",
    "amplitude_uA",
    "row",
    "family",
    "diameter_um",
    "activated",
    "lower_uA",
    "upper_uA",
    "threshold_uA",
    "status",
    "activation_fraction",
)

CURVE_SUMMARY_FIELDS = (
    "script",
    "case_name",
    "phase",
    "repeat",
    "curve",
    "row",
    "family",
    "diameter_um",
    "threshold_uA",
    "status",
    "amplitude_uA",
    "activation_count",
    "activation_fraction",
)


@dataclass(frozen=True, slots=True)
class _PhasePool:
    pool: tuple[axs.AxonInstance, ...]
    row_meta: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _AxonTemplate:
    axon: Any
    family: str
    diameter_um: float
    electrode_x_um: float
    positions: Any
    positions_um: np.ndarray


def run_threshold_curves(args: Any) -> int:
    """Run the currently supported activation-threshold benchmark."""

    options = resolved_options(args)
    _validate_real_run("threshold_curves", options)
    threshold_kind = str(options.get("threshold_kind", "activation"))
    if threshold_kind != "activation":
        raise SystemExit(
            "Real P11A threshold runs currently support --threshold-kind activation. "
            "Block thresholds stay in the validated case list until the block protocol "
            "semantics are defined."
        )

    output = _prepare_output("threshold_curves", options)
    if output is None:
        return 0

    selected_case = case_name("threshold_curves", options)
    phase_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with _benchmark_session(output, options, "threshold_curves", selected_case) as session:
        for warmup in range(int(options["warmups"])):
            _run_threshold_phase(
                options,
                selected_case=selected_case,
                phase="warmup",
                repeat=warmup,
                result_rows=phase_rows,
                summary_rows=summary_rows,
            )
        for repeat in range(int(options["repeats"])):
            _run_threshold_phase(
                options,
                selected_case=selected_case,
                phase="repeat",
                repeat=repeat,
                result_rows=phase_rows,
                summary_rows=summary_rows,
            )
        session.metadata["curve_result_rows"] = len(phase_rows)
        session.metadata["curve_summary_rows"] = len(summary_rows)

    _write_csv(output / "results.csv", RESULT_FIELDS, phase_rows)
    _write_csv(output / "curve_summary.csv", CURVE_SUMMARY_FIELDS, summary_rows)
    _write_manifest(output, "threshold_curves", selected_case, options)
    print(f"benchmark: {selected_case}")
    print(f"wrote: {output}")
    return 0


def run_recruitment_curves(args: Any) -> int:
    """Run the currently supported recruitment-curve benchmark."""

    options = resolved_options(args)
    _validate_real_run("recruitment_curves", options)

    output = _prepare_output("recruitment_curves", options)
    if output is None:
        return 0

    selected_case = case_name("recruitment_curves", options)
    phase_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with _benchmark_session(output, options, "recruitment_curves", selected_case) as session:
        for warmup in range(int(options["warmups"])):
            _run_recruitment_phase(
                options,
                selected_case=selected_case,
                phase="warmup",
                repeat=warmup,
                result_rows=phase_rows,
                summary_rows=summary_rows,
            )
        for repeat in range(int(options["repeats"])):
            _run_recruitment_phase(
                options,
                selected_case=selected_case,
                phase="repeat",
                repeat=repeat,
                result_rows=phase_rows,
                summary_rows=summary_rows,
            )
        session.metadata["curve_result_rows"] = len(phase_rows)
        session.metadata["curve_summary_rows"] = len(summary_rows)

    _write_csv(output / "results.csv", RESULT_FIELDS, phase_rows)
    _write_csv(output / "curve_summary.csv", CURVE_SUMMARY_FIELDS, summary_rows)
    _write_manifest(output, "recruitment_curves", selected_case, options)
    print(f"benchmark: {selected_case}")
    print(f"wrote: {output}")
    return 0


class _BenchmarkSessionContext:
    def __init__(
        self,
        output: Path,
        options: dict[str, Any],
        script_name: str,
        selected_case: str,
    ) -> None:
        self.output = output
        self.options = options
        self.script_name = script_name
        self.selected_case = selected_case
        self._manager: Any | None = None
        self.session: Any | None = None

    def __enter__(self):
        self._manager = axs.benchmark(
            self.output,
            print_summary=False,
            save=True,
            sync_device=True,
            record_shapes=True,
            memory_trace=self.options["memory_trace"],
            memory_top_n=int(self.options["memory_top_n"]),
            profile=bool(self.options["profile"]),
            profile_backend=self.options["profile_backend"],
            profile_output=self.options["profile_output"],
            profile_create_perfetto=bool(self.options["profile_create_perfetto"]),
            jax_device_memory_profile=bool(self.options["jax_device_memory_profile"]),
            jax_device_memory_profile_stages=self.options["jax_device_memory_profile_stages"],
        )
        self.session = self._manager.__enter__()
        self.session.metadata["benchmark_script"] = self.script_name
        self.session.metadata["benchmark_case_name"] = self.selected_case
        self.session.metadata["benchmark_options"] = dict(self.options)
        return self.session

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        assert self._manager is not None
        return self._manager.__exit__(exc_type, exc, tb)


def _benchmark_session(
    output: Path,
    options: dict[str, Any],
    script_name: str,
    selected_case: str,
) -> _BenchmarkSessionContext:
    return _BenchmarkSessionContext(output, options, script_name, selected_case)


def _run_threshold_phase(
    options: dict[str, Any],
    *,
    selected_case: str,
    phase: str,
    repeat: int,
    result_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    n_axons = int(options["n_axons"])
    low = np.full(n_axons, float(options["amplitude_min"]), dtype=float)
    high = np.full(n_axons, float(options["amplitude_max"]), dtype=float)
    phase_pool = _build_phase_pool(
        options,
        low,
        selected_case=selected_case,
        phase=phase,
        repeat=repeat,
        curve="activation_threshold",
        iteration=-2,
        curve_context="threshold",
    )
    active_low, row_meta = _evaluate_amplitudes(
        options,
        phase_pool,
        low,
        target=axs.positions.DISTAL,
        selected_case=selected_case,
        phase=phase,
        repeat=repeat,
        curve="activation_threshold",
        iteration=-2,
        update_amplitudes=False,
    )
    active_high, row_meta = _evaluate_amplitudes(
        options,
        phase_pool,
        high,
        target=axs.positions.DISTAL,
        selected_case=selected_case,
        phase=phase,
        repeat=repeat,
        curve="activation_threshold",
        iteration=-1,
    )
    _append_activation_rows(
        result_rows,
        script="threshold_curves",
        selected_case=selected_case,
        phase=phase,
        repeat=repeat,
        curve="activation_threshold",
        iteration=-2,
        amplitudes=low,
        activated=active_low,
        row_meta=row_meta,
    )
    _append_activation_rows(
        result_rows,
        script="threshold_curves",
        selected_case=selected_case,
        phase=phase,
        repeat=repeat,
        curve="activation_threshold",
        iteration=-1,
        amplitudes=high,
        activated=active_high,
        row_meta=row_meta,
    )

    status = np.full(n_axons, "bracketed", dtype=object)
    status[active_low] = "below_range"
    status[~active_high] = "above_range"
    searching = status == "bracketed"
    tolerance = max(float(options["amplitude_tolerance"]), 0.0)

    for iteration in range(int(options["max_iterations"])):
        if not bool(np.any(searching)):
            break
        mid = (low + high) / 2.0
        activated, row_meta = _evaluate_amplitudes(
            options,
            phase_pool,
            mid,
            target=axs.positions.DISTAL,
            selected_case=selected_case,
            phase=phase,
            repeat=repeat,
            curve="activation_threshold",
            iteration=iteration,
        )
        _append_activation_rows(
            result_rows,
            script="threshold_curves",
            selected_case=selected_case,
            phase=phase,
            repeat=repeat,
            curve="activation_threshold",
            iteration=iteration,
            amplitudes=mid,
            activated=activated,
            row_meta=row_meta,
        )
        active_searching = activated & searching
        inactive_searching = ~activated & searching
        high[active_searching] = mid[active_searching]
        low[inactive_searching] = mid[inactive_searching]
        if tolerance > 0.0 and float(np.max(high[searching] - low[searching])) <= tolerance:
            break

    thresholds = np.where(status == "bracketed", high, np.nan)
    summary_status = np.where(status == "bracketed", "ok", status)
    for row, meta in enumerate(row_meta):
        summary_rows.append(
            {
                "script": "threshold_curves",
                "case_name": selected_case,
                "phase": phase,
                "repeat": repeat,
                "curve": "activation_threshold",
                "row": row,
                "family": meta["family"],
                "diameter_um": meta["diameter_um"],
                "threshold_uA": thresholds[row],
                "status": summary_status[row],
                "amplitude_uA": "",
                "activation_count": "",
                "activation_fraction": "",
            }
        )


def _run_recruitment_phase(
    options: dict[str, Any],
    *,
    selected_case: str,
    phase: str,
    repeat: int,
    result_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    count = max(int(options["amplitude_count"]), 1)
    if count == 1:
        amplitudes = np.asarray([float(options["amplitude_max"])], dtype=float)
    else:
        amplitudes = np.linspace(
            float(options["amplitude_min"]),
            float(options["amplitude_max"]),
            count,
            dtype=float,
        )
    for iteration, amplitude in enumerate(amplitudes):
        values = np.full(int(options["n_axons"]), float(amplitude), dtype=float)
        if iteration == 0:
            phase_pool = _build_phase_pool(
                options,
                values,
                selected_case=selected_case,
                phase=phase,
                repeat=repeat,
                curve="recruitment",
                iteration=iteration,
                curve_context="recruitment",
            )
        activated, row_meta = _evaluate_amplitudes(
            options,
            phase_pool,
            values,
            target=axs.positions.ALL,
            selected_case=selected_case,
            phase=phase,
            repeat=repeat,
            curve="recruitment",
            iteration=iteration,
            update_amplitudes=iteration > 0,
        )
        _append_activation_rows(
            result_rows,
            script="recruitment_curves",
            selected_case=selected_case,
            phase=phase,
            repeat=repeat,
            curve="recruitment",
            iteration=iteration,
            amplitudes=values,
            activated=activated,
            row_meta=row_meta,
        )
        summary_rows.append(
            {
                "script": "recruitment_curves",
                "case_name": selected_case,
                "phase": phase,
                "repeat": repeat,
                "curve": "recruitment",
                "row": "",
                "family": "",
                "diameter_um": "",
                "threshold_uA": "",
                "status": "",
                "amplitude_uA": float(amplitude),
                "activation_count": int(np.sum(activated)),
                "activation_fraction": _activation_fraction(activated),
            }
        )


def _build_phase_pool(
    options: dict[str, Any],
    amplitudes_uA: np.ndarray,
    *,
    selected_case: str,
    phase: str,
    repeat: int,
    curve: str,
    iteration: int,
    curve_context: str,
) -> _PhasePool:
    with benchmark_span(
        "curve.build_pool",
        phase=phase,
        repeat=repeat,
        curve=curve,
        iteration=iteration,
        n_axons=int(options["n_axons"]),
        case_name=selected_case,
    ):
        pool, row_meta = _build_pool(
            options,
            amplitudes_uA,
            curve_context=curve_context,
        )
    return _PhasePool(pool=pool, row_meta=row_meta)


def _evaluate_amplitudes(
    options: dict[str, Any],
    phase_pool: _PhasePool,
    amplitudes_uA: np.ndarray,
    *,
    target: Any,
    selected_case: str,
    phase: str,
    repeat: int,
    curve: str,
    iteration: int,
    update_amplitudes: bool = True,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    if update_amplitudes:
        with benchmark_span(
            "curve.update_amplitudes",
            phase=phase,
            repeat=repeat,
            curve=curve,
            iteration=iteration,
            n_axons=int(options["n_axons"]),
        ):
            _update_pool_amplitudes(
                phase_pool.pool,
                amplitudes_uA,
                options,
            )
    with benchmark_span(
        "curve.activation_definition",
        phase=phase,
        repeat=repeat,
        curve=curve,
        iteration=iteration,
        recording=options["recording"],
    ):
        activation = axs.analysis.Activation(
            threshold=0.0 * axs.mV,
            blanking=_stim_start_ms(options) * axs.ms,
            target=target,
        )
        observers = (activation,) if options["recording"] == "observer_only" else None
    with benchmark_span(
        "curve.runtime_options",
        phase=phase,
        repeat=repeat,
        curve=curve,
        iteration=iteration,
        recording=options["recording"],
        platform=options["platform"],
        precision=options["precision"],
        time_chunk_policy=options.get("time_chunk_policy", "default"),
        time_chunk_steps=options.get("time_chunk_steps"),
        gpu_solver=options.get("double_cable_block_solver", "auto"),
        tiled_thomas_block_b=options.get("tiled_thomas_block_b"),
        benchmark_double_cable_block_solver=options.get(
            "benchmark_double_cable_block_solver"
        ),
    ):
        recording = _recording_policy(options)
        batch_options = _batch_options(options)
        execution_policy = _execution_policy(options)
    with benchmark_span(
        "curve.construct_simulation",
        phase=phase,
        repeat=repeat,
        curve=curve,
        iteration=iteration,
        recording=options["recording"],
        n_axons=int(options["n_axons"]),
    ):
        simulation = axs.AxonSimulation(
            axs.AxonPopulation(phase_pool.pool, name=selected_case),
            duration=float(options["tsim"]) * axs.ms,
            dt=float(options["dt"]) * axs.ms,
            recording=recording,
            batch_options=batch_options,
            observers=observers,
            execution_policy=execution_policy,
            progress=False,
        )
    with benchmark_span(
        "curve.simulate",
        phase=phase,
        repeat=repeat,
        curve=curve,
        iteration=iteration,
        recording=options["recording"],
        platform=options["platform"],
        precision=options["precision"],
        time_chunk_policy=options.get("time_chunk_policy", "default"),
        time_chunk_steps=options.get("time_chunk_steps"),
        gpu_solver=options.get("double_cable_block_solver", "auto"),
        tiled_thomas_block_b=options.get("tiled_thomas_block_b"),
        benchmark_double_cable_block_solver=options.get(
            "benchmark_double_cable_block_solver"
        ),
    ):
        result = simulation.run()
    with benchmark_span(
        "curve.analyze_activation",
        phase=phase,
        repeat=repeat,
        curve=curve,
        iteration=iteration,
        recording=options["recording"],
    ):
        activated = _activation_values(result, activation, recording_mode=options["recording"])
    return activated, phase_pool.row_meta


def _update_pool_amplitudes(
    pool: tuple[axs.AxonInstance, ...],
    amplitudes_uA: np.ndarray,
    options: dict[str, Any],
) -> None:
    amplitudes = np.asarray(amplitudes_uA, dtype=float).reshape(-1)
    if amplitudes.shape != (len(pool),):
        raise ValueError(f"expected one amplitude per axon, got shape {amplitudes.shape}.")
    stimulus_cache: dict[float, Any] = {}
    with benchmark_span(
        "curve.update_amplitudes.rows",
        n_axons=len(pool),
        unique_amplitudes=int(np.unique(amplitudes).size),
    ):
        for instance, amplitude in zip(pool, amplitudes, strict=True):
            stimulation = instance.extracellular_stimulation
            if stimulation is None:
                raise RuntimeError("benchmark pool row has no extracellular stimulation.")
            drive = stimulation.drives[0]
            amplitude_key = float(amplitude)
            stimulus = stimulus_cache.get(amplitude_key)
            if stimulus is None:
                with benchmark_span("curve.update_amplitudes.stimulus_build"):
                    stimulus = _stimulus_for_amplitude(options, amplitude_key)
                stimulus_cache[amplitude_key] = stimulus
            updated = stimulation.replace_drive(
                drive.id,
                stimulus=stimulus,
            )
            instance.add_extracellular_stimulation(stimulation=updated, replace=True)


def _build_pool(
    options: dict[str, Any],
    amplitudes_uA: np.ndarray,
    *,
    curve_context: str,
) -> tuple[tuple[axs.AxonInstance, ...], tuple[dict[str, Any], ...]]:
    rng = np.random.default_rng(int(options["seed"]))
    n_axons = int(options["n_axons"])
    amplitudes = np.asarray(amplitudes_uA, dtype=float).reshape(-1)
    if amplitudes.shape != (n_axons,):
        raise ValueError(f"expected one amplitude per axon, got shape {amplitudes.shape}.")

    with benchmark_span(
        "curve.build_pool.diameter_grid",
        diameters=options["diameters"],
        n_axons=n_axons,
    ):
        if options["diameters"] == "different_diameters":
            single_diameters = np.linspace(0.4, 1.2, n_axons, dtype=float)
            double_values = np.asarray([5.7, 7.3, 10.0, 12.8, 16.0], dtype=float)
            double_diameters = np.resize(double_values, n_axons)
        else:
            single_diameters = np.full(n_axons, 0.8, dtype=float)
            double_diameters = np.full(n_axons, 7.3, dtype=float)

    with benchmark_span(
        "curve.build_pool.spatial_layout",
        curve_context=curve_context,
        n_axons=n_axons,
    ):
        if curve_context == "recruitment":
            radius_um = 125.0
            angles = rng.uniform(0.0, 2.0 * np.pi, n_axons)
            radii = radius_um * np.sqrt(rng.uniform(0.0, 1.0, n_axons))
            y_um = radii * np.cos(angles)
            z_um = radii * np.sin(angles)
        else:
            y_um = np.zeros(n_axons, dtype=float)
            z_um = np.zeros(n_axons, dtype=float)

    templates: dict[tuple[str, float], _AxonTemplate] = {}
    stimulus_cache: dict[float, Any] = {}
    pool: list[axs.AxonInstance] = []
    row_meta: list[dict[str, Any]] = []
    with benchmark_span("curve.build_pool.rows", n_axons=n_axons):
        for row in range(n_axons):
            row_cable = _row_cable(options, row)
            diameter_um = (
                float(double_diameters[row])
                if row_cable == "double_cable"
                else float(single_diameters[row])
            )
            diameter_um = axs.axons.round_axon_diameter_um(diameter_um)
            template_key = (row_cable, diameter_um)
            template = templates.get(template_key)
            if template is None:
                with benchmark_span(
                    "curve.build_pool.template_build",
                    cable=row_cable,
                    diameter_um=diameter_um,
                ):
                    template = _build_axon_template(options, row_cable, diameter_um)
                templates[template_key] = template

            amplitude_key = float(amplitudes[row])
            stimulus = stimulus_cache.get(amplitude_key)
            if stimulus is None:
                stimulus = _stimulus_for_amplitude(options, amplitude_key)
                stimulus_cache[amplitude_key] = stimulus
            axon_id = axs.AxonId(f"row_{row:05d}")
            stimulation = _point_source_stimulation_for_template(
                template,
                stimulus=stimulus,
                curve_context=curve_context,
                axon_y_um=float(y_um[row]),
                axon_z_um=float(z_um[row]),
                axon_id=axon_id,
            )
            instance = axs.AxonInstance(template.axon)
            instance.add_extracellular_stimulation(stimulation=stimulation)
            pool.append(instance)
            row_meta.append(
                {
                    "row": row,
                    "family": template.family,
                    "diameter_um": float(template.diameter_um),
                    "cable": row_cable,
                    "axon_y_um": float(y_um[row]),
                    "axon_z_um": float(z_um[row]),
                }
            )
    return tuple(pool), tuple(row_meta)


def _build_axon_template(
    options: dict[str, Any],
    row_cable: str,
    diameter_um: float,
) -> _AxonTemplate:
    if row_cable == "double_cable":
        axon, family, resolved_diameter_um = _double_cable_axon(options, diameter_um)
        electrode_x_um = float(axon.node_position("center", unit="micrometer").magnitude)
    else:
        axon, family, resolved_diameter_um = _single_cable_axon(options, diameter_um)
        electrode_x_um = _fiber_length_um(options) / 2.0
    positions_um = np.asarray(axon.layout.position_values(unit="micrometer"), dtype=float)
    return _AxonTemplate(
        axon=axon,
        family=family,
        diameter_um=float(resolved_diameter_um),
        electrode_x_um=float(electrode_x_um),
        positions=positions_um * axs.um,
        positions_um=positions_um,
    )


def _point_source_stimulation_for_template(
    template: _AxonTemplate,
    *,
    stimulus: Any,
    curve_context: str,
    axon_y_um: float,
    axon_z_um: float,
    axon_id: Any,
) -> Any:
    electrode_z_um = 0.0 if curve_context == "recruitment" else 100.0
    min_distance_um = 5.0 if curve_context == "recruitment" else 1e-3
    dx_m = (template.positions_um - float(template.electrode_x_um)) * 1e-6
    dy_m = (0.0 - float(axon_y_um)) * 1e-6
    dz_m = (electrode_z_um - float(axon_z_um)) * 1e-6
    radius_m = np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)
    radius_m = np.maximum(radius_m, min_distance_um * 1e-6)
    values = 1.0 / (4.0 * np.pi * 0.3 * radius_m)
    footprint = axs.ExtracellularFootprint(
        values=values,
        positions=template.positions,
        axon_ids=(axon_id,),
        metadata={
            "builder": "benchmark.workloads.curve_runtime.point_source",
            "source": "point_source_helper",
            "electrode_x_um": template.electrode_x_um,
            "electrode_y_um": 0.0,
            "electrode_z_um": electrode_z_um,
            "axon_x_offset_um": 0.0,
            "axon_y_um": float(axon_y_um),
            "axon_z_um": float(axon_z_um),
            "sigma_S_m": 0.3,
        },
    )
    drive = axs.ExtracellularDrive(
        id=axs.DriveId("point_source"),
        footprint=footprint,
        stimulus=stimulus,
        metadata={"source": "point_source_helper"},
    )
    return axs.ExtracellularStimulation((drive,))


def _single_cable_axon(options: dict[str, Any], diameter_um: float) -> tuple[Any, str, float]:
    length = _fiber_length_um(options) * axs.um
    compartments = max(int(options["nx"]), 3)
    if options["precision"] == "fp64":
        membrane = axs.membranes.RattayAberham(dtype=np.float64)
        axon = axs.axons.Unmyelinated(
            membrane=membrane,
            length=length,
            diameter=float(diameter_um) * axs.um,
            compartments=compartments,
            v_init=-70.0 * axs.mV,
            temperature=37.0 * axs.degC,
        )
        return (axon, "rattay_aberham", float(axon.diameter))
    axon = axs.axons.RattayAberham(
        length=length,
        diameter=float(diameter_um) * axs.um,
        compartments=compartments,
        celsius=37.0 * axs.degC,
    )
    return (axon, "rattay_aberham", float(axon.diameter))


def _double_cable_axon(options: dict[str, Any], diameter_um: float) -> tuple[Any, str, float]:
    nodes = max(3, min(9, int(options["nx"]) // 5))
    axon = axs.axons.MRG(
        diameter=float(diameter_um) * axs.um,
        nodes=nodes,
        compartments={"node": 1, "MYSA": 1, "FLUT": 1, "STIN": 1},
    )
    return (axon, "mrg", float(axon.diameter))


def _stimulus_for_amplitude(options: dict[str, Any], amplitude_uA: float) -> Any:
    start = _stim_start_ms(options) * axs.ms
    width = _pulse_width_ms(options) * axs.ms
    amplitude = float(amplitude_uA) * axs.uA
    if options["stimulation"] == "monophasic":
        return axs.Stimulus.pulse(
            start=start,
            duration=width,
            amplitude=-amplitude,
        )
    if options["stimulation"] == "biphasic":
        return axs.Stimulus.biphasic(
            start=start,
            cathodic_amplitude=amplitude,
            cathodic_duration=width,
            interphase=min(0.02, float(options["tsim"]) * 0.01) * axs.ms,
        )
    raise SystemExit("--stimulation custom requires a benchmark workload adapter.")


def _activation_values(result: Any, activation: Any, *, recording_mode: str) -> np.ndarray:
    if recording_mode == "observer_only":
        with benchmark_span("curve.analyze_activation.vm_raster_extract"):
            observations = getattr(result, "observations", None)
            if observations is None:
                raise RuntimeError("observer-only benchmark produced no observations.")
            raster = observations[VM_RASTER_OBSERVATION_KEY]
        with benchmark_span("curve.analyze_activation.vm_raster_values"):
            values = activation_values_from_vm_raster(raster, activation)
    else:
        with benchmark_span("curve.analyze_activation.dense_values"):
            values = _dense_activation_values(result, activation)
        if values is None:
            with benchmark_span("curve.analyze_activation.result_analyze"):
                analysis = result.analyze(activation)
                values = analysis.values
    with benchmark_span(
        "curve.analyze_activation.materialize_values",
        recording=recording_mode,
    ):
        return np.asarray(values, dtype=bool).reshape(-1)


def _dense_activation_values(result: Any, activation: Any) -> np.ndarray | None:
    signal = getattr(activation, "signal", MEMBRANE_VOLTAGE)
    if not isinstance(signal, Signal) or signal.id != MEMBRANE_VOLTAGE.id:
        return None
    target = getattr(activation, "target", None)
    if not isinstance(target, PositionSelector):
        return None
    cohorts = tuple(getattr(result, "_cohorts", ()))
    size = getattr(result, "size", None)
    if not cohorts or size is None:
        return None
    try:
        threshold_mV = units.to_mV(getattr(activation, "threshold"))
        blanking_ms = units.to_ms(getattr(activation, "blanking"))
    except (TypeError, ValueError):
        return None
    if blanking_ms < 0.0:
        return None

    values = np.zeros(int(size), dtype=bool)
    try:
        for cohort in cohorts:
            vm = np.asarray(cohort.Vm)
            if vm.ndim != 3:
                return None
            time_ms = np.asarray(cohort.t, dtype=float)
            if time_ms.ndim != 1 or time_ms.shape[0] != vm.shape[1]:
                return None
            if vm.shape[2] == 0:
                return None
            start = int(np.searchsorted(time_ms, float(blanking_ms), side="left"))
            if start >= time_ms.shape[0]:
                continue
            groups = _dense_activation_groups(cohort, width=vm.shape[2], target=target)
            for group in groups:
                rows = np.asarray(group["rows"], dtype=int)
                selected = np.asarray(group["selected"], dtype=int)
                input_indices = np.asarray(group["input_indices"], dtype=int)
                window = _dense_activation_window(
                    vm,
                    rows=rows,
                    time_slice=slice(start, None),
                    selected=selected,
                )
                values[input_indices] = np.max(window, axis=(1, 2)) >= float(threshold_mV)
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        return None
    return values


def _dense_activation_groups(
    cohort: Any,
    *,
    width: int,
    target: PositionSelector,
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for row, (axon, record_indices) in enumerate(
        zip(cohort.axons, cohort.record_indices, strict=True)
    ):
        full_positions_um = np.asarray(
            axon.layout.position_values(unit="micrometer"),
            dtype=float,
        )
        if record_indices is None:
            if full_positions_um.shape != (int(width),):
                raise ValueError("recorded positions must match Vm columns.")
            original_indices = np.arange(int(width), dtype=int)
            positions_um = full_positions_um
        else:
            original_indices = np.asarray(record_indices, dtype=int)
            if original_indices.shape != (int(width),):
                raise ValueError("record_indices must match Vm columns.")
            positions_um = full_positions_um[original_indices]
        selected = np.asarray(
            target.columns(
                positions_um=positions_um,
                original_indices=original_indices,
            ),
            dtype=int,
        )
        key = tuple(int(value) for value in selected)
        group = grouped.setdefault(
            key,
            {
                "rows": [],
                "input_indices": [],
                "selected": selected,
            },
        )
        group["rows"].append(row)
        group["input_indices"].append(int(cohort.input_indices[row]))
    return tuple(grouped.values())


def _dense_activation_window(
    vm: np.ndarray,
    *,
    rows: np.ndarray,
    time_slice: slice,
    selected: np.ndarray,
) -> np.ndarray:
    all_rows = rows.shape == (vm.shape[0],) and np.array_equal(
        rows,
        np.arange(vm.shape[0]),
    )
    all_columns = selected.shape == (vm.shape[2],) and np.array_equal(
        selected,
        np.arange(vm.shape[2]),
    )
    column_index: Any = slice(None) if all_columns else selected
    if all_rows:
        return np.asarray(vm[:, time_slice, column_index])
    return np.asarray(vm[rows][:, time_slice, :][:, :, column_index])


def _recording_policy(options: dict[str, Any]) -> Any:
    if options["recording"] == "observer_only":
        return axs.Recording.none()
    if options["recording"] == "full_vm":
        return axs.Recording.voltage()
    if options["spatial_recording"] == "center":
        return axs.Recording.center(axs.signals.Vm)
    if options["spatial_recording"] == "indices":
        nx = max(int(options["nx"]), 3)
        return axs.Recording.indices((0, nx // 2, nx - 1), axs.signals.Vm)
    return axs.Recording.probes(axs.signals.Vm, count=8)


def _batch_options(options: dict[str, Any]) -> Any:
    policy = str(options.get("time_chunk_policy", "default"))
    time_chunk_steps = options["time_chunk_steps"]
    if policy == "default":
        return None
    if policy == "unchunked":
        return axs.BatchOptions(time_chunk_steps=None)
    if policy != "explicit":
        raise ValueError(f"unsupported time_chunk_policy: {policy!r}")
    if time_chunk_steps is None:
        raise ValueError("explicit time_chunk_policy requires time_chunk_steps.")
    return axs.BatchOptions(time_chunk_steps=int(time_chunk_steps))


def _execution_policy(options: dict[str, Any]) -> Any:
    precision = (
        axs.PrecisionPolicy.float64()
        if options["precision"] == "fp64"
        else axs.PrecisionPolicy.float32()
    )
    if options["platform"] == "cpu":
        device = axs.Device.cpu()
    elif options["platform"] == "gpu":
        device = axs.Device.gpu(0)
    else:
        device = axs.Device.auto()
    solver_policy = _solver_policy(options)
    return axs.ExecutionPolicy(
        runtime=axs.runtime.jax,
        device=device,
        precision=precision,
        solvers=solver_policy,
    )


def _solver_policy(options: dict[str, Any]) -> Any:
    solver = str(options.get("double_cable_block_solver", "auto"))
    platform = str(options.get("platform", "auto"))
    if platform == "cpu":
        double_cable_solver = axs.runtime.jax.cpu.DoubleCableSolver
    elif platform == "gpu":
        double_cable_solver = axs.runtime.jax.gpu.DoubleCableSolver
    else:
        double_cable_solver = axs.runtime.jax.DoubleCableSolver

    if solver in {"auto", "pcr_adaptive"}:
        double_cable = double_cable_solver.auto()
    elif solver == "thomas":
        double_cable = double_cable_solver.thomas()
    elif solver == "pcr":
        double_cable = axs.runtime.jax.gpu.DoubleCableSolver.pcr()
    elif solver == "pcr_soa":
        double_cable = axs.runtime.jax.gpu.DoubleCableSolver.pcr_soa()
    elif solver == "tiled_thomas":
        if platform != "gpu":
            raise ValueError("tiled_thomas double-cable solver requires --platform gpu.")
        block_b = options.get("tiled_thomas_block_b")
        double_cable = axs.runtime.jax.gpu.DoubleCableSolver.tiled_thomas(
            block_b=None if block_b in (None, "") else int(block_b),
        )
    else:
        raise ValueError(f"unsupported double-cable solver policy: {solver!r}")
    return axs.SolverPolicy(double_cable=double_cable)


def _row_cable(options: dict[str, Any], row: int) -> str:
    if options["population"] == "mixed_models":
        return "double_cable" if row % 2 else "single_cable"
    return str(options["cable"])


def _prepare_output(script_name: str, options: dict[str, Any]) -> Path | None:
    output = Path(options["output"])
    selected_case = case_name(script_name, options)
    if options["case_filter"] and options["case_filter"] not in selected_case:
        print("No cases selected by --case-filter.")
        return None
    if options["resume"] and (output / "results.csv").exists():
        print(f"resume: existing results found, skipping {selected_case}")
        return None
    output.mkdir(parents=True, exist_ok=True)
    (output / "artifacts").mkdir(exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    write_cases_csv(output, script_name, options)
    return output


def _validate_real_run(script_name: str, options: dict[str, Any]) -> None:
    if options["source"] != "point_source_axonscope":
        raise SystemExit(
            f"{script_name} real runs currently support --source point_source_axonscope. "
            "NRV nerve baselines stay in benchmark/baselines/ until their adapter is defined."
        )
    if options["platform"] == "nrv":
        raise SystemExit(
            f"{script_name} real runs do not execute NRV yet; use --dry-run for NRV case "
            "validation until benchmark/baselines/nrv is implemented."
        )
    if int(options["n_axons"]) < 1:
        raise SystemExit("--n-axons must be >= 1.")
    if int(options["repeats"]) < 1:
        raise SystemExit("--repeats must be >= 1.")
    if int(options["warmups"]) < 0:
        raise SystemExit("--warmups must be >= 0.")
    if float(options["amplitude_max"]) < float(options["amplitude_min"]):
        raise SystemExit("--amplitude-max must be >= --amplitude-min.")


def _append_activation_rows(
    rows: list[dict[str, Any]],
    *,
    script: str,
    selected_case: str,
    phase: str,
    repeat: int,
    curve: str,
    iteration: int,
    amplitudes: np.ndarray,
    activated: np.ndarray,
    row_meta: tuple[dict[str, Any], ...],
) -> None:
    fraction = _activation_fraction(activated)
    for row, meta in enumerate(row_meta):
        rows.append(
            {
                "script": script,
                "case_name": selected_case,
                "phase": phase,
                "repeat": repeat,
                "curve": curve,
                "iteration": iteration,
                "amplitude_uA": float(amplitudes[row]),
                "row": row,
                "family": meta["family"],
                "diameter_um": meta["diameter_um"],
                "activated": bool(activated[row]),
                "lower_uA": "",
                "upper_uA": "",
                "threshold_uA": "",
                "status": "",
                "activation_fraction": fraction,
            }
        )


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    output: Path,
    script_name: str,
    selected_case: str,
    options: dict[str, Any],
) -> None:
    manifest = {
        "script": script_name,
        "case_name": selected_case,
        "options": options,
        "outputs": {
            "environment": "environment.json",
            "events": "events.jsonl",
            "timing_summary": "summary.csv",
            "memory_summary": "memory_summary.csv",
            "cases": "cases.csv",
            "results": "results.csv",
            "curve_summary": "curve_summary.csv",
            "artifacts": "artifacts/",
            "plots": "plots/",
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _activation_fraction(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=bool)
    return float(np.mean(values)) if values.size else 0.0


def _fiber_length_um(options: dict[str, Any]) -> float:
    return max(300.0, float(options["nx"]) * 20.0)


def _stim_start_ms(options: dict[str, Any]) -> float:
    return min(0.2, float(options["tsim"]) * 0.25)


def _pulse_width_ms(options: dict[str, Any]) -> float:
    return min(0.1, max(0.02, float(options["tsim"]) * 0.05))
