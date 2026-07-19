from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import axonscope as axs
from benchmark.analysis.run_pool_detail import write_run_pool_detail
from benchmark.protocols import recruitment_amplitude_batch
from axonscope.dispatcher import build_dispatch_plan
from axonscope.preparation.cohort import PreparedCohort


def test_p14_realistic_defaults_match_reference_workload() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        ["--workload", "p14_realistic", "--cable", "double"]
    )

    recruitment_amplitude_batch._resolve_workload_args(args)

    assert args.axon_count == 196
    assert args.duration_ms == 3.0
    assert args.dt_ms == 0.001
    assert args.amplitudes == recruitment_amplitude_batch.P14_REALISTIC_AMPLITUDES_UA
    assert len(args.amplitudes) == 21


def test_cable_counts_preserve_requested_population_size() -> None:
    assert recruitment_amplitude_batch._cable_counts("single", 196) == (196, 0)
    assert recruitment_amplitude_batch._cable_counts("double", 196) == (0, 196)
    assert recruitment_amplitude_batch._cable_counts("mixed", 197) == (98, 99)


def test_realistic_double_population_shares_exact_mrg_templates() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--workload",
            "p14_realistic",
            "--cable",
            "double",
            "--axon-count",
            "12",
        ]
    )
    recruitment_amplitude_batch._resolve_workload_args(args)

    pool, *_ = recruitment_amplitude_batch._build_workload(args)

    assert len({id(row.axon) for row in pool}) <= 3


def test_realistic_single_population_shares_canonical_diameter_templates() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--workload",
            "p14_realistic",
            "--cable",
            "single",
            "--axon-count",
            "100",
        ]
    )
    recruitment_amplitude_batch._resolve_workload_args(args)

    pool, *_ = recruitment_amplitude_batch._build_workload(args)

    # Canonicalization yields at most 61 hundredth-scale values up to 1 um and
    # two tenth-scale values above it over this benchmark's [0.4, 1.2] range.
    assert len({id(row.axon) for row in pool}) <= 63


def test_realistic_population_benchmark_can_retain_distinct_ab_control() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--workload",
            "p14_realistic",
            "--cable",
            "double",
            "--axon-count",
            "6",
            "--axon-template-policy",
            "distinct",
        ]
    )
    recruitment_amplitude_batch._resolve_workload_args(args)

    pool, *_ = recruitment_amplitude_batch._build_workload(args)

    assert len({id(row.axon) for row in pool}) == len(pool)


def test_mrg_template_population_realizes_requested_diameter_shift_pairs() -> None:
    rng = np.random.default_rng(7)

    diameters, shifts = recruitment_amplitude_batch._mrg_population_templates(
        rng,
        row_count=64,
        template_count=32,
    )

    assert len(set(zip(diameters, shifts, strict=True))) == 32
    assert set(diameters) == set(recruitment_amplitude_batch.P14_MRG_DIAMETERS_UM)
    assert np.count_nonzero(shifts) > 0


def test_mrg_template_population_caps_templates_at_row_count() -> None:
    rng = np.random.default_rng(7)

    diameters, shifts = recruitment_amplitude_batch._mrg_population_templates(
        rng,
        row_count=8,
        template_count=32,
    )

    assert len(set(zip(diameters, shifts, strict=True))) == 8


def test_translated_mrg_workload_shares_structural_materialization() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--workload",
            "p14_realistic",
            "--cable",
            "double",
            "--axon-count",
            "12",
            "--mrg-template-count",
            "6",
            "--mrg-shift-semantics",
            "translation",
        ]
    )
    recruitment_amplitude_batch._resolve_workload_args(args)

    pool, *_ = recruitment_amplitude_batch._build_workload(args)
    plan = build_dispatch_plan(pool)
    cohort = PreparedCohort.from_dispatch_group(plan.groups[0])

    assert len({id(row.axon) for row in pool}) == 6
    assert cohort.materialized_axons.template_count == 3
    assert cohort.materialized_axons.translated_row_count > 0


def test_translated_mrg_workload_samples_row_specific_footprints() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--workload",
            "p14_realistic",
            "--cable",
            "double",
            "--axon-count",
            "12",
            "--mrg-template-count",
            "6",
            "--mrg-shift-semantics",
            "translation",
        ]
    )
    recruitment_amplitude_batch._resolve_workload_args(args)

    pool, *_ = recruitment_amplitude_batch._build_workload(args)

    for row in pool:
        footprint = row.extracellular_stimulation.drives[0].footprint
        positions_um = row.axon.layout.position_values(unit="micrometer")
        metadata = footprint.metadata
        electrode = axs.analytical.PointSourceElectrode(
            x=float(metadata["electrode_x_um"]) * axs.um,
            y=float(metadata["electrode_y_um"]) * axs.um,
            z=float(metadata["electrode_z_um"]) * axs.um,
            min_distance=5.0 * axs.um,
        )
        expected = electrode.footprint_for_axon(
            positions_um * 1e-6,
            sigma_S_m=float(metadata["sigma_S_m"]),
            axon_y_um=float(metadata["axon_y_um"]),
            axon_z_um=float(metadata["axon_z_um"]),
        )

        np.testing.assert_array_equal(footprint.positions_um, positions_um)
        np.testing.assert_allclose(
            footprint.values_for_axon(),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


def test_p14_dry_run_records_workload_shape(tmp_path: Path) -> None:
    assert (
        recruitment_amplitude_batch.main(
            [
                "--workload",
                "p14_realistic",
                "--cable",
                "single",
                "--policies",
                "1,2,full",
                "--dry-run",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["workload"] == "p14_realistic"
    assert manifest["cable"] == "single"
    assert manifest["profile_scope"] == "run"
    assert manifest["n_axons"] == 196
    assert manifest["amplitudes_uA"] == list(
        recruitment_amplitude_batch.P14_REALISTIC_AMPLITUDES_UA
    )
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert [row["policy"] for row in rows] == ["1", "2", "full"]
    assert {row["amplitude_count"] for row in rows} == {"21"}


def test_main_reuses_one_source_workload_across_cold_and_warm_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = object()
    build_calls: list[str] = []
    run_calls: list[tuple[object, float, str]] = []

    def fake_build_source(args, output, policy):
        del args, output
        build_calls.append(policy.label)
        return workload, 12.5

    def fake_run_one(
        args,
        output,
        policy,
        *,
        workload,
        source_population_build_ms,
        phase,
        repeat,
    ):
        del output
        run_calls.append((workload, source_population_build_ms, phase))
        return (
            {
                "policy": policy.label,
                "batch_amplitudes": True,
                "amplitude_batch_size": "full",
                "phase": phase,
                "repeat": repeat,
                "platform": args.platform,
                "workload": args.workload,
                "cable": args.cable,
                "drive_count": args.drive_count,
                "fibers_per_family": args.fibers_per_family,
                "n_axons": args.axon_count,
                "amplitude_count": len(args.amplitudes),
                "wall_ms": 1.0,
                "source_population_build_ms": source_population_build_ms,
                "one_shot_wall_ms": 13.5 if phase == "cold" else "",
                "failed": False,
                "error": "",
            },
            np.asarray([0], dtype=int),
        )

    monkeypatch.setattr(
        recruitment_amplitude_batch,
        "_build_source_workload",
        fake_build_source,
    )
    monkeypatch.setattr(recruitment_amplitude_batch, "_run_one", fake_run_one)

    assert (
        recruitment_amplitude_batch.main(
            [
                "--policies",
                "full",
                "--warmups",
                "1",
                "--repeats",
                "1",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert build_calls == ["full"]
    assert run_calls == [
        (workload, 12.5, "cold"),
        (workload, 12.5, "warmup"),
        (workload, 12.5, "warm"),
    ]


def test_profile_scope_can_target_only_the_sweep() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        ["--profile", "--profile-scope", "sweep"]
    )

    assert args.profile is True
    assert args.profile_scope == "sweep"


def test_profile_scope_can_target_only_run_pool() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        ["--profile", "--profile-scope", "run_pool"]
    )

    assert args.profile is True
    assert args.profile_scope == "run_pool"


def test_multi_drive_workload_keeps_one_static_and_one_variable_drive() -> None:
    args = recruitment_amplitude_batch.build_parser().parse_args(
        [
            "--workload",
            "legacy",
            "--cable",
            "single",
            "--axon-count",
            "1",
            "--drive-count",
            "2",
        ]
    )
    recruitment_amplitude_batch._resolve_workload_args(args)

    pool, update, *_ = recruitment_amplitude_batch._build_workload(args)

    stimulation = pool[0].extracellular_stimulation
    assert tuple(str(value) for value in stimulation.names) == ("variable", "static")
    assert update.drive_id == stimulation.names[0]
    assert not np.array_equal(
        stimulation.drives[0].footprint.values_V_per_A,
        stimulation.drives[1].footprint.values_V_per_A,
    )


def test_multi_drive_gpu_route_validation_rejects_non_triton_double(
    tmp_path: Path,
) -> None:
    events = [
        {
            "name": "inputs.extracellular",
            "metadata": {
                "extracellular_capability_cable": "double-cable",
                "extracellular_format": "factorized_footprint",
                "dense_vstim_avoided": True,
                "nstim": 2,
            },
        },
        {
            "name": "inputs.numeric_axis",
            "metadata": {
                "mode": "double",
                "extracellular_waveform_drive_count": 2,
            },
        },
        {
            "name": "dispatch.group.total",
            "metadata": {
                "mode": "double",
                "prepared_input_contract_extracellular_format": (
                    "factorized_footprint"
                ),
                "execution_policy_double_cable_block_solver": "jax_scan_thomas",
            },
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="production Triton route"):
        recruitment_amplitude_batch._validate_multi_drive_routes(
            tmp_path,
            cable="double",
            platform="gpu",
        )


def test_run_pool_detail_splits_single_and_double_group_timings(
    tmp_path: Path,
) -> None:
    events = [
        {"event_id": 1, "name": "simulation.run_pool", "duration_ms": 100.0},
        {
            "event_id": 2,
            "parent_event_id": 1,
            "name": "dispatch.group.total",
            "duration_ms": 80.0,
            "metadata": {"mode": "double"},
        },
        {
            "event_id": 3,
            "parent_event_id": 2,
            "name": "kernel.enqueue",
            "duration_ms": 20.0,
        },
        {
            "event_id": 4,
            "parent_event_id": 3,
            "name": "kernel.dispatch_jax",
            "duration_ms": 20.0,
        },
        {
            "event_id": 5,
            "parent_event_id": 2,
            "name": "kernel.wait",
            "duration_ms": 5.0,
        },
        {
            "event_id": 6,
            "parent_event_id": 1,
            "name": "dispatch.group.total",
            "duration_ms": 15.0,
            "metadata": {"mode": "single"},
        },
        {
            "event_id": 7,
            "parent_event_id": 6,
            "name": "kernel.enqueue",
            "duration_ms": 2.0,
        },
        {
            "event_id": 8,
            "parent_event_id": 7,
            "name": "kernel.dispatch_jax",
            "duration_ms": 2.0,
        },
        {
            "event_id": 9,
            "parent_event_id": 6,
            "name": "kernel.wait",
            "duration_ms": 1.0,
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    write_run_pool_detail(
        tmp_path,
        amplitudes=(300.0,),
    )

    rows = {
        row["mode"]: row
        for row in csv.DictReader((tmp_path / "run_pool_detail.csv").open())
    }
    assert rows["all"]["amplitudes_uA"] == "300"
    assert float(rows["all"]["kernel_solver_ms"]) == 28.0
    assert float(rows["double"]["group_ms"]) == 80.0
    assert float(rows["double"]["kernel_solver_ms"]) == 25.0
    assert float(rows["double"]["kernel_wait_pct_solver"]) == 20.0
    assert float(rows["single"]["kernel_solver_ms"]) == 3.0


def test_run_pool_detail_maps_numeric_chunk_to_all_amplitudes(tmp_path: Path) -> None:
    events = [
        {
            "event_id": 1,
            "name": "protocol.sweep.amplitude_chunk",
            "duration_ms": 20.0,
            "metadata": {"value_count": 3},
        },
        {
            "event_id": 2,
            "parent_event_id": 1,
            "name": "simulation.run_pool",
            "duration_ms": 18.0,
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    write_run_pool_detail(tmp_path, amplitudes=(0.0, 20.0, 80.0))

    rows = list(csv.DictReader((tmp_path / "run_pool_detail.csv").open()))
    assert rows[0]["amplitude_count"] == "3"
    assert rows[0]["amplitudes_uA"] == "0 20 80"
