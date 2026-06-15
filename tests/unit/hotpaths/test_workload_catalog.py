from __future__ import annotations

import axonscope as axs
from benchmark.hotpaths.catalog import HOTPATH_PRESETS, HOTPATH_WORKLOADS
from benchmark.hotpaths.run import (
    _describe_simulations,
    _simulation_labels,
    build_simulation,
    build_simulations,
    configure_jax_compile_logging,
    main,
    planned_runs,
    resolve_sizes,
)


def test_hotpath_catalog_lists_phase25_workloads():
    assert set(HOTPATH_WORKLOADS) == {
        "footprint_reuse_sweep",
        "intracellular_only",
        "double_cable_extracellular",
        "double_cable_observer",
        "hotpath_matrix",
        "observer_only",
        "path_comparison_matrix",
        "point_source_extracellular",
        "realistic_mixed_population",
        "solver_only_precomputed",
        "typed_footprint_drive_matrix",
    }
    assert HOTPATH_PRESETS["smoke"] == (5,)
    assert HOTPATH_PRESETS["scale"] == (5, 50, 500)


def test_hotpath_runner_prints_list(capsys):
    main(["--list"])

    out = capsys.readouterr().out
    assert "intracellular_only" in out
    assert "point_source_extracellular" in out
    assert "double_cable_extracellular" in out
    assert "scale" in out


def test_hotpath_runner_dry_run_expands_all_workloads(capsys):
    main(["--workload", "all", "--sizes", "2", "3", "--dry-run"])

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "intracellular_only size=2",
        "intracellular_only size=3",
        "point_source_extracellular size=2",
        "point_source_extracellular size=3",
        "double_cable_extracellular size=2",
        "double_cable_extracellular size=3",
        "double_cable_observer size=2",
        "double_cable_observer size=3",
        "footprint_reuse_sweep size=2",
        "footprint_reuse_sweep size=3",
        "solver_only_precomputed size=2",
        "solver_only_precomputed size=3",
        "typed_footprint_drive_matrix size=2",
        "typed_footprint_drive_matrix size=3",
        "observer_only size=2",
        "observer_only size=3",
        "realistic_mixed_population size=2",
        "realistic_mixed_population size=3",
        "hotpath_matrix size=2",
        "hotpath_matrix size=3",
        "path_comparison_matrix size=2",
        "path_comparison_matrix size=3",
    ]


def test_hotpath_runner_dry_run_keeps_registry_order(capsys):
    main(["--workload", "all", "--sizes", "1", "--dry-run"])

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "intracellular_only size=1",
        "point_source_extracellular size=1",
        "double_cable_extracellular size=1",
        "double_cable_observer size=1",
        "footprint_reuse_sweep size=1",
        "solver_only_precomputed size=1",
        "typed_footprint_drive_matrix size=1",
        "observer_only size=1",
        "realistic_mixed_population size=1",
        "hotpath_matrix size=1",
        "path_comparison_matrix size=1",
    ]


def test_hotpath_runner_accepts_jax_compile_logging_flag_in_dry_run(capsys):
    main(
        [
            "--workload",
            "intracellular_only",
            "--sizes",
            "1",
            "--dry-run",
            "--jax-log-compiles",
        ]
    )

    assert capsys.readouterr().out.splitlines() == ["intracellular_only size=1"]


def test_configure_jax_compile_logging_disabled_is_noop():
    assert configure_jax_compile_logging(False) == {"enabled": False}


def test_hotpath_runner_accepts_time_chunk_steps_in_dry_run(capsys):
    main(
        [
            "--workload",
            "observer_only",
            "--sizes",
            "1",
            "--dry-run",
            "--time-chunk-steps",
            "2",
        ]
    )

    assert capsys.readouterr().out.splitlines() == ["observer_only size=1"]


def test_hotpath_runner_accepts_double_cable_pcr_solver_in_dry_run(capsys):
    main(
        [
            "--workload",
            "double_cable_extracellular",
            "--sizes",
            "1",
            "--dry-run",
            "--double-cable-block-solver",
            "pcr",
        ]
    )

    assert capsys.readouterr().out.splitlines() == ["double_cable_extracellular size=1"]


def test_hotpath_runner_accepts_double_cable_auto_solver_in_dry_run(capsys):
    main(
        [
            "--workload",
            "double_cable_extracellular",
            "--sizes",
            "1",
            "--dry-run",
            "--double-cable-block-solver",
            "auto",
        ]
    )

    assert capsys.readouterr().out.splitlines() == ["double_cable_extracellular size=1"]


def test_hotpath_size_and_run_resolution():
    assert resolve_sizes("smoke", None) == (5,)
    assert resolve_sizes("scale", None) == (5, 50, 500)
    assert resolve_sizes("smoke", [2, 4]) == (2, 4)
    single = planned_runs("intracellular_only", (2,))
    assert len(single) == 1
    assert single[0].workload == "intracellular_only"
    assert single[0].size == 2
    assert [run.workload for run in planned_runs("all", (1,))] == [
        "intracellular_only",
        "point_source_extracellular",
        "double_cable_extracellular",
        "footprint_reuse_sweep",
        "solver_only_precomputed",
        "typed_footprint_drive_matrix",
        "observer_only",
        "realistic_mixed_population",
        "hotpath_matrix",
        "path_comparison_matrix",
    ]


def test_hotpath_build_simulation_uses_public_root_object():
    simulation = build_simulation(
        "point_source_extracellular",
        size=2,
        compartments=5,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
    )

    assert isinstance(simulation, axs.AxonSimulation)
    assert simulation.is_population
    assert len(simulation.axons) == 2
    assert all(instance.extracellular_context is not None for instance in simulation.axons)


def test_hotpath_double_cable_extracellular_uses_mrg_double_cable_rows():
    simulation = build_simulation(
        "double_cable_extracellular",
        size=2,
        compartments=51,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
    )

    assert isinstance(simulation, axs.AxonSimulation)
    assert simulation.is_population
    assert len(simulation.axons) == 2
    assert {type(instance.axon).__name__ for instance in simulation.axons} == {"MRG"}
    assert {
        instance.axon.resolved_formulation for instance in simulation.axons
    } == {"double-cable"}
    assert all(instance.extracellular_context is not None for instance in simulation.axons)


def test_hotpath_double_cable_observer_uses_mrg_observer_only_rows():
    simulation = build_simulation(
        "double_cable_observer",
        size=2,
        compartments=51,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
    )

    assert isinstance(simulation, axs.AxonSimulation)
    assert simulation.is_population
    assert len(simulation.axons) == 2
    assert not simulation.recording.voltage
    assert simulation.observers is not None
    assert {type(instance.axon).__name__ for instance in simulation.axons} == {"MRG"}
    assert {
        instance.axon.resolved_formulation for instance in simulation.axons
    } == {"double-cable"}
    assert all(instance.extracellular_context is not None for instance in simulation.axons)


def test_hotpath_footprint_reuse_workload_builds_repeated_simulations():
    simulations = build_simulations(
        "footprint_reuse_sweep",
        size=2,
        compartments=5,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
        sweep_repeats=3,
    )

    assert len(simulations) == 3
    assert all(isinstance(simulation, axs.AxonSimulation) for simulation in simulations)
    assert all(simulation.estimate().metadata["context_count"] == 2 for simulation in simulations)


def test_hotpath_realistic_population_mixes_models_and_geometry():
    simulation = build_simulation(
        "realistic_mixed_population",
        size=6,
        compartments=5,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
    )

    model_names = {type(instance.axon).__name__ for instance in simulation.axons}
    diameters = {instance.axon.diameter for instance in simulation.axons}
    compartment_counts = {instance.axon.n_compartments for instance in simulation.axons}

    assert simulation.is_population
    assert model_names == {"HodgkinHuxley", "RattayAberham"}
    assert len(diameters) > 1
    assert len(compartment_counts) > 1
    assert any(instance.extracellular_context is not None for instance in simulation.axons)


def test_hotpath_matrix_builds_labeled_coverage_scenarios():
    simulations = build_simulations(
        "hotpath_matrix",
        size=3,
        compartments=5,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
    )

    assert len(simulations) == 5
    assert all(isinstance(simulation, axs.AxonSimulation) for simulation in simulations)
    assert [simulation.recording.spatial.value for simulation in simulations[:2]] == [
        "center",
        "probes",
    ]
    assert not simulations[2].recording.voltage
    assert simulations[2].observers is not None
    assert any(instance.extracellular_context is not None for instance in simulations[3].axons)
    assert {type(instance.axon).__name__ for instance in simulations[4].axons} == {
        "HodgkinHuxley",
        "RattayAberham",
    }


def test_path_comparison_matrix_builds_controlled_path_scenarios():
    simulations = build_simulations(
        "path_comparison_matrix",
        size=3,
        compartments=5,
        length_um=40.0,
        duration_ms=0.1,
        dt_ms=0.05,
    )

    assert len(simulations) == 10
    assert all(isinstance(simulation, axs.AxonSimulation) for simulation in simulations)
    assert [simulation.recording.spatial.value for simulation in simulations[:3]] == [
        "center",
        "probes",
        "full",
    ]
    assert not simulations[3].recording.voltage
    assert simulations[3].observers is not None
    assert all(instance.extracellular_context is None for instance in simulations[0].axons)
    assert any(instance.extracellular_context is not None for instance in simulations[4].axons)
    assert not simulations[7].recording.voltage
    assert simulations[7].observers is not None
    assert {
        instance.axon.resolved_formulation for instance in simulations[8].axons
    } == {"double-cable"}

    metadata = _describe_simulations(
        _simulation_labels("path_comparison_matrix", len(simulations)),
        simulations,
    )
    assert metadata[0]["comparison_axes"]["recording_spatial"] == "center"
    assert metadata[1]["comparison_axes"]["recording_spatial"] == "probes"
    assert metadata[3]["comparison_axes"] == {
        "path_family": "single_intracellular",
        "stimulation": "intracellular_current_clamp",
        "recording_spatial": "none",
        "recording_voltage": False,
        "observer_mode": "solver_side",
    }
    assert metadata[7]["comparison_axes"] == {
        "path_family": "single_point_source_extracellular",
        "stimulation": "analytical_point_source_extracellular",
        "recording_spatial": "none",
        "recording_voltage": False,
        "observer_mode": "solver_side",
    }
