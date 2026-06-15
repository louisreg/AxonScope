from __future__ import annotations

import axonscope as axs
from benchmark.hotpaths.catalog import HOTPATH_PRESETS, HOTPATH_WORKLOADS
from benchmark.hotpaths.run import (
    build_simulation,
    build_simulations,
    main,
    planned_runs,
    resolve_sizes,
)


def test_hotpath_catalog_lists_phase25_workloads():
    assert set(HOTPATH_WORKLOADS) == {
        "footprint_reuse_sweep",
        "intracellular_only",
        "hotpath_matrix",
        "observer_only",
        "point_source_extracellular",
        "realistic_mixed_population",
    }
    assert HOTPATH_PRESETS["smoke"] == (5,)
    assert HOTPATH_PRESETS["scale"] == (5, 50, 500)


def test_hotpath_runner_prints_list(capsys):
    main(["--list"])

    out = capsys.readouterr().out
    assert "intracellular_only" in out
    assert "point_source_extracellular" in out
    assert "scale" in out


def test_hotpath_runner_dry_run_expands_all_workloads(capsys):
    main(["--workload", "all", "--sizes", "2", "3", "--dry-run"])

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "intracellular_only size=2",
        "intracellular_only size=3",
        "point_source_extracellular size=2",
        "point_source_extracellular size=3",
        "footprint_reuse_sweep size=2",
        "footprint_reuse_sweep size=3",
        "observer_only size=2",
        "observer_only size=3",
        "realistic_mixed_population size=2",
        "realistic_mixed_population size=3",
        "hotpath_matrix size=2",
        "hotpath_matrix size=3",
    ]


def test_hotpath_runner_dry_run_keeps_registry_order(capsys):
    main(["--workload", "all", "--sizes", "1", "--dry-run"])

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "intracellular_only size=1",
        "point_source_extracellular size=1",
        "footprint_reuse_sweep size=1",
        "observer_only size=1",
        "realistic_mixed_population size=1",
        "hotpath_matrix size=1",
    ]


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
        "footprint_reuse_sweep",
        "observer_only",
        "realistic_mixed_population",
        "hotpath_matrix",
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
