import importlib

import matplotlib.pyplot as plt


def test_basic_stimulus_example_runs(monkeypatch):
    module = importlib.import_module("examples.basic.example_01_stimulus_waveforms")
    monkeypatch.setattr(plt, "show", lambda: None)
    module.main()


def test_basic_point_source_example_runs(monkeypatch):
    module = importlib.import_module("examples.basic.example_02_point_source_electrode")
    monkeypatch.setattr(plt, "show", lambda: None)
    module.main()


def test_solver_examples_are_importable():
    intracellular = importlib.import_module("examples.basic.example_03_intracellular_hh")
    extracellular = importlib.import_module("examples.basic.example_04_extracellular_mrg")
    basic_pool = importlib.import_module("examples.basic.example_05_pool_dispatch_basic")
    velocity_diameter = importlib.import_module("examples.basic.example_06_velocity_vs_diameter")
    threshold_diameter = importlib.import_module("examples.basic.example_07_threshold_vs_diameter")
    recruitment_population = importlib.import_module(
        "examples.basic.example_08_recruitment_curve_population"
    )
    dispatch = importlib.import_module("examples.advanced.example_01_pool_dispatch_nrv")
    layout_options = importlib.import_module("examples.advanced.example_02_layout_options")
    custom_axon = importlib.import_module("examples.advanced.example_03_custom_axon_from_scratch")
    stimulation_contexts = importlib.import_module("examples.advanced.example_04_stimulation_contexts")
    recording_options = importlib.import_module("examples.advanced.example_05_recording_options")
    activation_criterion = importlib.import_module("examples.advanced.example_06_activation_criterion")
    recruitment_curve = importlib.import_module("examples.advanced.example_07_recruitment_curve")
    root_simulation = importlib.import_module("examples.advanced.example_08_root_axon_simulation")
    axon_population = importlib.import_module("examples.advanced.example_09_axon_population")
    typed_signals = importlib.import_module("examples.advanced.example_10_typed_recording_signals")
    typed_positions = importlib.import_module("examples.advanced.example_11_typed_position_selectors")
    cable_formulation = importlib.import_module("examples.advanced.example_12_cable_formulation")
    footprint_drive = importlib.import_module("examples.advanced.example_13_extracellular_footprint_drive")
    pool_benchmark = importlib.import_module("benchmark.runtime.pool_batch_demo")
    assert callable(intracellular.main)
    assert callable(extracellular.main)
    assert callable(basic_pool.main)
    assert callable(velocity_diameter.main)
    assert callable(threshold_diameter.main)
    assert callable(recruitment_population.main)
    assert callable(dispatch.main)
    assert callable(layout_options.main)
    assert callable(custom_axon.main)
    assert callable(stimulation_contexts.main)
    assert callable(recording_options.main)
    assert callable(activation_criterion.main)
    assert callable(recruitment_curve.main)
    assert callable(root_simulation.main)
    assert callable(axon_population.main)
    assert callable(typed_signals.main)
    assert callable(typed_positions.main)
    assert callable(cable_formulation.main)
    assert callable(footprint_drive.main)
    assert callable(pool_benchmark.main)
