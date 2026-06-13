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
    velocity_batch = importlib.import_module("examples.basic.example_06_velocity_vs_diameter_batch")
    dispatch = importlib.import_module("examples.advanced.example_01_pool_dispatch_nrv")
    layout_options = importlib.import_module("examples.advanced.example_02_layout_options")
    custom_axon = importlib.import_module("examples.advanced.example_03_custom_axon_from_scratch")
    stimulation_contexts = importlib.import_module("examples.advanced.example_04_stimulation_contexts")
    recording_options = importlib.import_module("examples.advanced.example_05_recording_options")
    activation_criterion = importlib.import_module("examples.advanced.example_06_activation_criterion")
    recruitment_curve = importlib.import_module("examples.advanced.example_07_recruitment_curve")
    pool_benchmark = importlib.import_module("benchmark.runtime.pool_batch_demo")
    assert callable(intracellular.main)
    assert callable(extracellular.main)
    assert callable(basic_pool.main)
    assert callable(velocity_batch.main)
    assert callable(dispatch.main)
    assert callable(layout_options.main)
    assert callable(custom_axon.main)
    assert callable(stimulation_contexts.main)
    assert callable(recording_options.main)
    assert callable(activation_criterion.main)
    assert callable(recruitment_curve.main)
    assert callable(pool_benchmark.main)
