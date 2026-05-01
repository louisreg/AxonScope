import importlib

import matplotlib.pyplot as plt


def test_basic_stimulus_example_runs(monkeypatch):
    module = importlib.import_module("examples.basic.stimulus_demo")
    monkeypatch.setattr(plt, "show", lambda: None)
    module.main()


def test_basic_point_source_example_runs(monkeypatch):
    module = importlib.import_module("examples.basic.point_source_electrode_demo")
    monkeypatch.setattr(plt, "show", lambda: None)
    module.main()


def test_solver_examples_are_importable():
    intracellular = importlib.import_module("examples.basic.intracellular_solver_demo")
    extracellular = importlib.import_module("examples.basic.mrg_extracellular_demo")
    assert callable(intracellular.main)
    assert callable(extracellular.main)
