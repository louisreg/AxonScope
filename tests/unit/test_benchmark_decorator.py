import pytest
from axonscope.benchmark import Benchmark
from time import sleep
import timeit
import pandas as pd

@pytest.fixture
def benchmark_instance():
    instance = Benchmark()
    instance._decorated_functions = {}
    instance._profiler = None
    instance._enabled_level = 0
    instance._auto_print = False
    instance._logger = None
    yield instance
    instance.stop()


def test_benchmark_levels_and_nested_calls(benchmark_instance):
    """
    Test Benchmark decorator with multiple levels and nested calls.
    Checks:
    - simple and nested functions
    - different levels
    - call counts
    - time consistency with timeit
    - functions below enabled level are ignored
    """

    # --- Define functions with different levels ---
    @benchmark_instance.benchmark(level=0)
    def func_level0():
        sleep(0.05)
        return "L0"

    @benchmark_instance.benchmark(level=1)
    def func_level1():
        sleep(0.1)
        return "L1"

    @benchmark_instance.benchmark(level=2)
    def func_level2():
        sleep(0.02)
        return "L2"

    # Nested function example
    @benchmark_instance.benchmark(level=1)
    def inner_func():
        sleep(0.03)
        return "inner"

    @benchmark_instance.benchmark(level=1)
    def outer_func():
        results = []
        for _ in range(2):
            results.append(inner_func())
        return results

    # Enable level 1: only level 0 and 1 functions will be recorded
    benchmark_instance.enable(level=1, auto_print=False)

    # Call functions
    for _ in range(2):
        func_level0()
    for _ in range(3):
        func_level1()
    for _ in range(2):
        outer_func()
    for _ in range(2):
        func_level2()  # Should NOT be recorded because level 2 > enabled level

    benchmark_instance.stop()

    # --- DataFrame checks ---
    df = benchmark_instance.get_filtered_output()
    expected_cols = ["Function", "Total Time (s)", "Calls", "Time/Call (s)"]
    for col in expected_cols:
        assert col in df.columns

    # --- Check that only functions level <= 1 are recorded ---
    recorded_funcs = df["Function"].tolist()
    assert "func_level0" in recorded_funcs
    assert "func_level1" in recorded_funcs
    assert "inner_func" in recorded_funcs
    assert "outer_func" in recorded_funcs
    assert "func_level2" not in recorded_funcs

    # --- Call counts ---
    assert df.loc[df["Function"] == "func_level0", "Calls"].values[0] == 2
    assert df.loc[df["Function"] == "func_level1", "Calls"].values[0] == 3
    assert df.loc[df["Function"] == "inner_func", "Calls"].values[0] == 2*2  # inner called 2 times per outer_func
    assert df.loc[df["Function"] == "outer_func", "Calls"].values[0] == 2

    # --- Timeit comparison ---
    tol = 0.05  # 5% tolerance

    def plain_level0(): sleep(0.05)
    def plain_level1(): sleep(0.1)
    def plain_inner(): sleep(0.03)
    def plain_outer():
        for _ in range(2): plain_inner()

    assert abs(df.loc[df["Function"] == "func_level0", "Total Time (s)"].values[0] -
               timeit.timeit(plain_level0, number=2)) / timeit.timeit(plain_level0, number=2) < tol

    assert abs(df.loc[df["Function"] == "func_level1", "Total Time (s)"].values[0] -
               timeit.timeit(plain_level1, number=3)) / timeit.timeit(plain_level1, number=3) < tol

    assert abs(df.loc[df["Function"] == "inner_func", "Total Time (s)"].values[0] -
               timeit.timeit(plain_inner, number=4)) / timeit.timeit(plain_inner, number=4) < tol

    assert abs(df.loc[df["Function"] == "outer_func", "Total Time (s)"].values[0] -
               timeit.timeit(plain_outer, number=2)) / timeit.timeit(plain_outer, number=2) < tol
