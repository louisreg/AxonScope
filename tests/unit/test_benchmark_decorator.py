import pytest
import pandas as pd
from axonscope.benchmark import Benchmark
from time import sleep

@pytest.fixture
def benchmark_instance():
    # Reset the singleton state
    instance = Benchmark()
    instance._decorated_functions = {}
    instance._profiler = None
    instance._enabled_level = 0
    instance._auto_print = False
    instance._logger = None
    yield instance
    instance.stop()


def test_call_count_and_level(benchmark_instance):
    #bench = Benchmark()
    @benchmark_instance.benchmark(level=1)
    def func1():
        sleep(0.1)
        return 1

    benchmark_instance.enable(level=1, auto_print=False)
    func1()
    func1()
    benchmark_instance.stop()

    df = benchmark_instance.get_filtered_output()
    # Ensure we always have the expected columns even if profiler missed something
    expected_cols = ["Function", "Total Time (s)", "Calls", "Time/Call (s)"]
    for col in expected_cols:
        assert col in df.columns

    assert "func1" in df["Function"].values
    calls_func1 = df.loc[df["Function"] == "func1", "Calls"].values[0]
    assert calls_func1 == 2


def test_dataframe_structure(benchmark_instance):
    @benchmark_instance.benchmark(level=1)
    def f():
        sleep(0.1)
        return sum(range(10))

    benchmark_instance.enable(level=1, auto_print=False)
    f()
    benchmark_instance.stop()
    df = benchmark_instance.get_filtered_output()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Function", "Total Time (s)", "Calls", "Time/Call (s)"]
    assert df["Total Time (s)"].sum() > 0

def test_multiple_calls(benchmark_instance):
    @benchmark_instance.benchmark(level=1)
    def g():
        sleep(0.1)
        return sum(range(5))

    benchmark_instance.enable(level=1, auto_print=False)
    for _ in range(5):
        g()
    benchmark_instance.stop()

    df = benchmark_instance.get_filtered_output()
    calls_g = df.loc[df["Function"] == "g", "Calls"].values[0]
    assert calls_g == 5
