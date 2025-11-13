import threading
from functools import wraps
from pyinstrument import Profiler
from pyinstrument.renderers import JSONRenderer
import json
import pandas as pd
from tabulate import tabulate
import logging
import atexit
from timeit import timeit


def minibench(func, *args, n_iter=10):
    """
    Benchmark a Python function with given arguments.

    This function executes `func(*args)` once to capture its return value,
    then measures the average execution time over multiple iterations
    using the built-in `timeit` module.

    Parameters
    ----------
    func : callable
        The function to benchmark.
    *args : any
        Positional arguments to pass to the function.
    n_iter : int, optional
        Number of iterations for timing (default is 10).

    Returns
    -------
    result : any
        The output of a single call to the function.
    avg_time : float
        The average execution time per call (in seconds).
    """
    # Run once to get the function's result
    result = func(*args)

    # Measure average execution time
    stmt = f"{func.__name__}(*args)"
    t = timeit(
        stmt,
        number=n_iter,
        globals={'func': func, 'args': args, func.__name__: func}
    )

    avg_time = t / n_iter
    print(f"{func.__name__:<15} {avg_time:.8f} s / call ({n_iter} iterations)")
    return result, avg_time

"""
Benchmarking Singleton Class

This module provides a singleton `Benchmark` class for profiling Python functions 
with optional levels of detail. It uses PyInstrument for performance measurement 
and allows filtering to only decorated functions. Results can be printed as a 
formatted table or logged via Python's logging module.

Key Features:
- Singleton pattern ensures a single shared instance across the program.
- Decorator-based benchmarking with configurable levels.
- Tracks function call counts and execution time.
- Supports automatic printing and logging of benchmark results at script exit.
- Outputs results as a Pandas DataFrame with:
    - Function name
    - Total execution time
    - Call count
    - Average time per call
- Recursive aggregation of nested function calls to report only decorated functions.

Usage Example:
    benchmark = Benchmark()  # singleton instance

    @benchmark.benchmark(level=1)
    def step1(data):
        return [x * 2 for x in data]

    @benchmark.benchmark(level=2)
    def step2(data):
        return sum(data)

    @benchmark.benchmark(level=1)
    def process_data(data):
        step1(data)
        step1(data)
        return step2(data)

    if __name__ == "__main__":
        benchmark.enable(level=2, auto_print=True, log=True)
        main_task()
"""

class Benchmark:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._decorated_functions = {}
                    cls._instance._profiler = None
                    cls._instance._profiler_running = False
                    cls._instance._enabled_level = 0
                    cls._instance._auto_print = False
                    cls._instance._logger = None
                    cls._instance._df = None
        return cls._instance

    def benchmark(self, level=1):
        def decorator(func):
            if self._enabled_level >= level:
                self._decorated_functions[func.__name__] = {"calls": 0, "level": level}

            @wraps(func)
            def wrapper(*args, **kwargs):
                if self._enabled_level >= level:
                    if self._profiler_running:
                        with self._lock:
                            if func.__name__ not in self._decorated_functions.keys():
                                self._decorated_functions[func.__name__] = {"calls": 1, "level": level}
                            else:
                                self._decorated_functions[func.__name__]["calls"] += 1
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def enable(self, level=1, auto_print=True, log=False):
        self._enabled_level = level
        self._auto_print = auto_print
        self._profiler = Profiler()
        self._profiler.start()
        self._profiler_running = True
        self._df = None

        if log:
            self._logger = logging.getLogger("benchmark")
            self._logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        atexit.register(self._finish)

    def stop(self):
        if self._profiler and self._profiler_running:
            self._profiler.stop()
            self._profiler_running = False
            self._df = self.get_filtered_output()
            if self._auto_print:
                self.print()
        self._profiler = None

    def _finish(self):
        try:
            self.stop()
            
            if self._logger:
                self._logger.info("\n" + tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=".6f"))
        except Exception:
            pass  # Ignore exceptions at exit

    def get_filtered_output(self) -> pd.DataFrame:
        if self._df is not None:
            return(self._df)
        cols = ["Function", "Total Time (s)", "Calls", "Time/Call (s)"]
        if not self._profiler or self._profiler_running:
            # Profiler n'a pas encore collecté de données
            return pd.DataFrame(columns=cols)

        try:
            json_output = self._profiler.output(JSONRenderer(show_all=True, timeline=False))
            data = json.loads(json_output)
        except Exception:
            return pd.DataFrame(columns=cols)

        results = {}
        self._aggregate_decorated_frames(data.get("root_frame", {}), results)

        # Merge call counts
        for func_name, stats in self._decorated_functions.items():
            if func_name in results:
                results[func_name]["calls"] = stats["calls"]

        rows = []
        for func_name, stats in results.items():
            total_time = stats["time"]
            calls = stats["calls"]
            per_call = total_time / calls if calls > 0 else 0
            rows.append({
                "Function": func_name,
                "Total Time (s)": total_time,
                "Calls": calls,
                "Time/Call (s)": per_call,
            })

        return pd.DataFrame(rows).sort_values("Function").reset_index(drop=True)

    def _aggregate_decorated_frames(self, frame, results):
        func_name = frame.get("function")
        if func_name in self._decorated_functions:
            if func_name not in results:
                results[func_name] = {"time": 0.0, "calls": 0}
            results[func_name]["time"] += frame.get("time", 0.0)

        for child in frame.get("children", []):
            self._aggregate_decorated_frames(child, results)

    def print(self):
        if self._df is None: 
            self._df = self.get_filtered_output()
        print(tabulate(
            self._df,
            headers="keys",
            tablefmt="fancy_grid",
            floatfmt=".6f"
        ))
