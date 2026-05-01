from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_scipy_banded


class ScipyBandedBenchmark(CNRuntimeBenchmark):
    label = "solve_banded"

    def build_runner(self, problem):
        return build_scipy_banded(problem)


if __name__ == "__main__":
    ScipyBandedBenchmark().run()
