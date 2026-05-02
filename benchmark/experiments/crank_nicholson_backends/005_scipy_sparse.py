from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_scipy_sparse


class ScipySparseBenchmark(CNRuntimeBenchmark):
    label = "scipy_sparse"

    def build_runner(self, problem):
        return build_scipy_sparse(problem)


if __name__ == "__main__":
    ScipySparseBenchmark().run()
