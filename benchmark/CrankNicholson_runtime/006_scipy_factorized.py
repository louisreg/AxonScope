from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_scipy_factorized


class ScipyFactorizedBenchmark(CNRuntimeBenchmark):
    label = "scipy_sparse_factorized"

    def build_runner(self, problem):
        return build_scipy_factorized(problem)


if __name__ == "__main__":
    ScipyFactorizedBenchmark().run()
