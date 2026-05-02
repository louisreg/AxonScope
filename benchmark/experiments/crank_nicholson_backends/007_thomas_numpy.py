from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_numpy_thomas


class NumpyThomasBenchmark(CNRuntimeBenchmark):
    label = "numpy_thomas_vectorized"

    def build_runner(self, problem):
        return build_numpy_thomas(problem)


if __name__ == "__main__":
    NumpyThomasBenchmark().run()
