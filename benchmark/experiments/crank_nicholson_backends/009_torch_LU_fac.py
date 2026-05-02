from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_torch_lu


class TorchLUBenchmark(CNRuntimeBenchmark):
    label = "torch_lu_factorized"
    Nx_values = [11, 21, 51, 101, 201]

    def build_runner(self, problem):
        return build_torch_lu(problem)


if __name__ == "__main__":
    TorchLUBenchmark().run()
