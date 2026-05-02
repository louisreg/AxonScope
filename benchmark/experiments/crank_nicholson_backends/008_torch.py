from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_torch_dense


class TorchDenseBenchmark(CNRuntimeBenchmark):
    label = "torch_linsolve"
    Nx_values = [11, 21, 51, 101, 201]

    def build_runner(self, problem):
        return build_torch_dense(problem, compile_run=False)


if __name__ == "__main__":
    TorchDenseBenchmark().run()
