from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_torch_dense


class TorchCompileBenchmark(CNRuntimeBenchmark):
    label = "torch_compile_linsolve"
    Nx_values = [11, 21, 51, 101, 201]

    def build_runner(self, problem):
        return build_torch_dense(problem, compile_run=True)


if __name__ == "__main__":
    TorchCompileBenchmark().run()
