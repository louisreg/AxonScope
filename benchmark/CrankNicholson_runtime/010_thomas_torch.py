from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_torch_thomas


class TorchThomasBenchmark(CNRuntimeBenchmark):
    label = "torch_thomas"
    Nx_values = [11, 21, 51, 101, 201]

    def build_runner(self, problem):
        return build_torch_thomas(problem)


if __name__ == "__main__":
    TorchThomasBenchmark().run()
