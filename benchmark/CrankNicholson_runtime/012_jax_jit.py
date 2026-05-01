from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_dense_jit


class JaxDenseJitBenchmark(CNRuntimeBenchmark):
    label = "jax_dense_jit"
    Nx_values = [11, 21, 51, 101, 201]

    def build_runner(self, problem):
        return build_jax_dense_jit(problem)


if __name__ == "__main__":
    JaxDenseJitBenchmark().run()
