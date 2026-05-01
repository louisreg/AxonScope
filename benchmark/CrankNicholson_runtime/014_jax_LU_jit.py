from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_lu_jit


class JaxLUJitBenchmark(CNRuntimeBenchmark):
    label = "jax_lu_jit"
    Nx_values = [11, 21, 51, 101, 201]

    def build_runner(self, problem):
        return build_jax_lu_jit(problem)


if __name__ == "__main__":
    JaxLUJitBenchmark().run()
