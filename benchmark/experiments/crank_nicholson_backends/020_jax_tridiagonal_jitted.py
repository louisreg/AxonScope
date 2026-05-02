from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag_jit


class JaxTridiagJitBenchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal_jit"

    def build_runner(self, problem):
        return build_jax_tridiag_jit(problem)


if __name__ == "__main__":
    JaxTridiagJitBenchmark().run()
