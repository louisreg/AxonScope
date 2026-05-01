from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag_jit_optim


class JaxTridiagJitOptimBenchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal_jit_optim"

    def build_runner(self, problem):
        return build_jax_tridiag_jit_optim(problem)


if __name__ == "__main__":
    JaxTridiagJitOptimBenchmark().run()
