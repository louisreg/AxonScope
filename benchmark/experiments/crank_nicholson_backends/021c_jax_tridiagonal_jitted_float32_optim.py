from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag_jit_f32_optim


class JaxTridiagF32OptimBenchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal_jit_ultra_f32"

    def build_runner(self, problem):
        return build_jax_tridiag_jit_f32_optim(problem)


if __name__ == "__main__":
    JaxTridiagF32OptimBenchmark().run()
