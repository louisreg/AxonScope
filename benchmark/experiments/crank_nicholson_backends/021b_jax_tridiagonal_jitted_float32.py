from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag_jit_f32


class JaxTridiagF32Benchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal_jit_f32"

    def build_runner(self, problem):
        return build_jax_tridiag_jit_f32(problem)


if __name__ == "__main__":
    JaxTridiagF32Benchmark().run()
