from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag_jit_f32_gateinterp2


class JaxTridiagGateInterp2Benchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal_jit_gateinterp2_f32"

    def build_runner(self, problem):
        return build_jax_tridiag_jit_f32_gateinterp2(problem)


if __name__ == "__main__":
    JaxTridiagGateInterp2Benchmark().run()
