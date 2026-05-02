from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag_jit_f32_gateinterp


class JaxTridiagGateInterpBenchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal_jit_gateinterp_f32"

    def build_runner(self, problem):
        return build_jax_tridiag_jit_f32_gateinterp(problem)


if __name__ == "__main__":
    JaxTridiagGateInterpBenchmark().run()
