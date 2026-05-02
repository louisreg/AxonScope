from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_tridiag


class JaxTridiagBenchmark(CNRuntimeBenchmark):
    label = "jax_tridiagonal"

    def build_runner(self, problem):
        return build_jax_tridiag(problem)


if __name__ == "__main__":
    JaxTridiagBenchmark().run()
