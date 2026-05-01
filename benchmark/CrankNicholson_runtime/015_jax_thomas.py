from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_thomas


class JaxThomasBenchmark(CNRuntimeBenchmark):
    label = "jax_thomas"

    def build_runner(self, problem):
        return build_jax_thomas(problem)


if __name__ == "__main__":
    JaxThomasBenchmark().run()
