from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_thomas_jit


class JaxThomasJitBenchmark(CNRuntimeBenchmark):
    label = "jax_thomas_jit"

    def build_runner(self, problem):
        return build_jax_thomas_jit(problem)


if __name__ == "__main__":
    JaxThomasJitBenchmark().run()
