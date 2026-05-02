from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_thomas_jit_optim


class JaxThomasJitOptimBenchmark(CNRuntimeBenchmark):
    label = "jax_thomas_jit_optim"

    def build_runner(self, problem):
        return build_jax_thomas_jit_optim(problem)


if __name__ == "__main__":
    JaxThomasJitOptimBenchmark().run()
