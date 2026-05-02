from benchmark_base import CNRuntimeBenchmark
from runtime_backends import build_jax_thomas_jit_optim_2


class JaxThomasJitOptim2Benchmark(CNRuntimeBenchmark):
    label = "jax_thomas_jit_optim_2"

    def build_runner(self, problem):
        return build_jax_thomas_jit_optim_2(problem)


if __name__ == "__main__":
    JaxThomasJitOptim2Benchmark().run()
