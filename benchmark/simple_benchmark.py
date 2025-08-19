from axonscope.benchmark import Benchmark

# === Example Usage ===
bench = Benchmark()  # singleton instance

@bench.benchmark(level=1)
def step1(data):
    return [x * 2 for x in data]

@bench.benchmark(level=2)
def step2(data):
    return sum(data)

@bench.benchmark(level=1)
def process_data(data):
    step1(data)
    step1(data)
    return step2(data)

def main_process():
    data = list(range(10_000_000))
    return process_data(data)

if __name__ == "__main__":
    bench.enable(level=2, auto_print=True)
    main_process()
    #bench.stop()
    #bench.print()
