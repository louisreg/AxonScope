# Realistic Basic Example Benchmarks

This benchmark runs workflow-level cases based on the public basic examples:

- example 06: velocity versus diameter (`simulate_pool`)
- example 07: threshold versus diameter (`find_activation_threshold_curve`)
- example 08: recruitment curve for a mixed population (`recruitment_sweep`)

It complements the solver-only benchmarks by measuring complete public
workflows, including pool construction, protocol loops, `Vext` materialization,
solver runtime, and result packaging.

Run a tiny local smoke on the current backend:

```bash
python benchmark/realistic_examples/bench_basic_examples.py \
  --preset smoke \
  --repeats 1 \
  --dry-run
```

Run the smoke for real:

```bash
python benchmark/realistic_examples/bench_basic_examples.py \
  --preset smoke \
  --repeats 1
```

Compare CPU and GPU in isolated child processes:

```bash
python benchmark/realistic_examples/bench_basic_examples.py \
  --preset standard \
  --platforms cpu gpu \
  --run-counts 2 5 10 \
  --family-counts 5 25 50 \
  --repeats 3 \
  --warmups 1
```

Outputs are written under `benchmark/results/realistic_examples/` as JSON and
CSV. Important columns:

- `workflow`: `example06_velocity`, `example07_threshold`, or
  `example08_recruitment`
- `fiber_type`: `hh`, `rattay`, `mrg`, or `mixed`
- `run_count`: number of simulated fibers/rows
- `platform_label`, `jax_backend`, `jax_devices`: backend information
- `build_s`: public object construction time
- `first_run_s`: first workflow execution time
- `warm.mean_s`, `warm.median_s`: measured repeat times after warmup

Use this benchmark as the next performance gate for `Vext` work. The solver
baseline should remain `pcr_adaptive` on GPU until a future candidate passes
both E2E speed and physiology agreement validation.
