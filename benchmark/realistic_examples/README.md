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

Run the longer stress preset used for a realistic Kaggle pass:

```bash
python benchmark/realistic_examples/bench_basic_examples.py \
  --preset stress \
  --platforms cpu gpu \
  --run-counts 5 10 20 \
  --family-counts 25 50 \
  --example07-max-iterations 20 \
  --example08-amplitude-count 8 \
  --repeats 3 \
  --warmups 1 \
  --profile
```

Compare CPU/GPU while retaining only one Vm column for example 08 recruitment:

```bash
python benchmark/realistic_examples/bench_basic_examples.py \
  --preset stress \
  --platforms cpu gpu \
  --example08-recording center \
  --run-counts 5 10 20 \
  --family-counts 25 50 \
  --example07-max-iterations 20 \
  --example08-amplitude-count 8 \
  --repeats 3 \
  --warmups 1 \
  --profile
```

Outputs are written under `benchmark/results/realistic_examples/` as JSON and
CSV. With `--platforms cpu gpu`, the parent process also writes
`<prefix>_cpu_vs_gpu.csv` with first-run, total-first, and warm-run speedups.
Plots are enabled by default:

- `<prefix>_cpu_timings.svg/png`
- `<prefix>_gpu_timings.svg/png`
- `<prefix>_cpu_vs_gpu_speedup.svg/png`

Use `--no-plots` for CSV-only dry benchmark runs.

Use `--profile` when you need solver/workflow breakdowns for the same cases.
This writes:

- `<prefix>_cpu_profile.csv` and `<prefix>_gpu_profile.csv`
- `<prefix>_profile_cpu_vs_gpu.csv`
- `<prefix>_cpu_profiles/` and `<prefix>_gpu_profiles/` with raw
  `events.jsonl`, per-run `summary.csv`, and metadata

Important profile events include `runtime.prepare`, `inputs.intracellular`,
`inputs.extracellular`, `kernel.enqueue`, `kernel.wait`,
`results.split_batch`, and `results.to_public`.

Important columns:

- `workflow`: `example06_velocity`, `example07_threshold`, or
  `example08_recruitment`
- `fiber_type`: `hh`, `rattay`, `mrg`, or `mixed`
- `run_count`: number of simulated fibers/rows
- `recording`: `full`, `probes9`, `center`, or `observer_only`
- `platform_label`, `jax_backend`, `jax_devices`: backend information
- `build_s`: public object construction time
- `first_run_s`: first workflow execution time
- `warm.mean_s`, `warm.median_s`: measured repeat times after warmup
- `first_run_peak_rss_mib`, `warm_peak_rss_mib`, `process_peak_rss_mib`:
  measured host-process RSS, sampled during each case; `process_peak_rss_mib`
  is the OS high-water mark for the benchmark process

Use this benchmark as the next performance gate for `Vext` work. The solver
baseline should remain `pcr_adaptive` on GPU until a future candidate passes
both E2E speed and physiology agreement validation.
