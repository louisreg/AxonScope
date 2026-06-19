# Kaggle CLI Solver Benchmarks

Kaggle can run the solver benchmarks as a script kernel when Colab GPU runtimes
are unavailable.

The Kaggle CLI is installed with:

```bash
python -m pip install kaggle
```

Authenticate with one of Kaggle's supported methods, for example:

```bash
kaggle auth login
```

or a legacy `~/.kaggle/kaggle.json` token.

## Prepare The Code

The one-command runner publishes the current commit to a dedicated Kaggle branch
named `kaggle-bench/<short-sha>` before submitting the kernel. Kaggle then clones
that branch, which prevents queued runs from accidentally pulling a moving
development branch. Only committed files are included in that branch.

Commit source changes before running:

```bash
git add -A
git commit -m "Benchmark Kaggle solver run"
make bench-colab-push
```

The manual branch path is still available with `--no-publish-branch --branch bench-colab`.

Generate `kernel-metadata.json`, `kaggle_config.json`, and the generated
self-contained Kaggle script locally:

```bash
python benchmark/kaggle/prepare_kernel_metadata.py \
  --username YOUR_KAGGLE_USERNAME \
  --benchmark linear
```

## Run On Kaggle

The one-command runner prepares the metadata, pushes the kernel, polls status,
fetches logs, lists files, and downloads outputs:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --benchmark smoke \
  --machine-shape NvidiaTeslaP100 \
  --poll-interval 30
```

Artifacts are saved under `benchmark/results/kaggle/<timestamp>_<benchmark>_<gpu>/`.
By default the runner downloads only files matching `axonscope_solver_results`;
the cloned repo stays under `/tmp/AxonScope` inside Kaggle so it is not persisted
as a bulky output.

`Ctrl+C` stops the local runner only. The remote Kaggle kernel can keep running;
the runner will fetch any available logs and print the Kaggle URL. To stop a run
without deleting the kernel, use the Kaggle UI. The CLI fallback is destructive:

```bash
kaggle kernels delete YOUR_KAGGLE_USERNAME/axonscope-double-cable-solver-benchmarks --yes
```

To resume monitoring/downloading an already submitted run without pushing a new
kernel:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --benchmark e2e \
  --machine-shape NvidiaTeslaP100 \
  --attach \
  --run-dir benchmark/results/kaggle/<existing-run-dir>
```

For disposable kernels, the runner can do that on interrupt:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --benchmark smoke \
  --machine-shape NvidiaTeslaP100 \
  --delete-kernel-on-interrupt
```

## Manual Commands

Run the isolated linear-solver matrix:

```bash
kaggle kernels push -p benchmark/kaggle --accelerator NvidiaTeslaP100 --timeout 43200
```

Run the end-to-end double-cable matrix:

```bash
python benchmark/kaggle/prepare_kernel_metadata.py \
  --username YOUR_KAGGLE_USERNAME \
  --benchmark e2e
kaggle kernels push -p benchmark/kaggle --accelerator NvidiaTeslaP100 --timeout 43200
```

The wrapper also accepts:

```text
--benchmark smoke
--benchmark linear
--benchmark linear_pcr_soa_trace
--benchmark e2e
--benchmark realistic_smoke
--benchmark realistic
--benchmark realistic_stress
--benchmark realistic_stress_single_vm
--benchmark realistic_stress_observer
--benchmark realistic_stress_observer_cpu
--benchmark realistic_stress_observer_gpu
--benchmark both
--benchmark e2e_full
--branch bench-colab
--repo-url https://github.com/louisreg/AxonScope.git
--no-require-gpu
```

`linear` measures the retained public solver routes: `thomas`, `pcr`,
`pcr_soa`, and `pcr_adaptive`. `linear_pcr_soa_trace` is the only focused
diagnostic preset kept in the active wrapper; it records JAX traces for the
retained PCR/SoA route. `e2e` is the bounded production matrix, while
`e2e_full` runs the exhaustive matrix and can take a long time on Kaggle.
`realistic_smoke` and `realistic` run workflow-level benchmarks based on basic
examples 06/07/08 twice inside the same GPU-enabled Kaggle kernel: once with
`JAX_PLATFORM_NAME=cpu`, once with `JAX_PLATFORM_NAME=gpu`. The output includes
per-platform CSV/JSON files plus a `realistic_examples_cpu_vs_gpu.csv`
comparison table and SVG/PNG timing plots. `realistic_stress` is the longer
CPU-vs-GPU workflow pass for Vext-oriented decisions: examples 06/07 run 5, 10,
and 20 diameters/fibers, example 08 runs mixed populations of 50 and 100 fibers,
and each case records warm-run repeats. It also enables realistic hotpath
profiling, so the output includes `realistic_examples_cpu_profile.csv`,
`realistic_examples_gpu_profile.csv`, and
`realistic_examples_profile_cpu_vs_gpu.csv` with event-level timings for
runtime preparation, input materialization, kernel enqueue/wait, batch splitting,
and public result packaging. `realistic_stress_observer` runs the same stress
matrix with example 08 recruitment in observer-only mode on both CPU and GPU.
The CPU child uses low-memory XLA/LLVM codegen flags and chunks example 08 into
single-fiber CPU sub-batches for this preset because larger observer-only CPU
batches exceed Kaggle's LLVM compile memory.
`realistic_stress_single_vm` is the fairer CPU/GPU comparison when recruitment
does not need full spatial traces: example 08 records a single center Vm column
instead of full `Vm`, while examples 06/07 keep their established recordings.
`realistic_stress_observer_gpu` is kept as a diagnostic preset, not a preferred
stress comparison: the first P100 stress attempt was stopped at the 50-fiber
recruitment case after already taking about 427 s, so the compact `center` Vm
path is the useful near-term comparison. CPU observer-only stress is currently
standby: Kaggle CPU-only and GPU-host attempts hit LLVM compile-memory errors
at the 50-fiber recruitment case. Use different slugs if submitting comparison
runs at the same time.

Closed exploration presets such as `linear_pallas_focus`, `linear_triton_focus`,
`linear_jax_triton_focus`, `linear_cuda_ffi_focus`, and
`e2e_jax_triton_focus` are intentionally no longer accepted by this wrapper.
Their code and historical evidence live under `benchmark/archived_solver_spikes/`,
`benchmark/triton_solver/`, `benchmark/jax_triton_solver/`, and
`benchmark/cuda_ffi_solver/`.

These values are written to `benchmark/kaggle/kaggle_config.json`, which is
uploaded with the kernel. The corresponding `AXONSCOPE_*` environment variables
are still supported for manual runs inside a Kaggle session.

Check status:

```bash
kaggle kernels status YOUR_KAGGLE_USERNAME/axonscope-double-cable-solver-benchmarks
```

List output files:

```bash
kaggle kernels files YOUR_KAGGLE_USERNAME/axonscope-double-cable-solver-benchmarks
```

Download outputs:

```bash
kaggle kernels output \
  YOUR_KAGGLE_USERNAME/axonscope-double-cable-solver-benchmarks \
  -p benchmark/results/kaggle \
  -o
```

The output zip is written by the script under Kaggle's working directory and
contains `linear/` and/or `e2e/` summary CSV/JSON files.
