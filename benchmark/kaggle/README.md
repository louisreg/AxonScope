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
--benchmark both
--benchmark linear_pcr_soa_layout_focus
--benchmark linear_triton_focus
--benchmark e2e_full
--branch bench-colab
--repo-url https://github.com/louisreg/AxonScope.git
--no-require-gpu
```

`e2e` is the bounded production matrix. `e2e_full` runs the exhaustive matrix and
can take a long time on Kaggle.

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
