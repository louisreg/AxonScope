# Colab GPU Hotpath Protocol

Use this when local GPU execution is not available. The local machine publishes
one committed AxonScope revision to the moving `bench-colab` branch, and Colab
always clones that branch before running the hotpath workloads.

## 1. Start A GPU Runtime

In Google Colab:

1. Open a new notebook.
2. Select `Runtime > Change runtime type`.
3. Select a GPU accelerator.
4. Restart the runtime after dependency installation if Colab asks for it.

## 2. Publish The Local Revision

From the local repository, commit the exact code to test and push it to the
dedicated Colab branch:

```bash
git add -A
git commit -m "Benchmark Colab run"
make bench-colab-push
```

`make bench-colab-push` refuses dirty working trees because Colab can only see
committed code. It updates the remote `bench-colab` branch without switching
your current local branch.

Override the remote or branch only if needed:

```bash
make bench-colab-push GIT_REMOTE=origin BENCH_COLAB_BRANCH=bench-colab
```

## 3. Clone, Install, And Run In Colab

Paste this cell into Colab. Replace `REPO_URL` once, then keep using the same
cell for later benchmark runs.

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
import datetime
import pathlib
import subprocess

REPO_URL = "https://github.com/YOUR_USER/YOUR_REPO.git"
BRANCH = "bench-colab"
PKG_DIR = pathlib.Path("/content/AxonScope")
DRIVE_ROOT = pathlib.Path("/content/drive/MyDrive/AxonScope/hotpaths")

def sh(command, cwd=None):
    print(f"\n$ {command}")
    subprocess.run(command, shell=True, cwd=cwd, check=True)

run_id = datetime.datetime.now().strftime("colab_gpu_%Y%m%d_%H%M%S")
out_dir = DRIVE_ROOT / run_id
out_dir.mkdir(parents=True, exist_ok=True)

sh(f"rm -rf {PKG_DIR}")
sh(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {PKG_DIR}")
sh("git rev-parse --short HEAD", cwd=PKG_DIR)
sh("python -m pip install -U pip", cwd=PKG_DIR)
sh('python -m pip install -e ".[examples,benchmark]"', cwd=PKG_DIR)
sh("nvidia-smi || true", cwd=PKG_DIR)
sh(
    "python - <<'PY'\n"
    "import jax\n"
    "print('jax backend:', jax.default_backend())\n"
    "print('jax devices:', jax.devices())\n"
    "if jax.default_backend() != 'gpu':\n"
    "    raise SystemExit('Colab runtime is not using a GPU backend.')\n"
    "PY",
    cwd=PKG_DIR,
)
sh(
    "python benchmark/hotpaths/run.py "
    "--workload all "
    "--preset scale "
    "--warmups 1 "
    f"--prefix {run_id} "
    f"--out-dir {out_dir.parent} "
    "--no-print-summary",
    cwd=PKG_DIR,
)

print(f"\nBenchmark results written to: {out_dir}")
```

The output folder will contain:

- `manifest.json`
- each workload's `events.jsonl`
- each workload's `summary.csv`
- each workload's `metadata.json`

## 4. Run A Matching Local CPU Reference

Run a local CPU reference with the same hotpath preset and warmup count:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --warmups 1 \
  --prefix cpu_YYYYMMDD_HHMMSS \
  --no-print-summary
```

If Google Drive Desktop is enabled locally, the Colab results can sync back
automatically from `MyDrive/AxonScope/hotpaths/<run_id>/`.

## 5. Compare Before Refactoring

Compare these stages first:

- `runtime.prepare`
- `inputs.intracellular`
- `inputs.extracellular`
- `kernel.enqueue`
- `kernel.wait`
- `results.split_batch`
- `results.to_public`

If CPU and GPU stay close while `inputs.extracellular` or `runtime.prepare`
dominates, prioritize Phase 3 preparation/cohort reuse before touching kernels.

If `kernel.wait` dominates and separates clearly by device, prioritize backend
kernel/runtime isolation.

## Troubleshooting

### `make bench-colab-push` Refuses To Run

The local working tree is dirty. Commit or stash first:

```bash
git status --short
```

### Colab Reports `Colab runtime is not using a GPU backend`

Switch the Colab runtime to a GPU accelerator and restart the runtime. The
trace is not useful for CPU/GPU comparison unless `jax.default_backend()` is
`gpu`.

### Colab Still Runs Old Code

The usual causes are:

- local changes were not committed before `make bench-colab-push`;
- the notebook cloned a branch other than `bench-colab`;
- the Colab runtime still has an old editable install.

The provided cell removes `/content/AxonScope` before cloning. If needed, also
restart the runtime.
