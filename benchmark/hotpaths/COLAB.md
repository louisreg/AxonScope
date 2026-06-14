# Manual Colab GPU Protocol

Use this when local GPU execution is not available. Keep it manual for now:
the goal is to collect comparable traces, not to build the final benchmark
automation.

## 1. Start A GPU Runtime

In Google Colab:

1. Open a new notebook.
2. Select `Runtime > Change runtime type`.
3. Select a GPU accelerator.
4. Restart the runtime after dependency installation if Colab asks for it.

## 2. Install AxonScope

Clone or upload this repository, then run from the repository root:

```bash
pip install -e ".[examples,benchmark]"
```

Check the JAX device:

```python
import jax

print(jax.default_backend())
print(jax.devices())
```

The GPU trace is only meaningful if `jax.default_backend()` reports `gpu`.

## 3. Run The Hotpath Scale Probe

From the repository root:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --prefix colab_gpu_YYYYMMDD \
  --no-print-summary
```

The output folder will be:

```text
benchmark/results/hotpaths/colab_gpu_YYYYMMDD/
```

Download the whole folder, including:

- `manifest.json`
- each workload's `events.jsonl`
- each workload's `summary.csv`
- each workload's `metadata.json`

## 4. Keep A CPU Reference

Run the same command on a CPU runtime or local CPU environment with a different
prefix:

```bash
python benchmark/hotpaths/run.py \
  --workload all \
  --preset scale \
  --prefix cpu_YYYYMMDD \
  --no-print-summary
```

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
