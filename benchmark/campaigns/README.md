# Benchmark Campaigns

Campaigns build reproducible matrices around the canonical benchmark scripts.
They call `benchmark/run.py` in isolated processes and do not implement solver
or workload logic.

## Time-Chunk Sweep

`time_chunk_sweep.py` compares observer/kernel chunk sizes across threshold or
recruitment workloads. Each policy receives a separate output directory and
the campaign writes a manifest, merged CSV summary, and Markdown report.

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,50,250,500 \
  --recordings full_vm,probe_vm,observer_only \
  --memory-trace rss \
  --output benchmark/results/time_chunk_sweep
```

Use `--recording` for one output mode or `--recordings` for a matrix. Summary
rows include observed chunk metadata and the runtime preparation, input,
kernel, observer, Vm materialization, and result-assembly stages.

## Double-Cable Solver Policy

`double_cable_solver_policy.py` validates the typed CPU and GPU double-cable
policy requests through the threshold and recruitment workloads. It is a
policy-level acceptance campaign; low-level solver benchmarks only explain
its results.

```bash
python benchmark/campaigns/double_cable_solver_policy.py \
  --platform cpu \
  --preset quick \
  --solver auto,thomas \
  --recording observer_only \
  --dry-run \
  --output benchmark/results/double_cable_solver_policy
```

On GPU, use `--solver auto,tiled_thomas` and optionally vary
`--tiled-thomas-block-b`. Every retained result must include the generated
manifest, raw child outputs, git metadata, and hardware metadata.
