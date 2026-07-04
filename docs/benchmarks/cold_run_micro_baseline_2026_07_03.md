# Cold-Run Micro Baseline - 2026-07-03

This note records the first P9 local cold-run baseline after the post-P7 docs
cleanup. It is evidence for optimization planning, not a performance claim.

## Commands

Model-codegen smoke:

```bash
/Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python \
  benchmark/runtime/run.py --suite model_codegen
```

Short cold-run hotpath baseline:

```bash
/Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python \
  benchmark/hotpaths/run.py \
  --workload cold_run_micro \
  --sizes 1 \
  --duration 1.0 \
  --dt 0.02 \
  --warmups 0 \
  --memory-trace rss \
  --prefix cold_run_micro_20260703
```

Raw outputs:

- `benchmark/results/runtime/model_codegen_20260703_155255.*`
- `benchmark/results/hotpaths/cold_run_micro_20260703/`
- `benchmark/results/hotpaths/cold_run_micro_scalar_spans_20260703/`

## Environment

- Backend: CPU
- JAX devices: `cpu:0`
- Platform: `macOS-15.7.7-x86_64-i386-64bit`
- Python: CPython 3.12.13 from `Axonscope-env`
- JAX: 0.10.1
- Host memory in hotpath metadata: 17.18 GB total, 6.295 GB available

## Model-Codegen Smoke

Correctness rows passed: `16/16`.

Selected source/codegen timings:

| Model | Cold codegen | Warm mean |
| --- | ---: | ---: |
| `passive` | 0.0098 s | 0.0027 s |
| `hodgkin_huxley` | 0.0118 s | 0.0028 s |
| `rattay_aberham` | 0.0104 s | 0.0023 s |
| `sundt` | 0.0160 s | 0.0050 s |
| `axnode` | 0.0367 s | 0.0056 s |
| `tigerholm` | 0.1070 s | 0.0367 s |
| `schild94` | 0.1605 s | 0.0379 s |
| `schild97` | 0.1581 s | 0.0399 s |

Selected model-step first-call signals:

| Model | First JAX runtime lowering | Subsequent mean |
| --- | ---: | ---: |
| `hodgkin_huxley` | 2.9675 s | 0.0249 s |
| `tigerholm` | 1.4315 s | 0.1983 s |
| `schild94` | 0.8440 s | 0.1095 s |
| `schild97` | 0.7900 s | 0.1558 s |

Interpretation: source/codegen cache behavior is healthy, but first JAX runtime
lowering is a visible cold-start component for non-trivial models.

## Cold-Run Micro Hotpath

`cold_run_micro` runs three public execution paths in one process:

1. `single_intracellular_center`
2. `single_intracellular_observer_none`
3. `single_point_source_center`

Case-level timings:

| Case | Wall time | RSS delta |
| --- | ---: | ---: |
| `single_intracellular_center` | 3425.570 ms | 185.395 MiB |
| `single_intracellular_observer_none` | 707.735 ms | 30.625 MiB |
| `single_point_source_center` | 1201.303 ms | 40.254 MiB |

Aggregate spans:

| Span | Count | Total | Max |
| --- | ---: | ---: | ---: |
| `simulation.case` | 3 | 5334.608 ms | 3425.570 ms |
| `simulation.pool.total` | 3 | 5329.404 ms | 3421.493 ms |
| `dispatch.group.total` | 3 | 5327.980 ms | 3420.854 ms |
| `kernel.enqueue` | 1 | 668.417 ms | 668.417 ms |
| `kernel.dispatch_jax` | 1 | 448.742 ms | 448.742 ms |
| `runtime.prepare.membrane_init` | 1 | 786.072 ms | 786.072 ms |
| `runtime.prepare.membrane_compile` | 1 | 354.185 ms | 354.185 ms |
| `observer.plan` | 1 | 26.226 ms | 26.226 ms |
| `dispatch.build_plan` | 3 | 0.151 ms | 0.069 ms |

RSS highlights:

| Span | Max RSS delta | Max RSS end |
| --- | ---: | ---: |
| `simulation.case` | 185.395 MiB | 444.379 MiB |
| `simulation.pool.total` | 185.363 MiB | 444.379 MiB |
| `dispatch.group.total` | 185.355 MiB | 444.379 MiB |
| `kernel.enqueue` | 30.574 MiB | 404.125 MiB |

## Audit Notes

- The short local baseline now has one retained command:
  `benchmark/hotpaths/run.py --workload cold_run_micro --sizes 1 --duration 1.0
  --dt 0.02 --warmups 0 --memory-trace rss --prefix cold_run_micro`.
- The broader `hotpath_matrix --preset smoke --memory-trace all` run was too
  heavy for an everyday local baseline. It reached 61 events and about 315 s of
  partial `simulation.pool.total` time before interruption, so it should remain
  a deliberate deep profiling run.
- `cold_run_micro` is useful for daily cold-start smoke, but it is not a fair
  independent cold comparison between path families because all cases run in
  one Python/JAX process. The first case absorbs one-time import/runtime/cache
  effects.
- The first baseline exposed a scalar retained-Vm instrumentation gap: retained
  scalar cases mostly showed dispatch/runtime membrane spans, while the
  observer-only batch path exposed preparation, input lowering, enqueue/wait,
  and result assembly. The follow-up run below closes that gap.

## Follow-Up: Scalar Span Coverage

After normalizing scalar retained-Vm instrumentation, the same command with
`--prefix cold_run_micro_scalar_spans_20260703` produced 45 events. All three
micro cases now emit the standard hotpath stages: `runtime.prepare`,
`inputs.positions`, `observer.plan`, `inputs.intracellular`,
`inputs.extracellular`, `kernel.enqueue`, `kernel.wait`, and
`results.split_batch`.

Selected aggregate spans from
`benchmark/results/hotpaths/cold_run_micro_scalar_spans_20260703/`:

| Span | Count | Total | Max |
| --- | ---: | ---: | ---: |
| `simulation.case` | 3 | 5000.181 ms | 2888.173 ms |
| `simulation.pool.total` | 3 | 4995.506 ms | 2884.706 ms |
| `dispatch.group.total` | 3 | 4994.076 ms | 2883.901 ms |
| `runtime.prepare` | 3 | 3170.018 ms | 2548.836 ms |
| `kernel.enqueue` | 3 | 1770.014 ms | 723.042 ms |
| `observer.plan` | 3 | 27.393 ms | 27.353 ms |
| `inputs.intracellular` | 3 | 1.570 ms | 1.391 ms |
| `inputs.positions` | 3 | 0.452 ms | 0.213 ms |
| `inputs.extracellular` | 3 | 0.238 ms | 0.149 ms |
| `kernel.wait` | 3 | 0.179 ms | 0.113 ms |
| `results.split_batch` | 3 | 0.111 ms | 0.053 ms |

## P9 Closeout

P9 is closed in `docs/benchmarks/p9_runtime_closeout_2026_07_04.md`. The
process-isolated cold comparison, persistent JAX compilation/cache policy,
amplitude micro-batching, large synthetic/GPU cold-path audit, GPU dispatch
scheduling, double-cable rank-K compact `Vext`, and exact GPU solver work are
future benchmark/optimization topics rather than current P9 tasks.
