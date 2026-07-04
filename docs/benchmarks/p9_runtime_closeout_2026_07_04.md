# P9 Runtime Performance Closeout - 2026-07-04

This note closes the short P9 cold-run/runtime-performance slice. It records
local evidence and product decisions; it is not a publication-grade performance
claim.

## Scope Closed

P9 was intentionally a small planning and observability phase before larger
runtime work. It now has:

- a short local cold-run baseline command, `cold_run_micro`;
- normalized scalar and batch benchmark span coverage;
- opt-in RSS/JAX profiling paths inherited from P0.5;
- explicit hotpath CLI controls for `time_chunk_steps`, including `default`,
  integer chunk sizes, and `none`/`unchunked`;
- a small observer-only chunking smoke on local CPU;
- clear decisions about what stays parked for future benchmark campaigns.

## Raw Outputs

- `benchmark/results/hotpaths/cold_run_micro_20260703/`
- `benchmark/results/hotpaths/cold_run_micro_scalar_spans_20260703/`
- `benchmark/results/hotpaths/p9_time_chunk_default_20260704/`
- `benchmark/results/hotpaths/p9_time_chunk_none_20260704/`
- `benchmark/results/hotpaths/p9_time_chunk_250_20260704/`
- `benchmark/results/hotpaths/p9_time_chunk_default_warm_20260704/`
- `benchmark/results/hotpaths/p9_time_chunk_none_warm_20260704/`
- `benchmark/results/hotpaths/p9_time_chunk_250_warm_20260704/`

## Time-Chunk Smoke

Command shape:

```bash
/Users/louisregnacq/miniforge3/envs/Axonscope-env/bin/python \
  benchmark/hotpaths/run.py \
  --workload observer_only \
  --sizes 5 \
  --duration 5.0 \
  --dt 0.01 \
  --warmups 1 \
  --memory-trace rss \
  --prefix p9_time_chunk_<policy>_warm_20260704
```

Local CPU warm smoke, `nt=500`, `n=5`, observer-only VmRaster:

| Policy | Simulation Case | Kernel Enqueue | Kernel Dispatch Events | Max RSS Delta |
| --- | ---: | ---: | ---: | ---: |
| workload default (`BatchOptions.none()` -> 50) | 15.609 ms | 8.598 ms | 1 | 0.008 MiB |
| `--time-chunk-steps none` | 14.656 ms | 8.815 ms | 1 | 0.008 MiB |
| `--time-chunk-steps 250` | 21.196 ms | 14.676 ms | 2 | 0.004 MiB |

Interpretation: this small CPU smoke does not justify changing the current
observer-only default. `none` and the workload default are close at this size;
`250` creates two dispatch annotations for `nt=500` and is slower here. Larger
CPU/GPU workloads still need a dedicated campaign before changing policy.

## Decisions

- Keep `cold_run_micro` as the short local P9 baseline. Do not add a
  process-isolated or rotated cold comparison yet; use that only for future
  publication-grade per-path cold-start evidence.
- Keep double-cable shape bucketing internal and opt-in through
  `AXONSCOPE_EXPERIMENTAL_DOUBLE_CABLE_SHAPE_BUCKETING`. There is no
  end-to-end evidence to make it a public/default route.
- Do not add an AxonScope-owned persistent JAX compilation cache policy in P9.
  Use `--jax-log-compiles`, JAX traces, and existing generated membrane-code
  cache metadata for diagnosis. Revisit only if cold compilation becomes a
  product requirement.
- Keep recruitment/amplitude sweeps sequential by default. The synthetic NRV
  GPU report already showed full-Vm output would dominate memory if all 21
  amplitudes were retained together, so amplitude micro-batching belongs in a
  future benchmark campaign, not in the runtime default.
- Keep `BatchOptions.none()` defaulting to
  `DEFAULT_OBSERVER_TIME_CHUNK_STEPS`. It remains a conservative observer-only
  default for duration-sweep signature stability; explicit `time_chunk_steps=None`
  is available when unchunked comparison is desired.
- Park runtime optimization after this evidence pass. The next optimization
  round should start from large synthetic/GPU profiling (`n=1000`) and split
  pool construction, dispatch planning, runtime preparation, kernel dispatch,
  memory pressure, and result assembly before changing solver routes.
- Keep GPU dispatch scheduling, double-cable rank-K compact `Vext`, and exact
  GPU solver improvements as future work requiring separate validation and
  benchmark evidence.

## Useful Commands After P9

Short local baseline:

```bash
python benchmark/hotpaths/run.py \
  --workload cold_run_micro \
  --sizes 1 \
  --duration 1.0 \
  --dt 0.02 \
  --warmups 0 \
  --memory-trace rss \
  --prefix cold_run_micro
```

Observer-only chunk policy probe:

```bash
python benchmark/hotpaths/run.py \
  --workload observer_only \
  --sizes 5 \
  --duration 5.0 \
  --dt 0.01 \
  --warmups 1 \
  --memory-trace rss \
  --time-chunk-steps none \
  --prefix observer_only_unchunked_probe
```

Deep GPU-oriented work should start from `benchmark/hotpaths/` for hotpath
spans and from `benchmark/realistic_examples/` or `benchmark/nrv_performance/`
for public workflow evidence.
