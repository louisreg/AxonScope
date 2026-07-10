# P11 GPU Hot-Path Cleanup Plan - 2026-07-10

This note records the current non-solver GPU bottleneck cleanup state before
making any double-cable solver policy decision.

## Current Evidence

Reference mini-gate shape:

- Kaggle P100, fp32, `recruitment_curves`
- double-cable observer-only
- `Naxons=4096`, target/actual `Nx=89`
- same-diameter cohort
- `tsim=2 ms`, `dt=0.02 ms`
- `tiled_thomas_b32`
- 1 warmup, 1 repeat, 3 amplitudes

Recent commits:

- `a4b3f88`: ampere-native benchmark stimuli reduced
  `curve.update_amplitudes.rows` from about `2.22 s` total to about `0.49 s`
  total over the mini-gate.
- `a5effea`: cached factorized footprints already converted to mV and
  detected shared rank-1 stimuli by identity. `inputs.extracellular` fell from
  about `494.7 ms` total to about `353.7 ms` total on the same mini-gate.
- `c6e842b`: eager JAX repack of VmRaster chunks was tested and rejected.
  It removed the explicit chunk-state wait, but increased
  `kernel.combine_observer_chunks` from about `0.34 s` to about `0.68 s` total
  and increased `kernel.finalize_observer.to_host` from about `0.011 s` to
  about `0.041 s`.

Interpretation:

- Factorized input preparation is improved but still visible.
- A simple post-hoc device repack is not useful.
- The remaining observer-only target is structural: avoid chunk repacking, or
  make a benchmark-backed chunk policy change.

## Next Prototype

Add a benchmark-only observer-state scope override:

```bash
--benchmark-observer-state-scope full
```

The solver time loop can stay chunked with `--time-chunk-steps N`, while
VmRaster keeps one full-duration state and receives the absolute time start for
each chunk. This tests:

- same chunked solver workload,
- no `kernel.combine_observer_chunks`,
- one final `kernel.finalize_observer.to_host`,
- no public `BatchOptions` or runtime policy change.

Acceptance for the prototype:

- Local unit tests show identical VmRaster words versus the default chunked
  observer-state path.
- Local smoke run finishes and reports no `kernel.combine_observer_chunks` when
  the override is `full`.
- P100 mini-gate improves or at least cleanly shifts time without increasing
  `kernel.dispatch_jax`, `kernel.finalize_observer.to_host`, or memory enough
  to erase the gain.

If the P100 mini-gate is positive, run a small CPU/GPU matrix across `Nt`,
`Naxons`, and observer-only/probe Vm before considering a default policy. If it
is negative, keep the evidence and move to `curve.update_amplitudes.rows` or a
more structural VmRaster result representation.

## First P100 Gate

Commit `5bf9a8e` passed the same P100 mini-gate with explicit
`--time-chunk-steps 50` and `--benchmark-observer-state-scope full`.

Compared with the `a5effea` default chunk-local observer-state mini-gate:

- `kernel.combine_observer_chunks`: about `0.336 s` total -> `0`.
- `kernel.finalize_observer`: about `0.012 s` total -> about `0.215 s`.
- `kernel.finalize_observer.to_host`: about `0.011 s` total -> about `0.040 s`.
- warm `curve.simulate` mean: about `329 ms` -> about `256 ms`.
- total `curve.simulate`: about `11.59 s` -> about `10.56 s`.

Interpretation:

- The structural idea works: the repack disappears.
- Some cost moves into finalization because the final observer state is larger.
- The net mini-gate signal is positive, but it needs a same-commit A/B matrix
  before any default or policy decision.

Artifact:

```text
benchmark/results/kaggle/20260710_193127_double_cable_solver_policy_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-p11-observer-full-state-5bf9a8e/outputs/extracted
```

## Same-Commit A/B Preparation

The double-cable solver policy campaign now accepts comma-separated observer
state scopes:

```bash
--benchmark-observer-state-scope default,full
```

This runs the default chunk-local observer state and the full-duration observer
state in the same campaign, so the comparison shares one commit, one Kaggle
image, and one hardware allocation. The campaign summary/report include
`observer_state_scope`, and the solver-policy plotter treats that value as a
condition dimension instead of merging `default` and `full` rows.

Local gate:

- two-run CPU smoke with `default,full` passed;
- plot smoke on that summary passed;
- unit coverage checks that `default` keeps the historical command and `full`
  gets a distinct `__obs_full` run with the explicit benchmark override.

Next gate:

- rerun the P100 mini shape with
  `--benchmark-observer-state-scope default,full`;
- use that same-commit A/B result to decide whether the full observer-state
  prototype deserves a broader CPU/GPU matrix.
