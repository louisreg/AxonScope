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
