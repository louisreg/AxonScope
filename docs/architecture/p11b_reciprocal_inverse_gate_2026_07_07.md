# P11B Reciprocal Inverse Gate

Date: 2026-07-07

This note records the GPU gate for the generic 2x2 inverse rewrite:

```python
inv_det = jnp.reciprocal(det)
return m11 * inv_det, -m01 * inv_det, -m10 * inv_det, m00 * inv_det
```

The candidate was intentionally generic: it was not a membrane-model branch and
not an MRG-specific runtime path. It targeted the shared 2x2 block inverse used
by the JAX cable solvers.

## Artifacts

Baseline P100 artifacts:

- `benchmark/results/kaggle/20260707_143514_pcr_soa_probe_lowering_gpu_512/outputs/extracted`
- `benchmark/results/kaggle/20260707_143514_pcr_soa_probe_real_gpu_512/outputs/extracted`
- `benchmark/results/kaggle/20260707_145211_one_step_probe_real_gpu_512/outputs/extracted`

Candidate P100 artifacts, commit `21ffaef`:

- `benchmark/results/kaggle/20260707_183755_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-recip-inv-lowering-gpu-512/outputs/extracted`
- `benchmark/results/kaggle/20260707_183812_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-recip-inv-real-gpu-512/outputs/extracted`

Workload: `Naxons=512`, requested `Nx=101`, actual `Nx=89`, fp32,
different-diameter observer-only double-cable inputs on Kaggle
`Tesla P100-PCIE-16GB`.

## Lowering

| case | optimized HLO lines | bytes | divides | multiplies | fusions | gathers | selects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `pcr_soa_batched` | 2267 | 348833 | 60 | 410 | 7 | 184 | 117 |
| reciprocal candidate | 2300 | 353565 | 15 | 470 | 7 | 184 | 117 |

The candidate did exactly what it was supposed to do at the HLO counter level:
`count_divide` dropped by 75%. It also added multiplies and slightly increased
compiled HLO size. Gather/select/fusion structure was unchanged.

## Timing

| case | mean block solve | min block solve | first run |
| --- | ---: | ---: | ---: |
| baseline sweep | 0.417 ms | 0.352 ms | 2426 ms |
| reciprocal candidate | 0.474 ms | 0.398 ms | 2517 ms |

The hot block solve regressed by about 13.7% on the mean and about 13.0% on
the min.

For the fused one-step proxy:

| case | mean one-step | min one-step | first run |
| --- | ---: | ---: | ---: |
| baseline one-step probe | 0.472 ms | 0.399 ms | 2342 ms |
| reciprocal candidate | 0.518 ms | 0.398 ms | 2750 ms |

The one-step mean regressed by about 9.8%; the minimum was essentially tied.

## Decision

Reject the reciprocal rewrite and keep the original division form. The code was
reverted after this gate.

This is useful negative evidence: the P100 hot solve is not improved by simply
reducing the HLO divide count. Future low-level solver work should target
measured runtime cost, memory movement, launch/staging behavior, or a genuinely
different algorithmic structure, not isolated instruction-count wins.
