# P11B Thomas Batched Scan Gate

Date: 2026-07-07

This note records the benchmark-only gate for the existing batch-native
block-Thomas scan:

- `solve_block_tridiagonal_2x2_scalar_batched`
- benchmark variant name: `thomas_batched_scan`

The candidate was intentionally generic. It was not a membrane-model branch and
not an MRG-specific runtime path. It was exposed only in benchmark analysis
tools to compare the active GPU PCR-SoA path against a strong many-small-system
Thomas-family baseline, as suggested by the solver roadmap notes.

## Artifacts

Local CPU smoke artifacts, commit `c0bf619`:

- `benchmark/results/p11b_thomas_batched_scan_real_smoke`
- `benchmark/results/p11b_thomas_batched_scan_lowering_smoke`

Kaggle P100 artifacts, commit `c0bf619`:

- `benchmark/results/kaggle/20260707_190539_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-thomas-scan-lower-gpu/outputs/extracted`
- `benchmark/results/kaggle/20260707_190552_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-thomas-scan-real-gpu/outputs/extracted`

Workload: `Naxons=512`, requested `Nx=101`, actual kernel `Nx=89`, fp32,
different-diameter observer-only double-cable inputs on Kaggle
`Tesla P100-PCIE-16GB`.

## Lowering

| variant | optimized HLO lines | bytes | divides | fusions | gathers | selects | transposes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `thomas_vmap` | 775 | 96034 | 8 | 36 | 0 | 0 | 18 |
| `thomas_batched_scan` | 795 | 97022 | 8 | 42 | 0 | 0 | 10 |
| `pcr_soa_batched` | 2267 | 348962 | 60 | 7 | 184 | 117 | 0 |

The Thomas-family variants have much smaller IR than PCR-SoA and avoid the
gather/select-heavy PCR lowering shape. The batch-native scan also reduces
transpose count versus the vmap Thomas route. This is a compiler-size win, but
not a runtime win on the tested GPU.

## Timing

| variant | mean block solve | min block solve | first run |
| --- | ---: | ---: | ---: |
| `thomas_vmap` | 2.267 ms | 2.248 ms | 476.9 ms |
| `thomas_batched_scan` | 3.698 ms | 3.283 ms | 411.7 ms |
| `pcr_soa_batched` | 0.453 ms | 0.371 ms | 2416.0 ms |

For the one-step proxy:

| variant | mean one-step | min one-step | first run |
| --- | ---: | ---: | ---: |
| `thomas_batched_scan_real` | 2.172 ms | 2.082 ms | 704.3 ms |
| `thomas_batched_scan_real_precomputed_static` | 2.176 ms | 2.068 ms | 607.6 ms |
| `pcr_soa_batched_real` | 0.564 ms | 0.471 ms | 2566.8 ms |
| `pcr_soa_batched_real_precomputed_static` | 0.535 ms | 0.377 ms | 2449.1 ms |

The plain XLA scan Thomas route is about `8.2x` slower than PCR-SoA for the
isolated hot block solve and about `4.1x` slower in the one-step proxy.

## CPU Smoke

The local CPU smoke was useful as a sanity check:

| variant | mean block solve |
| --- | ---: |
| `thomas_vmap` | 0.285 ms |
| `thomas_batched_scan` | 0.210 ms |
| `pcr_soa_batched` | 0.706 ms |

This supports the existing CPU direction: Thomas-style scans are still the
right family on CPU-like platforms. It does not justify a GPU policy change.

## Decision

Reject `thomas_batched_scan` as a GPU runtime-policy candidate in its current
JAX/XLA form. Keep the benchmark-only variant because it is useful for future
CPU and compiler diagnostics, but do not route GPU `auto` through it.

This gate is useful negative evidence: smaller HLO and fewer gather/select
operations are not enough when the algorithm exposes a long sequential scan over
`Nx`. The active GPU route remains PCR-SoA for now.

Next GPU work should change a measured hot-path cost, not just the compiler
surface size. Candidate directions are:

- reduce PCR-SoA stage live state or memory traffic only if hot timing improves;
- inspect launch/stage boundaries around the fused one-step, since the
  no-solve proxy is already a material part of the fused cost;
- consider genuinely GPU-native tiled/custom/Pallas-style solver work only
  after the current JAX PCR-SoA path is fully mapped and a narrow candidate is
  justified.
