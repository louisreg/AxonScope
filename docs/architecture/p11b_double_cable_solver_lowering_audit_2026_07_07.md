# P11B Double-Cable Solver Lowering Audit

Date: 2026-07-07

Commit: `e1101ab9d8b126457b393b7e9452fa34c7e6d24c`

This note records the first Kaggle GPU lowering/codegen audit for
`benchmark/analysis/double_cable_solver_lowering_audit.py`. The goal was to
inspect current GPU PCR/SoA lowering before adding any new double-cable solver
route.

## Run Shape

- Platform image: Kaggle P100 image.
- GPU hardware: Tesla P100-PCIE-16GB.
- OS: Linux 6.12.90 x86_64.
- Python: 3.12.13.
- JAX: 0.10.2.
- Workload: real prepared double-cable observer-only VmRaster inputs.
- Requested shape: `Naxons=512`, `Nx=101`.
- Actual kernel shape: `kernel_group_size=512`, `actual_nx=89`.
- Precision: fp32.
- Device: `cuda:0`.
- Compiled IR: optimized HLO from XLA GPU compilation.

Valid artifact roots:

- Different diameters:
  `benchmark/results/kaggle/20260707_141000_solver_lowering_diff_gpu_512/outputs/extracted`
- Same diameter:
  `benchmark/results/kaggle/20260707_141000_solver_lowering_same_gpu_512/outputs/extracted`

## Important Context

The two runs are not identical membrane/compiler cases:

- Different diameters used `GatedLeakStackMembraneBackend`,
  `JaxMembraneProgram`, `uses_generated_model_step=true`,
  `unique_row_diameters=5`, and `shared_coefficients=false`.
- Same diameter used `HeterogeneousMembraneBackend`,
  `HeterogeneousMembraneModel`, `uses_generated_model_step=false`,
  `unique_row_diameters=1`, and `shared_coefficients=true`.

That means the isolated `block_solve` lowering is the cleanest comparison for
PCR/SoA layout. The `one_step_proxy` lowering is still useful, but it mixes
solver, membrane backend, conductance terms, system assembly, and observer
boundary effects.

## Optimized HLO Counts

### Isolated Block Solve

| diameters | variant | lines | gather | broadcast | select | fusion | transpose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| different | `pcr_soa_batched` | 2267 | 184 | 80 | 117 | 7 | 0 |
| different | `pcr_soa_vmap` | 2271 | 184 | 86 | 117 | 7 | 0 |
| same | `pcr_soa_batched` | 2279 | 184 | 84 | 117 | 7 | 0 |
| same | `pcr_soa_vmap` | 2286 | 184 | 90 | 117 | 7 | 0 |

### One-Step Proxy

| diameters | variant | lines | gather | broadcast | select | fusion | transpose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| different | `pcr_soa_real` | 2862 | 182 | 130 | 125 | 10 | 0 |
| same | `pcr_soa_real` | 2999 | 186 | 147 | 125 | 19 | 3 |

## Reading

The isolated GPU PCR/SoA solver paths are almost structurally identical after
XLA optimization. `pcr_soa_batched` is slightly smaller than `pcr_soa_vmap`,
mostly through fewer broadcasts and a few fewer lines, but gather/select/fusion
counts are the same and no transpose remains in the optimized solver HLO.

Same-diameter coefficient sharing does not simplify the isolated optimized
PCR/SoA block solver at this shape. The optimized HLO counts remain within the
same small band as the different-diameter case.

The `one_step_proxy` differs more, especially for fusion count and a few
remaining transposes in the same-diameter run. Because that run also changes
the membrane backend and generated-model status, this should be treated as a
compiler/backend signal rather than a pure solver signal.

## Follow-Up Fusion/Layout View

The next tooling slice added `benchmark/analysis/hlo_fusion_summary.py` plus
automatic fusion/layout summaries in the lowering audit. Re-reading the two
GPU artifact roots above shows the dominant isolated PCR/SoA pattern more
clearly:

- both `pcr_soa_batched` and `pcr_soa_vmap` compile to seven solver fusions;
- the five large reduction/update fusions are `loop_select_subtract*`;
- the four central reduction fusions each return 22 arrays of shape
  `[512,89]`, about 3.82 MiB of output per fusion in fp32;
- the last `loop_add_fusion` returns the two solution arrays;
- both variants use the same batch-first `[512,89]{1,0}` entry layout, while
  gather bodies still expose many `[89,512,1]{2,0,1}` gather intermediates
  that are bitcast back to `[512,89]`.

This reinforces the same decision: the next PCR/SoA work should inspect
fusion bodies, live tuples, and gather/layout behavior inside the current
solver, not add a vmap-vs-batched public route.

The first benchmark-only candidate following this reading is
`pcr_soa_symmetric_batched`. It uses the exact double-cable symmetry invariant
to carry one side of the PCR coupling state and reconstruct the opposite side
by shifted transpose. It is intentionally wired only into benchmark analysis
tools. Local CPU smoke checks show the candidate matches the current masked
PCR/SoA solver on synthetic symmetric systems, lowers to a smaller optimized
HLO (`23393` to `16751` lines and `228` to `186` fusions on the tiny CPU
smoke), and runs through the real-stage profiler. On a tiny real prepared fp32
system, the candidate differs from current PCR/SoA by `6.7e-4` absolute on a
roughly `80 mV` solution range, with max relative residual `9.0e-6` versus
`8.4e-6` for current PCR/SoA and `8.4e-7` for Thomas. This is good enough for
GPU benchmark exploration, but not enough to choose a runtime route; the next
gate is GPU HLO/timing on the P100 workload.

That P100 gate was run under:

- `benchmark/results/kaggle/20260707_140519_symmetric_pcr_lowering_gpu_512/outputs/extracted`
- `benchmark/results/kaggle/20260707_140519_symmetric_pcr_real_gpu_512/outputs/extracted`

For `Naxons=512`, requested `Nx=101`, actual kernel `Nx=89`, fp32,
different-diameter observer-only double-cable inputs on `Tesla P100-PCIE-16GB`,
the candidate does reduce compiler pressure: optimized HLO shrinks from `2267`
to `1855` lines, gathers from `184` to `134`, selects from `117` to `87`, and
the largest PCR fusion output drops from `22` arrays / `3.82 MiB` to `14`
arrays / `2.43 MiB`. Total estimated PCR fusion outputs drop from `21.90 MiB`
to `14.25 MiB`.

The hot runtime gain is small: isolated block solve mean goes from `0.415 ms`
to `0.404 ms` (`-2.6%`), with first-run compile/execute time from `2253.6 ms`
to `1839.4 ms` (`-18.4%`). This confirms that live tuple pressure is real, but
also that reducing this particular carried state is not enough to justify a
runtime route by itself.

## Decision

Do not add a new `pcr_soa_vmap` versus `pcr_soa_batched` runtime route. The
current batch-native PCR/SoA path is a reasonable active GPU route and has no
obvious high-level lowering defect relative to the vmap shape.

Do not promote `pcr_soa_symmetric_batched` as-is. Keep it as a benchmark-only
probe: it is useful evidence that PCR live-state size affects the generated
program, but it does not yet produce a large enough hot-path win.

## Existing PCR/SoA Probe Sweep

A second P100 gate compared existing low-level PCR/SoA probes already present
in `src/axonscope/backends/jax/common.py`:

- `benchmark/results/kaggle/20260707_143514_pcr_soa_probe_lowering_gpu_512/outputs/extracted`
- `benchmark/results/kaggle/20260707_143514_pcr_soa_probe_real_gpu_512/outputs/extracted`

The run used commit `e5e0c85`, `Naxons=512`, requested `Nx=101`, actual kernel
`Nx=89`, fp32, different-diameter observer-only double-cable inputs on
`Tesla P100-PCIE-16GB`.

| variant | optimized HLO lines | gathers | selects | fusions | max fusion output | hot block solve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pcr_soa_batched` | 2267 | 184 | 117 | 7 | 22 arrays / 3.82 MiB | 0.417 ms |
| `pcr_soa_nomask_batched` | 2054 | 184 | 13 | 7 | 22 arrays / 3.82 MiB | 0.607 ms |
| `pcr_soa_shift_batched` | 1821 | 0 | 0 | 7 | 22 arrays / 3.82 MiB | 0.425 ms |
| `pcr_soa_transposed_batched` | 2410 | 184 | 117 | 7 | 22 arrays / 3.82 MiB | 0.513 ms |
| `pcr_soa_padded_batched` | 2140 | 184 | 112 | 7 | 14 arrays / 3.50 MiB | 0.416 ms |
| `pcr_soa_hybrid_batched` | 7105 | 304 | 125 | 206 | 73 arrays / 3.82 MiB | 2.904 ms |

`pcr_soa_shift_batched` is the most useful diagnostic: it eliminates gathers
and selects from the optimized HLO and lowers first-run time (`2426 ms` to
`1636 ms`), but does not improve hot GPU runtime (`+1.9%` versus baseline).
`pcr_soa_padded_batched` slightly reduces the largest fusion tuple and lands at
baseline-equivalent hot runtime (`-0.4%`), but the difference is within
measurement noise and it carries extra padding semantics. `nomask`,
`transposed`, and `hybrid` are worse on hot runtime for this workload.

Decision: do not promote any of these probes as-is. Keep `shift` and `padded`
as diagnostic references for future lowering experiments, because they isolate
useful compiler effects, but require a stronger hot-path or one-step win before
runtime integration.

The final one-step check was run under
`benchmark/results/kaggle/20260707_145211_one_step_probe_real_gpu_512/outputs/extracted`
on commit `c89ff93`, with the same P100 workload and explicit one-step variants
`pcr_soa_batched`, `pcr_soa_shift_batched`, and `pcr_soa_padded_batched`.
Results:

| one-step variant | mean | min | max | first run | delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pcr_soa_batched_real` | 0.472 ms | 0.399 ms | 0.511 ms | 2342 ms | baseline |
| `pcr_soa_shift_batched_real` | 0.529 ms | 0.402 ms | 0.678 ms | 1954 ms | +12.1% |
| `pcr_soa_padded_batched_real` | 0.485 ms | 0.412 ms | 0.537 ms | 2275 ms | +2.8% |

Conclusion: `shift` still improves first-run/compile behavior, but not hot
one-step runtime; `padded` is close but not a clear win. This closes the current
PCR/SoA micro-variant branch. Future work should use `shift` and `padded` only
as diagnostic references unless a new structural hypothesis changes the target.

## Precomputed Assembly Diagnostic

A separate benchmark-only diagnostic on commit `38dcfe5` compared the regular
one-step proxy with a `precomputed_static` assembly variant that precomposes
static double-cable diagonal bases, offsets, and absolute correction/background
terms outside the measured assembly call:

- `benchmark/results/kaggle/20260707_151450_double_cable_real_stage_profile_quick_gpu_NvidiaTeslaP100_axonscope-p11b-precomputed-assembly-gpu-512/outputs/extracted`

The run used the same P100 shape as the previous one-step check: `Naxons=512`,
requested `Nx=101`, actual kernel `Nx=89`, fp32, different-diameter
observer-only double-cable inputs, and `pcr_soa_batched`.

| stage | variant | mean | min | max | first run |
| --- | --- | ---: | ---: | ---: | ---: |
| `system_assembly` | `real_double_cable` | 0.299 ms | 0.261 ms | 0.351 ms | 0.361 ms |
| `system_assembly` | `precomputed_static` | 0.339 ms | 0.269 ms | 0.495 ms | 0.373 ms |
| `one_step_proxy` | `pcr_soa_batched_real` | 0.543 ms | 0.396 ms | 0.612 ms | 2501 ms |
| `one_step_proxy` | `pcr_soa_batched_real_precomputed_static` | 0.500 ms | 0.378 ms | 0.580 ms | 2541 ms |

The isolated assembly result moves the wrong way, but the fused one-step proxy
improves in the same run; measured one-step medians were roughly `0.558 ms`
baseline versus `0.479 ms` for `precomputed_static`. Treat this as a compiler
diagnostic, not a runtime decision. The production scan already precomputes
some static terms (`cm_over_dt`, `cx_over_dt`, axial left/right coefficients,
and off-diagonal edge arrays), so the relevant question is whether precomposing
the remaining diagonal/RHS pieces changes fusion shape enough to matter in the
real scan body.

A matching P100 lowering audit was run here:

- `benchmark/results/kaggle/20260707_151930_double_cable_solver_lowering_audit_quick_gpu_NvidiaTeslaP100_axonscope-p11b-precomputed-lowering-gpu-512/outputs/extracted`

The compiled one-step HLO changed only modestly:

| metric | baseline | `precomputed_static` |
| --- | ---: | ---: |
| optimized-HLO lines | 2862 | 2831 |
| fusions | 10 | 10 |
| gathers | 182 | 182 |
| selects | 125 | 125 |
| broadcasts | 130 | 129 |
| dominant PCR fusion output | 22 arrays / 3.82 MiB | 22 arrays / 3.82 MiB |
| assembly-side loop fusion output | 5 arrays / 0.87 MiB | 4 arrays / 0.70 MiB |

This supports a narrow hypothesis: if this path is pursued, the runtime
prototype should precompute the remaining static diagonal/RHS bases in the
double-cable scan body and verify real curve-level timing. It should not be
treated as a new solver algorithm, and it does not change the earlier
conclusion that the PCR loop fusions remain the dominant low-level structure.

## Runtime Scan Precompute Check

A narrow runtime prototype was then validated on the real public recruitment
curve path. It precomputes the static double-cable diagonal bases and absolute
background/correction helpers outside the PCR/SoA scan step, without adding a
new public option or solver route.

Artifacts:

- `benchmark/results/kaggle/20260707_153410_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-baseline-gpu/outputs/extracted`
- `benchmark/results/kaggle/20260707_153636_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-prototype-gpu/outputs/extracted`
- `benchmark/results/kaggle/20260707_154008_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-baseline-gpu-r3/outputs/extracted`
- `benchmark/results/kaggle/20260707_154232_recruitment_curves_quick_gpu_NvidiaTeslaP100_axonscope-p11b-runtime-precompute-prototype-gpu-r3/outputs/extracted`

All runs used P100 GPU, fp32, `Naxons=512`, requested `Nx=101`, actual kernel
`Nx=89`, different-diameter observer-only double-cable recruitment,
`pcr_soa`, three amplitudes, and `time_chunk_steps=50`. The R3 run used one
warmup and three repeats.

| metric | baseline R3 | runtime precompute R3 | delta |
| --- | ---: | ---: | ---: |
| all `curve.simulate` | 7651.5 ms | 8139.1 ms | +6.4% |
| warmup `curve.simulate` | 6306.5 ms | 6903.7 ms | +9.5% |
| repeat `curve.simulate` | 1345.0 ms | 1235.4 ms | -8.1% |
| repeat `runtime.prepare` | 204.3 ms | 177.3 ms | -13.2% |
| repeat `kernel.dispatch_jax` | 25.7 ms | 25.7 ms | ~0.0% |
| repeat `kernel.wait` | 99.8 ms | 96.9 ms | -2.9% |

The recruitment curve summaries were identical. The prototype is therefore
acceptable as a small repeated-workload runtime cleanup, but it is not a solver
optimization direction: hot `kernel.dispatch_jax` and the PCR lowering shape do
not materially change, while cold compile/warmup gets worse. Stop this
precompute family here unless a future compiler-level change gives a new
structural reason to revisit it.

The next low-level work should inspect the code generated inside the current
PCR/SoA implementation and the one-step composition around it, not add a new
public or policy route:

1. Drill into the current PCR/SoA HLO/fusion bodies and memory layout.
2. Add a CPU/generated-membrane lowering view for the different-diameter case,
   because CPU stage profiles still show membrane and assembly costs.
3. Prototype solver changes only as benchmark-only candidates with correctness
   gates, then validate in real double-cable curve benchmarks before any
   runtime policy change.
