# AxonScope Benchmark Surface

P11A resets benchmarking around two canonical curve scripts:

- `benchmark/curves/threshold_curves.py`
- `benchmark/curves/recruitment_curves.py`

Historical scripts, notebooks, reports, and raw outputs live under
`benchmark/legacy/pre_p11/`. They are archive material, not current performance
evidence.

## Commands

Use the shared launcher for local, GPU, and future NRV runs:

```bash
python benchmark/run.py --list
python benchmark/run.py --script threshold_curves --preset quick --platform cpu --dry-run
python benchmark/run.py --script recruitment_curves --preset gpu_smoke --platform gpu --dry-run
python benchmark/run.py --script threshold_curves --preset quick --platform cpu
python benchmark/run.py --script recruitment_curves --preset quick --platform cpu
python benchmark/run.py --script basic_examples --preset quick --platform cpu --examples 06,07,08
python benchmark/run.py --script with_nrv_examples --preset quick --platform cpu --examples 01,02
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script threshold_curves --cpu
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script threshold_curves --preset gpu_smoke --platform gpu --machine-shape NvidiaTeslaP100
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script basic_examples --preset gpu_smoke --platform gpu --machine-shape NvidiaTeslaP100 -- --examples 06,07,08
python benchmark/kaggle/run_kernel.py --username YOUR_KAGGLE_USERNAME --script threshold_curves --preset gpu_trace_smoke --platform gpu --machine-shape NvidiaTeslaP100
```

Both curve scripts use the same option vocabulary:

```bash
python benchmark/run.py \
  --script threshold_curves \
  --preset local_smoke \
  --platform cpu \
  --recording probe_vm \
  --cable single_cable \
  --population single_model \
  --diameters same_diameter \
  --memory-trace all \
  --profile \
  --dry-run
```

`benchmark/membrane_runtime_cache.py` isolates P17 source-cache loading and
`JaxMembraneProgram` construction for the cached Model IR reference path and
the autonomous generated JAX/NumPy path. It also measures first-build and
cache-hit compilation for a labelled composite membrane:

```bash
MPLBACKEND=Agg python benchmark/membrane_runtime_cache.py \
  --models hodgkin_huxley,schild97 \
  --repeats 25 \
  --output benchmark/results/p17_generated_runtime_cache_local/summary.json
```

P17 GPU membrane-layout A/B traces reuse
`benchmark/protocols/recruitment_amplitude_batch.py`. The Naxon=1024 P100
artifacts ending in `axs-p17-membrane-trace-d1024-p100-d8ccff1` and
`axs-p17-static-gates-d1024-p100-cf8adcd` compare the same five-amplitude,
3000-step double-cable workload before and after removing invariant layout
columns from the temporal scan carry. Repeated unprofiled scaling results are
stored in artifacts ending in `axs-p17-sg-time-d{1024,4096}-5623f0a`.
The corresponding single-cable P100 artifacts end in
`axs-p17-sg-single-trace-1024-dd62242` and
`axs-p17-sg-single-time-{1024,4096}-dd62242`; they use the same five-amplitude,
3000-step workload and compare against the P16 baseline artifacts for
Naxon={1024,4096} plus the Naxon=1024 warm trace.
The compiler-IR capture ending in `axs-p17-single-hlo-1024-7e7c7d9` attributes
the remaining single-cable gate, conductance, and assembly fusions before the
generated Triton experiment.
The generated membrane outer-kernel smoke/timing artifacts end in
`axs-p17-generated-membrane-guard-66f9546` and
`axs-p17-generated-membrane-time-n1024-911e5d7`. They retain the canonical
single-cable observer scans and replace generated gate plus `Gm`/`GE` work
behind a guard that rejects JAX fallback. A subsequent diagonal/RHS fusion was
numerically valid but performance-neutral (`-0.14%` median warm run-pool) and
was reverted; its audit artifacts end in `axs-p17-system-smoke-10c14e2` and
`axs-p17-system-time-1024-10c14e2`.
Generated-kernel launch tuning is recorded in artifacts ending in
`axs-p17-mem-b{128w4,256w4,256w8,512w8}-da19b9d`; the complete warm run-pool
spread was only `0.26%`, so the default `256x4` launch remains fixed.
The corresponding generated double-cable membrane route is validated by the
guarded artifact ending in `axs-p17-double-mem-guard-v3-5ea8a2a` and timed by
the artifacts ending in `axs-p17-double-mem-time-{1024,4096}-5ea8a2a`. These
retain the canonical node-first observer scan and fused tiled-Thomas solve,
with exact activation counts and dense-kernel validation.
A later direct same-step fusion of generated membrane equations inside the
Thomas recurrence was rejected. The guarded smoke, Perfetto trace, and
unprofiled Naxon=1024 timing artifacts end in
`axs-p17-double-fused-smoke-08ad0d2`,
`axs-p17-double-fused-trace-1024-08ad0d2`, and
`axs-p17-double-fused-time-1024-08ad0d2`. The standalone membrane launches
disappeared, but the combined kernel was slower than the two retained kernels;
warm run-pool improved by only `0.9%`, the complete sweep regressed by `0.9%`,
and cold compilation worsened. Use these artifacts as evidence against
launch-only inlining, not as a production baseline.

P17B gates a benchmark-only exact scalar tiled-Thomas Triton candidate before
any single-cable runtime change. It compares node-first `[Nx, B]` systems with
the canonical JAX/cuSPARSE solve, validates a subset with dense NumPy, and
reports first-call and warm timings across launch widths:

```bash
python benchmark/run.py \
  --script single_cable_triton_gate \
  --platform gpu \
  --nx 17,63,127,200 \
  --batch-sizes 5120,20480 \
  --block-b 64,128,256 \
  --output benchmark/results/p17b_single_cable_triton_gate
```

This candidate remains under `benchmark/solvers/`. Promote it only by replacing
the canonical internal GPU solve after the numerical gate and the 15% realistic
end-to-end retention threshold pass; do not expose a public solver policy.
The first P100 solve-only artifact ends in
`axs-p17b-single-thomas-gate-fc4ae5c`. For `Nx=200`, Triton-128 improved warm
solve time from `0.773` to `0.450 ms` at `B=5120` (`1.72x`) and from `2.412` to
`0.565 ms` at `B=20480` (`4.27x`), with `1.907e-5` maximum absolute difference.
Its first Triton startup/compile cost was `3.73 s`, so persistent-cache replay
and realistic temporal integration remain mandatory before promotion.

The realistic P100 gate at `Naxon=1024/4096`, `Nx=200`, five amplitudes, and
compact activation also passes the performance threshold. Median warm
`run_pool` improves from `2.202` to `1.790 s` at N=1024 (`1.23x`) and from
`8.448` to `5.288 s` at N=4096 (`1.60x`). At N=4096, median
`kernel.dispatch_jax` falls from `7.344` to `4.321 s`; all five activation
counts match. A focused N=1024 threshold run bounds the one boundary-count
difference at `225 uA` to a transition movement below `0.1 uA`. Fresh-process
persistent-cache replay reduces candidate wall time from `9.71` to `3.64 s`
and lower/JIT time from `3.58` to `0.049 s`. Artifacts end in
`axs-p17b-single-e2e-{base,triton}-{1024,4096}-3da043c`,
`axs-p17b-single-threshold-{base,triton}-3da043c`, and
`axs-p17b-single-cache-replay-1024-3da043c`.
The validation artifact ending in `axs-p17b-single-validation-4010332` covers
`Nx={2,17,63,127,200}` and batch tails `B={1,127,129,513,5123}`. All 25
heterogeneous float32 systems match JAX/cuSPARSE and a float64 dense NumPy
subset within `1.907e-5` absolute error.
The N=1024 persistent-cache artifact also retains the P100 PTX. It confirms
contiguous node-first batch-lane accesses, one 128-thread program per tile,
39 `b32` plus 35 `b64` virtual registers, and no shared/local memory or
`ld.local`/`st.local` spill traffic. The measured gate does not justify cable
assembly fusion during the initial production replacement.

Current real runs support AxonScope point-source activation-threshold and
recruitment curves. `--dry-run` still only writes `cases.csv` for case review.
Real execution writes timing, memory, environment, raw activation rows, and
curve summaries. Block thresholds and NRV execution are intentionally left as
future benchmark/baseline work until their adapter contracts are defined.

`basic_examples` is an executable-docs perf gate for
`examples/basic/06_activation_velocity.py`,
`examples/basic/07_threshold_vs_diameter.py`, and
`examples/basic/08_recruitment_curve_population.py`. It wraps the examples
without changing their public workflow, records one cold run plus optional
warmups/repeats, and writes `runs.csv`, per-run benchmark traces, and
`report.md`.

`benchmark/examples/basic_08_startup.py` is the dedicated construction/startup
probe for the basic-08 workload. It reproduces the workload without importing
or modifying the public example and instruments heavy module imports, Python
population construction, position and footprint generation, stimulation
attachment, protocol setup, and optional first/full sweep execution. The
default stops before the sweep so a 1000-fiber-per-family startup profile stays
focused:

```bash
MPLBACKEND=Agg python benchmark/examples/basic_08_startup.py \
  --fibers-per-family 1000 \
  --scope startup \
  --profile \
  --output benchmark/results/basic_08_startup_local_f1000
```

Use `--template-policy distinct` for the public-example construction baseline
and `--template-policy shared` to measure the proposed lazy lowering: diameter
units are validated and quantized once, unique immutable axon templates are
constructed once, and row-specific positions, footprints, and instances remain
distinct.

Use `--waveform-update-policy callback` to reproduce per-row
`Stimulus`/`Drive`/`ExtracellularStimulation` replacement, or `typed` to use
the production numeric-axis path and one reusable simulation. The typed factory
returns one complete waveform payload for each sampled value; source simulation
descriptions remain immutable, and positive and negative phases need not share
one scale.

Use `--scope first-amplitude` to include dispatch, preparation, JIT, and one
solver call, or `--scope full` for all eight amplitudes. Run timing and
`--profile` cases separately because deterministic profiling substantially
inflates object-construction time.

The exact public basic-08 P100 checkpoint after shared templates and typed
waveform reuse is retained under
`results/kaggle/20260715_225850_basic_examples_gpu_smoke_gpu_NvidiaTeslaP100_axonscope-basic-08-p14-gpu-timing-8e113bf`.
It deliberately uses `--memory-trace off`; per-span JAX device-memory sampling
substantially synchronizes this short workload and must be captured in a
separate diagnostic run.

`with_nrv_examples` is the matching executable-docs gate for
`examples/with_nrv/01_synthetic_fascicle_geometry.py` and
`examples/with_nrv/02_realistic_fascicle_geometry.py`. It preserves the public
NRV-to-AxonScope handoff path but uses smoke-scale defaults unless explicitly
overridden, because NRV/FEM setup dominates the wall time.

The realistic CPU typed-waveform validation is retained in
`results/with_nrv_01_realistic_cpu_20260715_typed_waveform_r1`. It runs example
01 with 193 generated axons, the sampled FEM footprints, 3 ms at 1 us, and 21
sequential amplitudes. Use `runs.csv` for the NRV/AxonScope boundary and
`01/sequential/cold_00/run_pool_detail.csv` for per-amplitude single/double
cable, preparation, enqueue/dispatch, and wait timings.

`recruitment_amplitude_batch` also provides the NRV-independent P14 temporal
solver baseline. The `p14_realistic` workload keeps the AxonScope dimensions
of `with_nrv/01` (196 axons, 5 mm cables, 3 ms at 1 us, and 21 amplitudes from
0 to 300 uA) while replacing NRV geometry/FEM setup with deterministic
analytical footprints. Run single- and double-cable populations separately so
their compilation and execution costs remain attributable:

```bash
python benchmark/run.py \
  --script recruitment_amplitude_batch \
  --platform cpu \
  --workload p14_realistic \
  --cable double \
  --policies 1,2,full \
  --output benchmark/results/p14_double_cpu
```

Each run writes aggregate timings to `runs.csv` and one row per numeric
amplitude chunk and cable mode to `run_pool_detail.csv`, including
`kernel.dispatch_jax`, `kernel.wait`, their combined solver time, and solver
percentages. `protocol.sweep.amplitude_chunk` records compact plan boundaries.
Chunk sizes no longer clone `Namplitude x Naxon` Python rows: compatible
waveforms lower to one current table and one solver invocation per chunk over
shared source descriptions and factorized footprints. Use `--drive-count 2`
to retain an independent nonzero static drive while the typed numeric axis
varies the selected drive. This mode writes
`multi_drive_route_validation.json` and fails if either cable formulation
materializes dense `Vext`; GPU double-cable runs also require the production
`jax_triton_loop_xb` solver. Add
`--profile` only for a dedicated trace run; profiling is off for timing
baselines. Use `--profile-scope run_pool` for large populations so Python
population and dispatch-plan construction cannot fill the trace before device
execution starts. The generic benchmark instrumentation also accepts
`profile_span="simulation.run_pool"` for the same first-matching-span capture.
`inputs.numeric_axis` isolates numeric waveform sampling and pattern
construction from `inputs.extracellular`, which owns the source cohort's
static footprints. Its metadata reports logical/kernel batch sizes, unique
temporal pattern count, cache hits/misses, and current/index payload bytes.

For JAX Perfetto JSON traces, use
`python benchmark/analysis/jax_perfetto_summary.py TRACE --tracks` to list
host/device tracks, or add `--track-pattern device:GPU` for GPU kernels only.
Device durations explain where asynchronous work actually ran; do not interpret
`kernel.wait` alone as solver time or `kernel.dispatch_jax` as pure host
overhead. Profiling perturbs host timings, so retain an unprofiled matching run
for wall-time comparisons.

The P100 multi-drive checkpoint is
`results/kaggle/20260716_120035_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14c-multidrive-p100-400cf49`.
It covers 16 Rattay-Aberham and 16 MRG axons, two distinct point-source
footprints, four amplitudes, and chunk sizes `1/full`. Every phase returned
activation counts `0 18 20 21`; both cable modes stayed factorized, and the
independent double-cable Triton comparison passed at `1.439e-7` maximum absolute
error.

The matching attribution trace is
`results/kaggle/20260716_121220_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14c-dispatch-profile-p100-400cf49`.
Its warm full run recorded `531.8 ms` in `kernel.dispatch_jax`, `0.12 ms` in the
final wait, and `330.9 ms` of events on the serial GPU compute stream. Therefore
dispatch absorbed substantial solver/device execution through JAX backpressure;
it is not a separable non-solver cost. The trace contains 67,675 compute-stream
events, including 3,072 fused double-cable Triton solves (`136.4 ms`) and 3,000
single-cable PCR loop solves (`25.1 ms`) plus their first passes (`9.4 ms`).
This is launch/fusion evidence, not a timing baseline: Perfetto inflated host
preparation and total wall time.

The large-population follow-up is
`results/kaggle/20260716_142325_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14c-run-pool-profile-1024-p100-9725f34-v2`.
It profiles only the canonical `simulation.run_pool` span for 1024 mixed axons,
two drives, and one 300 uA amplitude. The warm trace stays below saturation and
contains 63,811 compute-stream events totaling `898.7 ms`; host
`simulation.run_pool` is `2.757 s`. The non-overlapping host attribution is
`runtime.prepare=1.047 s`, numeric-axis lowering about `0.339 s`, factorized
extracellular preparation `0.266 s`, `kernel.enqueue=1.013 s`, and final
`kernel.wait=68.7 ms`. Inside preparation, deep runtime-signature construction
takes about `0.971 s`, including `0.850 s` in repeated `repr`; this is the next
P14D host target. Trace serialization occurs after the measured run-pool span
and inflates the enclosing sweep, so use the matching unprofiled run for
end-to-end timing.

P14D's strict unprofiled trusted-signature A/B uses
`results/kaggle/20260716_145813_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-signature-baseline-1024-p100-9725f34`
and
`results/kaggle/20260716_145512_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-trusted-signatures-1024-p100-95b1327`.
Both runs return 758 activated axons for the same 1024 mixed axons, two drives,
and one 300 uA amplitude. Warm `runtime.prepare` drops from `914.1` to
`67.8 ms`, `simulation.run_pool` from `2.190` to `1.422 s`, and the complete
sweep from `2.473` to `1.773 s`. Enqueue, dispatch, and wait remain stable;
the gain is preparation reuse. Cold compilation varies between these single
runs, so they are not evidence for a cold solver speedup.

The indexed multi-drive P100 comparison uses
`results/kaggle/20260716_152802_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-indexed-baseline-1024-p100-95b1327`
and the retained compact-host run
`results/kaggle/20260716_153356_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-compact-host-1024-p100-73aadd8`.
Both return `0 565 637 707 758` for 1024 mixed axons, two drives, and five
amplitudes. Each cable group now carries five unique `[S, Nt]` current patterns
(`120 KB`) plus row indices instead of 2560 repeated patterns (`61.44 MB`).
Warm `simulation.run_pool` is `4.545 s` before and `4.440 s` after; dispatch is
stable within about 0.5%. A rejected intermediate run ending in
`axs-p14d-indexed-multidrive-1024-p100-e432bb7` fused the gather into the large
double-cable JIT and raised cold double dispatch from `7.02 s` to `10.09 s`.
The retained route therefore keeps the compact host representation while
preserving the canonical solver executable.

The follow-up P100 run
`results/kaggle/20260716_162034_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-deferred-current-1024-p100-31bd6d6`
stops sampling the base extracellular currents when a typed numeric axis will
immediately replace them. It returns the same `0 565 637 707 758` activations.
Against the retained compact-host run, warm `inputs.extracellular` decreases
from `438.1` to `380.1 ms` and `simulation.run_pool` from `4.440` to `4.417 s`;
dispatch (`3.512` versus `3.498 s`) and wait (`223.9` versus `223.8 ms`) are
stable. The removed `current_scaled_shared_waveform` spans totaled `132.7 ms`;
the net input reduction is smaller because footprint construction varied in
the opposite direction. A separate prototype ending in
`axs-p14d-footprint-reuse-1024-p100-8b7cc66` is rejected evidence: hashing full
footprint arrays cost `378.6 ms` and raised `inputs.extracellular` to
`530.9 ms`.

The retained spatial-axis follow-up is
`results/kaggle/20260716_163340_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-compact-spatial-1024-p100-5f1e37c`.
For each cable group it samples `512` source footprints once and expands them
to the trusted `2560`-row amplitude-major runtime shape. Against the preceding
deferred-current run, warm footprint compute drops from `291.4` to `53.5 ms`,
footprint-key construction from `37.8` to `6.5 ms`,
`inputs.extracellular` from `380.1` to `99.1 ms`, and
`simulation.run_pool` from `4.417` to `4.099 s`. Dispatch remains
`3.50-3.52 s`, wait remains about `224 ms`, and activation counts remain
exactly `0 565 637 707 758`. This is source-row spatial preparation reuse
inside one numeric-axis run, not yet persistent prepared-plan reuse across
separate calls.

The dispatch-plan follow-up compares
`results/kaggle/20260716_210823_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-build-plan-r3-pin-1024-p100-0c060eb`
with
`results/kaggle/20260716_212310_recruitment_amplitude_batch_gpu_smoke_gpu_NvidiaTeslaP100_axs-p14d-build-plan-reuse-r3-1024-p100-fae6750`.
Both use JAX/JAXlib 0.10.2 and return exactly `0 565 637 707 758` in every
repeat. Reusing one prepared row signature for cache lookup and normalization
reduces median warm `dispatch.build_plan` from `397.3` to `216.4 ms` (`-45.5%`)
while median `simulation.run_pool` remains `4.12/4.11 s`. In the optimized
run, median cache-key construction is `11.4 ms`; normalization is `128.6 ms`,
grouping `60.1 ms`, and group materialization `13.0 ms`. The Kaggle installer
now derives the CUDA JAX requirement from `pyproject.toml`; the earlier run
ending in `axs-p14d-build-plan-r3-1024-p100-5f1e37c`, which silently upgraded
to JAX 0.11.0, is rejected as incomparable evidence.

The P14E solver-bound scaling matrix uses five amplitudes
`0,75,150,225,300 uA`, full amplitude batching, `3 ms` at `1 us`, two drives,
and `Naxon={196,1024,4096}` separately for each cable formulation. The corrected
P100 artifacts end in
`axs-p14e-source-reuse-{single,double}-{196,1024,4096}-p100-f46bbec`. They
construct one source population per policy and reuse that exact object for
cold and warm execution:

| cable | Naxon | source build | cold run_pool | one-shot cold | warm run_pool | solver share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single | 196 | 0.303 s | 3.784 s | 4.345 s | 0.988 s | 97.0% |
| single | 1024 | 1.221 s | 8.786 s | 10.504 s | 4.774 s | 97.4% |
| single | 4096 | 5.648 s | 26.832 s | 33.852 s | 18.706 s | 96.3% |
| double | 196 | 0.376 s | 8.225 s | 8.830 s | 0.649 s | 96.2% |
| double | 1024 | 1.708 s | 10.731 s | 12.770 s | 2.666 s | 95.8% |
| double | 4096 | 6.912 s | 18.188 s | 25.791 s | 9.971 s | 95.8% |

The P15 compact-activation checkpoint reruns this exact matrix at commit
`375fe59`, with a benchmark guard requiring `output_sink=activation` and
`observer=activation` on every dispatch. The P100 artifacts end in
`axs-p15-compact-{single,double}-{196,1024,4096}-p100-375fe59`:

| cable | Naxon | cold run_pool | warm run_pool | P14E warm | warm speedup | warm dispatch_jax | warm wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single | 196 | 3.151 s | 0.512 s | 0.988 s | 1.93x | 0.429 s | 0.023 s |
| single | 1024 | 5.920 s | 2.346 s | 4.774 s | 2.04x | 2.057 s | 0.123 s |
| single | 4096 | 15.900 s | 9.066 s | 18.706 s | 2.06x | 7.863 s | 0.470 s |
| double | 196 | 11.957 s | 0.523 s | 0.649 s | 1.24x | 0.391 s | 0.050 s |
| double | 1024 | 12.448 s | 1.875 s | 2.666 s | 1.42x | 1.494 s | 0.180 s |
| double | 4096 | 16.391 s | 6.576 s | 9.971 s | 1.52x | 5.468 s | 0.487 s |

All cold/warm rows match their same-backend activation reference exactly.
The compact state is `[5 * Naxon, 1]` boolean for this five-amplitude matrix:
`0.96 KiB`, `5 KiB`, and `20 KiB` at the three population sizes. At the
21-amplitude target workload, 4096 axons retain about `84 KiB`, rather than
the roughly `6.02 GiB` full single-cable threshold raster. Warm improvements
are stable and scale with population size. Small double-cable cold compilation
remains noisy (`0.69x/0.86x` at 196/1024), while double 4096 improves `1.11x`;
do not interpret the small cold rows as steady-state regressions.

The fresh P16 baseline at commit `d1313b9` repeats the same guarded P100
workload after P15 completion. Each case has one warmup and two measured warm
runs; the table reports their median `simulation.run_pool` time. Artifacts end
in `axs-p16-baseline-{single,double}-{196,1024,4096}-p100-d1313b9`:

| cable | Naxon | cold run_pool | median warm run_pool | warm solver share | activation counts |
| --- | ---: | ---: | ---: | ---: | --- |
| single | 196 | 3.299 s | 0.542 s | 93.9% | `0 17 48 67 89` |
| single | 1024 | 6.622 s | 2.359 s | 93.4% | `0 85 197 326 433` |
| single | 4096 | 15.590 s | 8.920 s | 94.5% | `2 383 878 1340 1768` |
| double | 196 | 11.519 s | 0.517 s | 95.1% | `0 196 196 196 196` |
| double | 1024 | 11.308 s | 1.870 s | 93.2% | `0 1024 1024 1024 1024` |
| double | 4096 | 18.101 s | 6.703 s | 91.9% | `0 4096 4096 4096 4096` |

Every cold, warmup, and warm row matches its same-process activation reference.
The current double-drive workload saturates the double-cable population above
zero current; retain these counts for route and timing equivalence, not as a
graded recruitment-curve validation. Local CPU N=196 confirms that full
five-amplitude batching is about `1.8x` faster for single cable and `2.1x`
faster for double cable than five unit chunks. P16 therefore uses full batching
as its temporal-program baseline and profiles chunk policies separately.

Use `--capture-jit-phases` only for a dedicated compiler diagnostic run. It
captures the first production compact-factorized JIT for each selected cable,
writes trace/lower/compile/first-execution timings and stable identities under
`jax_phase_capture/*.jit_phases.json`, and retains both StableHLO and compiled
optimized HLO. Summarize the latter with:

```bash
python benchmark/analysis/hlo_fusion_summary.py \
  OUTPUT/jax_phase_capture \
  --output OUTPUT/hlo_summary
```

The capture compiles and executes explicitly, so its wall timings are compiler
diagnostics rather than clean baseline measurements.

Warm P100 Perfetto traces at N=1024 decompose the actual 3000-step device
program. The single-cable stream spends about 51% in the two cuSPARSE GTSV
kernels, 18% updating gates, 14% assembling the cable system, 11% evaluating
membrane terms, and the remainder in the compact observer and scan
bookkeeping. The double-cable stream spends only about 19% in the tiled-Thomas
Triton kernel; gate reconstruction, batch/node layout transforms, system
assembly, and device copies dominate the rest. Artifacts end in
`axs-p16-warmtrace-{single,double}-1024-p100-7648bbc`.

A rejected P16 experiment replaced the stacked gated/leak gate concatenation
with `dynamic_update_slice`. Although numerically exact, optimized HLO changed
the gate carry from the favorable batch-first layout to `{gate,batch,node}` and
introduced a full transpose in every step. Double-cable N=1024 median warm
runtime regressed from about `1.87 s` to `2.62 s`; commit `7112d22` removes the
candidate. Future layout work must measure the whole solver interval and keep
one coherent state layout across membrane evaluation and the Triton solve.

The retained node-first double-cable scan at commit `ac542e4` instead keeps
`Vi`, `Ve`, and gate state node-first whenever the compiled membrane backend
advertises model-agnostic node-first batch support. At N=1024, median warm
`simulation.run_pool` improves from about `1.870 s` to `1.508 s` (`-19.4%`)
and total sweep wall from `2.065 s` to `1.682 s` (`-18.5%`), with the same
`0 1024 1024 1024 1024` activation counts. At N=4096, median warm run-pool
time improves from `6.703 s` to `5.077 s` (`-24.3%`) and cold run-pool time
from `18.101 s` to `14.678 s` (`-18.9%`), with exact
`0 4096 4096 4096 4096` counts. The matching N=1024 warm Perfetto stream falls
from about `1.76 s` to `1.39 s` (`-21%`). Five per-step layout kernels totaling
about `639 ms` disappear. The remaining device work is approximately `430 ms`
system/physical-term assembly, `340 ms` tiled Thomas, `330 ms` membrane/gate
reconstruction, `161 ms` observer reductions, and `122 ms` device copies.
Artifacts end in
`axs-p16-node-first-double-1024-p100-ac542e4` and
`axs-p16-nftrace-d1024-p100-ac542e4`, with the large timing run ending in
`axs-p16-nf-d4096-p100-ac542e4`.

Commit `ff80df8` then replaces the production
`materialized coefficients/RHS -> Triton` boundary with one model-agnostic
physical-term custom call. It forms each 2x2 block in Triton registers before
Thomas elimination and removes the measured `430 ms` JAX assembly fusion. At
N=1024, median warm `simulation.run_pool` is `1.378 s`, `8.6%` below node-first
alone and `26.3%` below the P16 baseline. At N=4096 it is `4.945 s`, `2.6%`
below node-first alone and `26.2%` below baseline; the non-overlapping solver
interval improves by about `8.9/6.8%` over node-first. The first Triton miss is
more expensive (`4.98 s` compilation versus `3.70 s`), and cold run-pool time
regresses `5.5%` at N=4096 versus node-first while remaining `14.4%` faster than
the P16 baseline. Direct P100 validation against independent dense NumPy solves
passes at `Nx=22`, batch 7, with max absolute/scaled errors
`8.82e-7/1.96e-7`. Artifacts end in
`axs-p16-fusedasm-d{1024,4096}-p100-ff80df8`,
`axs-p16-fusedtrace-d1024-p100-ff80df8`, and
`axs-p16-fusedvalidate-p100-ff80df8`.

The matching single-cable scan-order experiment did not earn a production
change. Replacing `vmap(scan(step))` with `scan(vmap(step))` preserved exact
activation counts but regressed median warm P100 runtime by `3.1%` at N=1024
and `2.9%` at N=4096. Commit `7483737` restores the original route. Optimized
HLO and Perfetto already place about 51% of its device time in cuSPARSE, so P16
does not add an unproven second scalar solver.

Commit `f43d7d1` validates persistent compilation replay as an exact structural
cache, not a value cache. On P100 double cable, a fresh process with changed
amplitudes, footprints, waveform values, and parameter rows produces identical
StableHLO and adds zero XLA cache files. The captured cold phase falls from
`6.034 s` on a miss to `0.518 s` for exact replay and `0.520 s` for changed
dynamic values (`11.6x`); Triton lowering falls from `3.906 s` to `0.122 s`.
The isolated cache occupies about `720 KiB` for XLA and `24 KiB` for Triton.
Local CPU single cable likewise reuses identical StableHLO with changed values,
reducing captured cold work from `1.007 s` to `0.641 s` (`1.57x`). Artifacts
end in `axs-p16-dynamic-cache-double-p100-f43d7d1` and
`p16_compilation_cache_single_196_cpu_local_20260718_dynamic`.

The final compact-observer time-chunk matrix uses 3000 steps (`15 ms` at
`dt=0.005 ms`). On local CPU N=196, single cable varies by only about `3.6%`
between chunk 512 and unchunked, while double varies by about `0.6%`; these
small laptop differences do not justify a policy change. On P100 N=1024,
double cable's best policy improves `curve.simulate` by only `3.3%` over chunk
128. Single cable is more sensitive: unchunked is `1.403 s` versus `1.787 s`
at 128 and `1.531 s` at the current 512 default. The result does not generalize:
at N=4096, chunk 512 is fastest at `3.393 s`, ahead of unchunked (`3.569 s`),
128 (`3.625 s`), and 1024 (`3.635 s`). The separate memory run shows only
about `0.34 MiB` additional JAX bytes for unchunked/1024 versus 128. P16 keeps
the single global default at 512 rather than adding an adaptive specialization.
Artifacts end in `p16_time_chunk_compact_196_cpu_local_20260718_v3`,
`axs-p16-timechunk-realistic-1024-p100-e0bde84`, and
`axs-p16-timechunk-single-4096-{clean-,}p100-e0bde84`.

`one-shot cold` includes reusable source construction; `cold run_pool` and
`warm run_pool` do not. Solver share is the non-overlapping
`(kernel.enqueue + kernel.wait) / simulation.run_pool`; `dispatch_jax` is
nested in enqueue. Compared with the matching local CPU matrix, warm P100
run-pool speedups are about `18.0/26.3/29.2x` for single cable and
`21.7/35.0/46.9x` for double cable. Local CPU warm solver share is
`99.6-99.9%`. The P100 routes are the factorized single-cable JAX tridiagonal
solver and guarded double-cable `jax_triton_loop_xb`; local double cable uses
Thomas.

The corrected local single-N=1024 boundary artifact is
`results/p14e_source_reuse_single_1024_cpu_local_20260716`. Cold and warm both
return `0 85 197 325 433`; P100 cold and warm both return
`0 85 197 326 433`. The one-fiber difference at 225 uA is a cross-backend
near-threshold boundary within the documented tolerance, while every
same-backend cold/warm comparison is exact. Full native amplitude batching is
one solver execution over the numeric axis, so there is deliberately no
separate Python-call timing for each amplitude.

N=4096 memory diagnostics are separate from timing baselines. Artifacts ending
in `axs-p14e-device-memory-{single,double}-4096-p100-f46bbec` report actual
peak JAX bytes in use of `5.84 GB` single and `2.63 GB` double. Matching
`axs-p14e-rss-{single,double}-4096-p100-f46bbec` artifacts report peak host RSS
of `1.87/1.93 GiB`. The `12475 MiB` process allocation visible through
`nvidia-smi` is JAX's default GPU preallocation, not active benchmark state.
Do not use `--memory-trace all` for this large workload: Python tracemalloc
makes it impractically slow. Collect `device` and `rss` diagnostics separately.

`kernel.dispatch_jax` is nested inside `kernel.enqueue`; do not add the two
when reading the CSV. `kernel_solver_ms` is the non-overlapping
`kernel.enqueue + kernel.wait` pipeline because enqueue can execute deferred
JAX work through queue backpressure, particularly on CPU. The separate dispatch
column remains useful for locating Python-to-JAX call overhead.

The realistic workload canonicalizes diameters and shares exact single- and
double-cable axon templates by default, matching basic 08. Use
`--axon-template-policy distinct` only for the P14 population-construction A/B;
it deliberately restores per-row template construction inside this benchmark
and is not a second production execution path.

Use `--mrg-template-count N` to vary exact MRG geometry diversity while keeping
the realistic diameter set fixed at `7.3/10.0/12.8 um`. Values above three add
intrinsic node shifts, which isolates the cost of shifted cable layouts from
membrane-model diversity.

Use `--time-chunk-steps default` or omit the option to keep AxonScope's
recording-specific default; for observer-only runs this currently means the
stable VmRaster default. Use `--time-chunk-steps unchunked` or `none` to force
one full scan, and use an integer such as `--time-chunk-steps 500` for an
explicit local chunk size. Benchmark artifacts record both `time_chunk_policy`
and `time_chunk_steps` so default, unchunked, and explicit one-chunk runs can be
compared without ambiguity.

Kaggle runs use `benchmark/kaggle/run_kernel.py`, which packages a script
kernel around the same `benchmark/run.py` command, forwards extra options, and
downloads a zipped result directory after success. Use `--cpu` or `--platform
cpu` without `--machine-shape` for a CPU-only Kaggle run. Use `--platform cpu
--machine-shape NvidiaTeslaP100` when you deliberately want the CPU benchmark
path on a Kaggle GPU machine for closer CPU/GPU environment comparisons.

Low-level solver gates use standalone campaigns. P11C's large-population
double-cable solver gate is intentionally benchmark-private and does not change
runtime policy:

Workflow-level solver-policy checks should use the dedicated double-cable
campaign. It compares typed public solver policies through the curve workloads
and writes one summary/report for policy decisions:

```bash
python benchmark/campaigns/double_cable_solver_policy.py \
  --preset quick \
  --platform cpu \
  --curve-script threshold_curves,recruitment_curves \
  --solver auto,thomas \
  --recording observer_only,probe_vm \
  --n-axons 1,64 \
  --nx 89 \
  --precision fp32 \
  --repeats 2 \
  --warmups 1 \
  --output benchmark/results/p11c_solver_policy_cpu
```

Use the matching GPU/Kaggle run for the public GPU solver surface:

```bash
python benchmark/kaggle/run_kernel.py \
  --username YOUR_KAGGLE_USERNAME \
  --slug axonscope-p11c-solver-policy-gpu \
  --campaign double_cable_solver_policy \
  --preset gpu_smoke \
  --platform gpu \
  --machine-shape NvidiaTeslaP100 \
  --curve-script threshold_curves,recruitment_curves \
  --solver auto,tiled_thomas \
  --recording observer_only,probe_vm \
  --n-axons 64,1024,4096,8192 \
  --nx 89,129 \
  --precision fp32 \
  --tiled-thomas-block-b 32,64 \
  --repeats 3 \
  --warmups 1 \
  --memory-trace rss \
  --memory-top-n 0
```

For a small workflow-level Triton smoke, use the same typed solver policy
surface as the larger policy campaign:

```bash
python benchmark/run.py \
  --script recruitment_curves \
  --preset quick \
  --platform gpu \
  --cable double_cable \
  --recording observer_only \
  --double-cable-block-solver tiled_thomas \
  --tiled-thomas-block-b 64 \
  --output benchmark/results/p11c_tiled_thomas_smoke
```

The old P11B/P11C low-level PCR and large-population solver exploration
scripts are historical analysis aids, not active benchmark entry points. Use
the curve scripts and solver-policy campaigns above for current validation.

## Presets

Presets live in `benchmark/workloads/curve_options.py`:

- `quick`
- `local_smoke`
- `local_realistic`
- `cpu_publication`
- `gpu_smoke`
- `gpu_trace_smoke`
- `gpu_realistic`
- `nrv_smoke`
- `nrv_full`

They define scale and defaults for repeats, warmups, duration, `dt`, `Nx`,
`Naxons`, precision, recording mode, platform, memory tracing, profiling,
threshold iterations, and recruitment amplitude count.

`gpu_smoke` is a short GPU functional smoke with lightweight RSS tracing. It
should not enable whole-session JAX tracing, device memory tracing, or
device-memory pprof capture by default. Use `gpu_trace_smoke` when you
explicitly want tracing: it is intentionally limited to one small pool and two
or three amplitude evaluations so Perfetto/XPlane artifacts stay inspectable.

Device-memory pprof capture is stage-filtered. Curve scripts default to
`kernel.wait`; pass `--jax-device-memory-profile-stage runtime.prepare` or
repeat the flag to capture more stages. Use
`--jax-device-memory-profile-stage all` only on tiny trace cases.

FP64 runs require a JAX process with x64 enabled before importing JAX. For a
fresh shell, use the project environment and set `JAX_ENABLE_X64=1` before
running an FP64 preset.

## Outputs

Every real run should write a self-contained result directory:

- `environment.json`: machine, OS, Python, package, git, backend, CPU/GPU/RAM,
  precision, execution, recording, observer, cache, and NRV metadata.
- `cases.csv`: the exact benchmark cases requested.
- `events.jsonl`: stage-level wall-clock events.
- `summary.csv`: aggregate stage timing.
- `memory_summary.csv`: RSS, `tracemalloc`, device-memory, and profile summary.
- `artifacts/`: raw traces, device-memory profiles, and debug outputs.
- `plots/`: generated figures for accepted campaign outputs.
- `results.csv`: row-level activation observations for each tested amplitude.
- `curve_summary.csv`: threshold or recruitment summaries.
- `manifest.json`: the selected script, case name, options, and output map.

Do not make speed or memory claims from console output alone. Use a fresh result
directory with git metadata and saved traces.

## Instrumentation

For scripts, prefer the context-manager style:

```python
import axonscope as axs

with axs.benchmark(
    "benchmark/results/example",
    print_summary=False,
    sync_device=True,
    record_shapes=True,
    memory_trace="all",
    memory_top_n=10,
    profile=True,
    profile_runtime="jax",
    jax_device_memory_profile=True,
):
    result = axs.AxonSimulation(...).run()
```

Keep that heavy `memory_trace="all"` style for tiny diagnostic runs. Use
`memory_trace="off"` or `"rss"` when the timing itself is the signal.

For notebooks and debugging, use the explicit enable/disable style:

```python
import axonscope as axs

session = axs.enable_benchmark(
    "benchmark/results/notebook_debug",
    print_summary=False,
    memory_trace="rss",
    profile=True,
    profile_runtime="auto",
)
try:
    result = axs.AxonSimulation(...).run()
finally:
    report = axs.disable_benchmark(print_summary=True)
```

Use explicit instrumentation imports around non-solver preparation,
post-processing, or external baseline work:

```python
from axonscope.benchmarking import benchmark_span, record_benchmark_metadata

with benchmark_span("stage.name"):
    record_benchmark_metadata(case="example")
```

A standalone teaching script shows the same instrumentation around one normal
AxonScope simulation and writes timing/memory plots:

```bash
MPLBACKEND=Agg python benchmark/examples/runtime_benchmarking_options.py
```

## Trace Analysis

Summarize saved events and trace/profile artifacts with:

```bash
python benchmark/analysis/trace_summary.py benchmark/results/example
```

JAX profiler traces are TensorBoard/Perfetto artifacts. JAX device-memory
profiles are pprof artifacts; open them with `pprof --web <profile.prof>`.

## P11B Cold-Path Audit

Before changing solver routes or scheduling, turn fresh curve outputs into a
stage-level timing and memory map:

```bash
python benchmark/analysis/cold_path_audit.py \
  benchmark/results/p11b_baseline/threshold_large_cpu_7ebe7c3 \
  benchmark/results/p11b_baseline/recruitment_large_cpu_7ebe7c3 \
  --output benchmark/results/p11b_baseline/cold_path_cpu_audit_7ebe7c3
```

The audit writes:

- `cold_path_stage_rows.csv`: one row per benchmark span with timing, RSS,
  `tracemalloc`, device-memory, environment, git, and case metadata.
- `cold_path_group_summary.csv`: grouped P11B view for pool build, dispatch,
  runtime preparation, input lowering, kernel, and result assembly.
- `plots/cold_path_group_time.png`, `plots/cold_path_top_stages.png`, and
  `plots/cold_path_memory.png`.

Use `memory_trace=off` or `rss` for timing-focused large local/GPU sweeps.
Keep `device`, `all`, JAX profiling, and device-memory pprof capture for tiny
trace cases only. Device memory tracing samples JAX memory stats and
`nvidia-smi` around spans, so it can visibly perturb fine GPU timing.

For optimization triage, rank nested event spans by exclusive self time:

```bash
python benchmark/analysis/bottleneck_report.py \
  benchmark/results/p11b_baseline/threshold_n1000_cpu_scout_f895a03 \
  benchmark/results/p11b_baseline/recruitment_n1000_cpu_scout_f895a03 \
  --phase repeat \
  --output benchmark/results/p11b_baseline/bottleneck_n1000_current
```

The bottleneck report writes event-level rows, stage/group rankings, cache
signals, memory context, and a Markdown summary. Use `--phase repeat` for
hot-path solver triage. The report is a triage artifact, not a benchmark claim
by itself.

For time-chunk policy triage, use the campaign runner instead of hand-written
loops:

```bash
python benchmark/campaigns/time_chunk_sweep.py \
  --script recruitment_curves \
  --preset quick \
  --platform cpu \
  --policies default,unchunked,50,250,500,1000 \
  --recordings full_vm,probe_vm,observer_only \
  --memory-trace rss
```

It writes separate raw result directories per policy and a merged summary of
observed chunk metadata plus kernel, observer, Vm-materialization, and
result-assembly timings.

To turn multiple CPU/GPU time-chunk campaigns into bottleneck plots, use:

```bash
python benchmark/analysis/time_chunk_matrix_report.py \
  --run threshold_cpu=benchmark/results/kaggle/<threshold-cpu>/outputs/extracted_cpu \
  --run threshold_gpu=benchmark/results/kaggle/<threshold-gpu>/outputs/extracted_gpu \
  --run recruitment_cpu=benchmark/results/kaggle/<recruitment-cpu>/outputs/extracted_cpu \
  --run recruitment_gpu=benchmark/results/kaggle/<recruitment-gpu>/outputs/extracted_gpu \
  --output benchmark/results/p11b_time_chunk_matrix_report
```

The matrix report writes normalized rows, best-policy rows, heatmaps, CPU/GPU
speedup plots, exclusive pipeline-group stage plots, kernel/result sub-stage
plots, and separate CPU RSS, GPU JAX-device, and GPU `nvidia-smi` memory plots.

Older P11B/P11C solver-stage, lowering, PCR-state, and large-population
analysis scripts remain historical references only. They are no longer Kaggle
campaigns and should not be used for current runtime policy decisions. Current
performance claims should come from the curve scripts, policy campaigns, and
fresh artifact directories with git metadata.

## Publishability

A benchmark result is publishable only if the run directory contains the full
case list, fresh environment/git metadata, timing traces, memory traces, and
the exact script/preset/options used. GPU claims need either local GPU metadata
or Kaggle metadata from the future P11A Kaggle runner. NRV comparisons wait for
the baseline adapter contract in `benchmark/baselines/`.
