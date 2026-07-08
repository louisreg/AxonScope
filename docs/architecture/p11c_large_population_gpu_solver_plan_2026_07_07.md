# P11C Large-Population GPU Solver Plan

Date: 2026-07-07

P11C is a go/no-go implementation track for a new exact double-cable GPU
execution family dedicated to large axon populations. It is not a choice among
existing solver options. If the benchmark gates show a real simulation win, the
route can be promoted into the backend. If not, it stays benchmark-only and
AxonScope keeps the current JAX solver policy.

## Problem

AxonScope needs to simulate large double-cable axon populations with GPU
acceleration:

```text
Naxons = 4096, 8192, 16384+
Nx     = short to moderate axon systems, bucketed
Nt     = many time steps
output = observer-only / probe first, full Vm only for validation/debug
```

The current GPU path is exact and useful, but it is still a general JAX
batch-first solver route. P11B evidence showed that small algebraic tweaks,
HLO-size reductions, and simple XLA batch-Thomas scans did not produce the
kind of win needed for large populations.

## Strategy

Build a backend-private large-population execution shape:

```text
prepared GPU layout
    -> packed state and static terms
    -> factorized forcing
    -> dynamic membrane/RHS assembly
    -> exact double-cable solve
    -> compact recording
```

The target is to maximize useful GPU work during the simulation, not only to
win an isolated linear-solve benchmark. Packing, transposes, result assembly,
and host/device boundaries are part of the solver route and must be measured.

## Solver Direction

The ideal long-term solver is a tiled exact block-Thomas/PTA-style route:

```text
[tile, Nx_pad, BLOCK_B]
```

with:

- exact 2x2 double-cable block system;
- strict SoA arrays: `a00`, `a01`, `a10`, `a11`, `off0`, `off1`, `rhs0`,
  `rhs1`;
- forward/backward Thomas over `Nx_pad`;
- massive parallelism over axon tiles;
- electrically neutral padding rows;
- static geometry and axial terms prepared once per execution bucket.

This is different from the existing `thomas_batched_scan`: that route is a
generic JAX/XLA scan over a batch-first input and was rejected as a GPU policy
candidate. P11C is about an execution layout and solver family built for large
GPU populations, not a renamed scan.

Because custom kernels are out of scope for the immediate slice, the first
prototype is JAX-only and benchmark-private. Its job is to validate the
large-population layout, bucket, memory, and staging assumptions. If JAX-only
cannot move the benchmark, P11C should stop instead of spending time on more
micro-variants.

## What Differs From Existing Options

- `thomas`: exact CPU/reference route, not a GPU large-population layout.
- `thomas_batched_scan`: batch-native JAX scan, compact IR, poor P100 timing.
- `pcr_soa_batched`: current exact GPU-oriented JAX route, but no persistent
  large-population layout contract and large PCR fusion state.
- `pcr_adaptive`: policy switch, not a solver.
- PCR probes such as `padded`, `transposed`, `shift`, `symmetric`, and
  `hybrid`: useful negative/diagnostic evidence, not a coherent large-pop
  execution family.

P11C must own the layout before the time loop. If the candidate only calls an
existing solver with a new name, it should be rejected.

## Reused Ideas

Keep from `/ideas`:

- optimize many short independent systems, not one cable;
- use large `B_effective`;
- keep SoA instead of tiny dense 2x2 matrices;
- use backend-owned layouts `[Nx_pad, B]` and `[tile, Nx_pad, BLOCK_B]`;
- precompute geometry, masks, axial coefficients, padding rows, static system
  terms, and recording tables per bucket;
- assemble directly into the solver layout;
- avoid full Vm and dense `Vext` in performance mode;
- keep factorized extracellular drive:

```text
Vext[b, t, x] = footprint[b, x] * waveform[t]
```

Defer for now:

- split/fixed-K iterative routes, because they are not exact unless converged;
- associative transfer, because stability was not proven;
- Triton/Pallas/CUDA FFI/custom kernels, until JAX-only P11C proves that the
  layout direction is worth deeper GPU integration.

## Buckets

Do not restrict the design to `Nx_pad = 64/128`. Benchmark candidate buckets:

```text
32, 48, 64, 80, 96, 128, 160, 192, 256
```

The final retained set should be data-driven. Power-of-two buckets are useful
for PCR-style staging; intermediate buckets reduce wasted padding for
realistic morphologies.

Padding rule:

```text
D   = I_2
L/U = 0
rhs = [0, 0]
```

## P11C-A - Design And Benchmark Harness

Deliverables:

- define `large_population_exact_double_cable_jax` as a benchmark-private
  candidate name;
- define the internal layout vocabulary: `BX`, `XB`, `TILED`;
- define bucket selection and padding metadata;
- define `BLOCK_B` candidates by bucket:

```text
Nx_pad <= 64:  BLOCK_B = 64, 128, 256
Nx_pad <= 128: BLOCK_B = 32, 64, 128
Nx_pad > 128:  BLOCK_B = 16, 32, 64
```

- add synthetic benchmark support for large `B` without public runtime routing;
- include memory estimates, compile time, hot time, node-solves/s, and output
  bytes;
- report whether time is solve, assembly, packing, or recording.

Exit gate:

- local CPU smoke compiles and validates correctness against current exact
  solvers on tiny shapes;
- GPU Kaggle smoke can run one small case without full Vm.

## P11C-B - JAX-Only Prototype

Prototype a benchmark/private route that differs structurally from
`pcr_soa_batched` by owning:

- explicit `Nx_pad`;
- explicit `B` tiling;
- persistent layout selection;
- static-term preparation in the selected layout;
- dynamic RHS/diagonal assembly in the selected layout;
- compact observer/probe-friendly outputs.

Implementation preference:

1. Start with an exact PCR-SoA-like JAX body in the large-pop layout, because it
   exposes parallelism over `B` and `Nx`.
2. Measure whether layout, padding, and static-term staging improve real
   throughput at large `B`.
3. Keep tiled block-Thomas/PTA as the target design, but do not implement a
   fake version if JAX cannot control the needed scratch/occupancy.

Exit gate:

- candidate matches current exact solver on synthetic systems;
- no model-specific branches;
- no public API or `auto` policy change.

## P11C-C - Large-Population Benchmarks

Synthetic solver/layout matrix:

```text
B       = 1024, 4096, 8192, 16384, optional 32768
Nx_true = values mapping across all candidate buckets
dtype   = fp32 first, fp64 spot checks
coeffs  = shared and batched
output  = solver output only, then compact observer/probe proxy
```

Real-stage matrix:

```text
Naxons    = 4096, 8192, optional 16384 if memory allows
Nx target = representative short/moderate MRG-like shapes
recording = observer_only, probe_vm
tsim/dt    = reduced but realistic enough to expose repeated time steps
```

Metrics:

- hot solve time;
- full one-step proxy time;
- simulation throughput;
- compile/first-run time;
- JAX device memory and RSS where available;
- packing/transposes/layout conversion time;
- result/recording time;
- node-solves/s and axon-steps/s;
- correctness against current exact route.

## P11C-D - Decision

Promote only if the candidate shows a meaningful large-population simulation
win, not just a solver-only or HLO counter win.

Promotion requirements:

- large-pop real-stage win on GPU;
- no worse correctness than current exact route;
- memory growth acceptable for `Naxons >= 8192`;
- no hidden host packing/result bottleneck;
- route remains backend-private until API policy is explicitly reviewed.

Reject if:

- JAX-only route does not beat current runtime on real-stage large-pop cases;
- the speedup appears only in synthetic solver-only tests;
- the route mostly shifts cost to packing, compile, or result assembly;
- correctness is fragile near activation/threshold cases.

If rejected, close P11C with evidence and keep the current GPU solver path. The
next serious step after rejection would require a custom kernel track, not more
JAX micro-variants.
