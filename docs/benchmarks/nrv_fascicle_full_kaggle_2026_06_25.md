# NRV Fascicle Full Kaggle Benchmark - 2026-06-25

This run validates the full synthetic NRV/FEM/LIFE recruitment benchmark on a
Kaggle P100 GPU. It uses the reproducible synthetic NRV geometry rather than the
histology-image contour, so the benchmark stresses NRV footprint generation,
NRV validation, AxonScope GPU dispatch, and recruitment sweep scaling without
depending on fragile Gmsh polygon meshing.

## Run

- Kaggle preset: `realistic_fascicle_nrv_gpu_full`
- Benchmark suite: `realistic_fascicle_synthetic_full`
- Code commit: `a45bac8`
- Backend: JAX GPU
- Geometry: one synthetic nerve with four circular fascicles
- Population: 397 simulated fibers
- Amplitudes: 21 sequential current steps
- Timebase: `nt=3000`, `time_chunk_steps=1000`, 3 chunks per group
- Solver groups: 118 double-cable MRG rows and 279 single-cable rows per step

## Summary

| Metric | Value |
| --- | ---: |
| AxonScope cold step | 22.385 s |
| AxonScope warm step | 8.541 s |
| AxonScope 21-step sweep | 190.867 s |
| NRV FEM first footprint | 126.513 s |
| NRV cached footprint sampling | 11.134 s |
| NRV validation, one amplitude | 290.684 s |
| Estimated NRV 21-amplitude sweep | 6104.361 s |
| Estimated NRV / AxonScope sweep | 31.98x |
| Estimated NRV / AxonScope plus footprints | 18.58x |
| Peak RSS | 8298.211 MiB |

The result is full-size Kaggle GPU evidence that AxonScope can sweep a large
NRV-derived recruitment workload much faster than repeating the same population
simulation in NRV. The main non-AxonScope cost remains the first NRV FEM solve
and the one-amplitude NRV validation pass.

## Memory Notes

- Stored current-independent LIFE footprints: 2.37 MB per amplitude step.
- Estimated factorized footprints if amplitudes were batched at once: 49.86 MB.
- Estimated dense full-Vm output for one amplitude: 1.78 GB.
- Estimated dense full-Vm output for all 21 amplitudes at once: 37.40 GB.

This supports keeping amplitude sweeps sequential by default for now. It also
supports treating `time_chunk_steps` and output mode as first-class benchmark
axes before changing protocol defaults.

## Caveats

- `kaggle kernels status` reached `COMPLETE`, and the Kaggle log shows the
  result archive was written.
- `kaggle kernels output` produced an empty `_output_.zip` locally, so the
  durable metrics stored here were reconstructed from the complete Kaggle log.
- The forced-exit hook printed `No stray NRV/FEM processes detected before
  forced exit`; Kaggle still ran notebook HTML conversion afterwards, but the
  final status became `COMPLETE`.

## Next Measurements

- Repeat the full run with `time_chunk_steps` values such as 250, 500, 1000,
  and unchunked to quantify peak memory and chunk overhead.
- Compare observer-only, probe Vm, and full Vm outputs on the same workload.
- Keep `amplitude_batch_size=1` as the default until batching is benchmarked
  against footprint duplication, memory pressure, and warm-step throughput.
