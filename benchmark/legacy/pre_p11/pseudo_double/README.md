# Experimental pseudo-double-cable validation

Status: standby as of 2026-06-16.

This folder is for Phase 7.6.4 validation work. It is not a public solver API.
Exact double-cable remains the reference model. The first validation pass found
useful harness infrastructure but did not produce a physiology-accepted
replacement for exact double-cable, so new optimization work should focus on the
exact double-cable GPU solver unless pseudo modes are explicitly needed as a
high-recall pre-filter.

The electrical-reduction direction is tracked in
`ideas/axonscope_double_to_single_electrical_reduction_plan.md`. The harness now
contains coefficient-level helpers for the first two reduction levels:

- `series_equivalent`: local series admittance/capacitance reduction.
- `schur_local_v1`: diagonal-`App` local Schur complement reduction. It is
  exact when the eliminated periaxonal/myelin block has no spatial
  off-diagonal coupling, and approximate otherwise.

The first harness compares exact MRG double-cable against an opt-in candidate on
the same deterministic population and stimulus sweep. The currently implemented
candidate modes are:

- `mrg_single_cable_surrogate`: the same MRG morphology and membranes forced
  through the existing single-cable extracellular path. It is a baseline
  surrogate, not the final pseudo-double model.
- `pseudo_double_effective`: experimental v0 pseudo-double candidate using the
  single-cable path with a calibrated extracellular coupling multiplier. This
  is physiology-validation plumbing, not a public solver mode.
- `pseudo_double_single_myelinated_chain`: experimental v0 one-voltage
  NODE/MYSA/FLUT/STIN chain built directly from AxonScope single-cable
  primitives. It preserves the MRG-like segment taxonomy, uses active nodes and
  passive effective internodes, and supports a segment-specific extracellular
  alpha vector through typed validation stimulations.
- `pseudo_double_series`: experimental coefficient-derived v1 candidate using
  a local axolemma/myelin RC-series reduction and a scalar tridiagonal solve.
  It keeps node compartments on the axolemmal fallback when the myelin
  capacitance branch is degenerate.
- `pseudo_double_split`: experimental v0 split candidate using the same scalar
  cable path plus a local implicit auxiliary state applied to the extracellular
  waveform. It is a first field-filtered split probe, not a full periaxonal
  spatial solver.
- `pseudo_double_schur_local`: experimental coefficient-derived v1 candidate.
  It derives the local scalar system from the exact double-cable block
  coefficients using diagonal-`App` Schur elimination, then solves one scalar
  tridiagonal system per step.

Run a small local validation:

```bash
python benchmark/pseudo_double/validate.py \
  --candidate mrg_single_cable_surrogate \
  --size 2 \
  --nodes 3 \
  --duration 0.3 \
  --dt 0.05 \
  --amplitudes-uA 20 60 100 \
  --out-dir benchmark/results/pseudo_double/local_smoke
```

Run the experimental effective pseudo-double v0 with a tiny calibration grid
and an automatic baseline comparison:

```bash
python benchmark/pseudo_double/validate.py \
  --candidate pseudo_double_effective \
  --size 1 \
  --nodes 3 \
  --duration 0.5 \
  --dt 0.05 \
  --amplitudes-uA 20 60 100 140 \
  --calibrate-vext-scales 8.0 10.0 12.0 16.0 \
  --out-dir benchmark/results/pseudo_double/effective_smoke
```

Run the single-cable myelinated-chain v0 mode:

```bash
python benchmark/pseudo_double/validate.py \
  --candidate pseudo_double_single_myelinated_chain \
  --size 1 \
  --nodes 3 \
  --duration 0.5 \
  --dt 0.05 \
  --amplitudes-uA 20 60 100 140 \
  --calibrate-vext-scales 1.0 2.0 4.0 8.0 \
  --plots \
  --out-dir benchmark/results/pseudo_double/single_chain_smoke
```

Segment-specific extracellular coupling can be probed with
`--single-chain-alpha-mysa`, `--single-chain-alpha-flut`, and
`--single-chain-alpha-stin`; the default is direct coupling (`alpha=1`) for all
segments.

Run the experimental split pseudo-double v0:

```bash
python benchmark/pseudo_double/validate.py \
  --candidate pseudo_double_split \
  --size 1 \
  --nodes 3 \
  --duration 0.5 \
  --dt 0.05 \
  --amplitudes-uA 20 60 100 140 \
  --split-aux-tau-ms 0.05 \
  --calibrate-vext-scales 4.0 6.0 8.0 10.0 \
  --out-dir benchmark/results/pseudo_double/split_smoke
```

Run the coefficient-derived RC-series v1 mode:

```bash
python benchmark/pseudo_double/validate.py \
  --candidate pseudo_double_series \
  --size 1 \
  --nodes 3 \
  --duration 0.5 \
  --dt 0.05 \
  --amplitudes-uA 20 60 100 140 \
  --series-capacitance-floor-fraction 0.02 \
  --calibrate-vext-scales 1.0 2.0 4.0 8.0 \
  --plots \
  --out-dir benchmark/results/pseudo_double/series_smoke
```

With `--plots`, the harness writes PNGs under `OUT_DIR/plots/`:

- `activation_summary.png`
- `error_summary.png`
- `thresholds.png`
- `timings.png`
- `trace_amp_<A>_row_<R>.png` for selected trace samples

Use `--plot-trace-rows` and `--plot-trace-amplitudes-uA` to limit trace plots
on larger runs.

Run the coefficient-derived Schur-local v1 mode:

```bash
python benchmark/pseudo_double/validate.py \
  --candidate pseudo_double_schur_local \
  --size 1 \
  --nodes 3 \
  --duration 0.5 \
  --dt 0.05 \
  --amplitudes-uA 20 60 100 140 \
  --calibrate-vext-scales 4.0 6.0 8.0 10.0 \
  --out-dir benchmark/results/pseudo_double/schur_local_smoke
```

Outputs:

- `summary.json`: run parameters, mode metadata, per-amplitude summaries,
  threshold estimates, timings, row-level comparisons, optional calibration
  trials, and optional baseline comparison.
- `rows.csv`: flat per-row/per-amplitude comparison table.

Physiology metrics come before speed claims. Track activation agreement, false
negatives, threshold estimates, peak voltage error, activation time error, and
RMS Vm error against exact double-cable. Near-threshold or ambiguous cases
should be rerun with exact double-cable.

Current validation status: `pseudo_double_effective`,
`pseudo_double_single_myelinated_chain`, `pseudo_double_series`,
`pseudo_double_split`, and `pseudo_double_schur_local` can be calibrated on
tiny smoke cases for activation/threshold probes, but peak and RMS trace errors
are still large.
Treat them as experimental rough-screening probes only, not validated
double-cable replacements. `pseudo_double_series` and
`pseudo_double_schur_local` are the first runnable coefficient-derived paths;
the next fidelity pass, if this work resumes, should test them on broader
workloads and add local/periaxonal coupling corrections rather than tuning only
stimulus scale.

Do not add these modes to `BatchOptions.double_cable_block_solver`, the hotpath
`--double-cable-block-solver` choices, or public examples while the work is in
standby.

Planned experimental modes:

- `pseudo_double_modal`

They are registered as validation names but intentionally raise
`NotImplementedError` until their kernels and physiology tests exist.
