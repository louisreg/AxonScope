# Benchmark Baselines

Baselines are external comparison entry points for benchmark campaigns. They
must not become AxonScope runtime paths, public APIs, or imports from
`src/axonscope`.

## Scope

The first baseline to define is NRV comparison for the two P11A curve scripts:

- activation/block threshold curves;
- recruitment curves.

The adapter should accept the same case vocabulary as `benchmark/run.py` where
that vocabulary makes sense: duration, `dt`, population size, recording mode,
cable/model family, diameter cohort, platform, repeats/warmups, and output
directory. Unsupported AxonScope-only axes must be recorded explicitly instead
of silently approximated.

## Contract

Each baseline run should write the same evidence shape as AxonScope runs:

- `cases.csv`
- `results.csv`
- `curve_summary.csv`
- `events.jsonl`
- `summary.csv`
- `memory_summary.csv`
- `environment.json`
- `metadata.json`
- `manifest.json`

The baseline manifest must record:

- baseline name and version;
- adapter version;
- package/environment metadata;
- git state when the baseline is local code;
- hardware metadata;
- exact input mapping from AxonScope case options to baseline options;
- known semantic differences.

## Non-Goals For P11A

- Do not call NRV from `src/axonscope`.
- Do not mix NRV objects into `AxonSimulation`.
- Do not claim numerical or performance equivalence until the adapter writes a
  fresh artifact directory and the case mapping is reviewed.
- Do not add a NumPy solver baseline until the NumPy solver exists.

