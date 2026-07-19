# P18 NRV Membrane Model Audit

This audit compares the built-in AxonScope membrane and axon templates with
the official NRV source at commit
`7dc11954bfa69291cd5853dd99a99de12f106bef`. The local numerical campaign uses
NRV 1.4.0.

The NRV reference surface is:

- `nrv/nmod/_axons.py` for the supported model names;
- `nrv/nmod/_unmyelinated.py` for mechanism assembly and effective defaults;
- `nrv/nmod/_myelinated.py` for MRG, Gaines, and Markov assembly;
- `nrv/_misc/mods/` for the mechanism equations.

## Validation Levels

The audit keeps four different claims separate:

1. **Source mapped**: every NRV mechanism and wrapper default has an identified
   AxonScope owner.
2. **Static checked**: equations, parameters, states, temperature factors, and
   mechanism composition were compared with the source.
3. **Integrated checked**: cable propagation or extracellular behavior was
   compared with a fresh NRV run.
4. **Observable checked**: membrane voltage, individual currents, gates, and
   state trajectories were compared directly.

Passing an integrated campaign does not by itself prove every internal
observable. The dormant dense-observable campaign must be restored before the
fourth claim is made.

## Existing Model Status

| AxonScope model | NRV source | Current status |
| --- | --- | --- |
| Passive | NEURON `pas` | Source mapped; dedicated passive numerical campaign passes. |
| Hodgkin-Huxley | NEURON `hh` plus `pas` | Defaults corrected to NRV's 32 degC template and additional `pas` leak. Integrated velocity campaign passes. |
| Rattay-Aberham | `RattayAberham.mod` plus `pas` | Sodium reversal corrected from 50 to 45 mV. Integrated velocity campaign passes. |
| Sundt | `nahh.mod`, `kdr.mod`, and `pas` | Sodium and leak reversals corrected to 50 and -60 mV. Integrated velocity campaign passes. |
| Tigerholm | Tigerholm mechanism set in `_unmyelinated.py` | Stateful current and reversal-term execution repaired. Integrated velocity campaign passes. Fine current/state comparison remains. |
| Schild 1994 | Schild mechanism set with `naf.mod` and `nas.mod` | Stateful current and reversal-term execution repaired. Integrated velocity campaign passes. Fine current/state comparison remains. |
| Schild 1997 | Schild mechanism set with `naf97mean.mod` and `nas97mean.mod` | Stateful current and reversal-term execution repaired. Integrated velocity campaign passes. Fine current/state comparison remains. |
| MRG / AxNode | `AXNODE.mod` plus the MRG cable assembly | AxNode formulas and defaults match the MOD source. Fresh morphology, compartment, node-delay, passive, and velocity campaigns pass. |

The runtime defect found during this audit was model agnostic. Stateful models
passed their state to `prepare_membrane_step()` and `finalize_membrane_step()`
but not to the current and membrane conductance/reversal evaluations. This made
Tigerholm and both Schild models fail before a solve and would have used stale
reversal terms even after a narrow crash-only fix. The canonical single-cable
scan now forwards the same state tuple through all membrane evaluations.

## Missing NRV Families

NRV also exposes:

- Gaines motor and sensory myelinated models;
- Markov Nav1.1 and Nav1.6 node mechanisms used as optional MRG node
  replacements.

These families have no retained AxonScope implementation yet. They remain P18
implementation work and must enter through the same membrane-source compiler
and generated runtime as the existing models.

## Fresh Evidence

After the runtime and default corrections:

```text
tests/unit/membranes/test_nrv_model_defaults.py
tests/unit/solvers/test_single_row_batch_runtime_path.py
11 passed

tests/unit
747 passed, 1 skipped

tests/nrv/velocity_vs_diameter/test_velocity_systematic_vs_nrv.py
7 passed

tests/nrv/numerics/test_passive_vs_nrv.py
tests/nrv/numerics/test_mrg_morphology_vs_nrv.py
tests/nrv/numerics/test_mrg_compartment_geometry_vs_nrv.py
tests/nrv/numerics/test_mrg_node_delay_vs_nrv.py
86 passed
```

Both the intracellular and extracellular systematic suites currently skip all
seven models with the reason `dense observable NRV comparison awaits
batch-native observables` (`14 skipped` across the two suites). Restoring those
campaigns is the next existing-model audit task.
