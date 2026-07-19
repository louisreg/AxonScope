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
observable. The intracellular and extracellular dense-observable campaigns are
therefore run separately from the velocity campaign.

## Existing Model Status

| AxonScope model | NRV source | Current status |
| --- | --- | --- |
| Passive | NEURON `pas` | The configured NRV mechanism and AxonScope recording satisfy the same `g_pas * (Vm - e_pas)` current definition; cable numerics remain covered separately. |
| Hodgkin-Huxley | NEURON `hh` plus `pas` | Defaults corrected to NRV's 32 degC template and additional `pas` leak. Velocity and detailed Vm/current/gate campaigns pass. |
| Rattay-Aberham | `RattayAberham.mod` plus `pas` | Sodium reversal corrected from 50 to 45 mV and gate updates aligned with NEURON `cnexp`. Velocity and detailed Vm/current/gate campaigns pass. |
| Sundt | `nahh.mod`, `kdr.mod`, and `pas` | Sodium and leak reversals corrected to 50 and -60 mV. Velocity and detailed Vm/current/gate campaigns pass. |
| Tigerholm | Tigerholm mechanism set in `_unmyelinated.py` | Stateful execution, dynamic Nernst reversals, the HCN Na/K split, Na/K pump stoichiometry, concentration budgets, a float32-stable slow-inactivation rate, and NEURON `cnexp` gate ordering are aligned. Velocity and detailed Vm/current/gate campaigns pass. |
| Schild 1994 | Schild mechanism set with `naf.mod` and `nas.mod` | Stateful current and reversal-term execution repaired. Velocity and detailed Vm/current/gate/state campaigns pass. |
| Schild 1997 | Schild mechanism set with `naf97mean.mod` and `nas97mean.mod` | Stateful current and reversal-term execution repaired. Velocity and detailed Vm/current/gate/state campaigns pass. |
| MRG / AxNode | `AXNODE.mod` plus the MRG cable assembly | The four AxNode current formulas (`I_na`, `I_nap`, `I_k`, `I_l`) and defaults match the MOD source. Fresh morphology, compartment, node-delay, passive, velocity, and Vm campaigns pass. Dense current traces are not claimed because the double-cable path intentionally does not support dense observable recording. |

The runtime defect found during this audit was model agnostic. Stateful models
passed their state to `prepare_membrane_step()` and `finalize_membrane_step()`
but not to the current and membrane conductance/reversal evaluations. This made
Tigerholm and both Schild models fail before a solve and would have used stale
reversal terms even after a narrow crash-only fix. The canonical single-cable
scan now forwards the same state tuple through all membrane evaluations.

The reactivated diameter campaign also exposed an unrelated model-agnostic
runtime defect. Batched static JAX arrays were cached by `id(...)` without
retaining or validating the source object. Recycled Python identities could
therefore select cable coefficients from another diameter, causing intermittent
NaNs or failed propagation. Source identity is now weakly validated on every
identity-based cache hit.

The detailed current audit found a Tigerholm-specific semantic mismatch that
propagation tests had hidden. NRV's `I_na` and `I_k` are NEURON ion totals:
they use concentration-dependent reversal potentials, split HCN equally
between Na and K, include the Na/K pump, and drive the `naoi`/`koi`
concentration mechanisms. AxonScope had instead recorded static channel
component summaries and omitted HCN/pump terms from the concentration budgets.
The canonical source model now exposes the same ion totals while retaining one
stateful solver path. At the intracellular reference point this reduced the
Tigerholm Vm RMSE from `0.3214` to `0.0723 mV`, Na-current RMSE from `0.0278`
to `0.0187 mA/cm2`, and K-current RMSE from `0.0666` to `0.0095 mA/cm2`.

Integer-lag diagnostics over +/-3 time steps select zero lag for the Rattay
and corrected Tigerholm Na/K traces. Their remaining peak-local current error
therefore comes from small gate-trajectory differences, not from a shifted
recording convention. Focused Rattay and Tigerholm current tolerances are now
model-scale checks rather than the previous broad multi-model placeholders.
The HH, Sundt, Schild 1994, and Schild 1997 current thresholds were likewise
replaced with model-scale limits after fresh integrated runs. Passive current
is checked against the configured NRV `pas` equation, while MRG/AxNode is
checked directly against all four `AXNODE.mod` current equations because dense
double-cable current recording is outside the supported recording contract.

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
749 passed, 1 skipped

tests/nrv/velocity_vs_diameter/test_velocity_systematic_vs_nrv.py
7 passed

tests/nrv/intracellular/test_intracellular_systematic_vs_nrv.py
7 passed

tests/nrv/extracellular/test_extracellular_systematic_vs_nrv.py
7 passed

tests/nrv
116 passed
```

The systematic velocity campaign now uses bilateral crossing-time regression
with a shared `2%` relative tolerance and a `0.001 m/s` numerical floor. The
previous `0.5 m/s` floor could accept a zero velocity for unmyelinated fibers
and is not retained.
