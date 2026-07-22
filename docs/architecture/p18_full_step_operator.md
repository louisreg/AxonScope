# P18 Full-Step Operator

This note formalizes the operator already implemented by the generated
membrane runtime and the single- and double-cable kernels. It is a statement
about the current execution structure, not a proposal for another public API.

## Per-axon structure

For one axon and one time step, let `p` be the predicted local membrane state,
`u` the cable voltage unknowns, and `q` the finalized local state. With rates
and conductance coefficients evaluated at the voltages prescribed by the
current time discretization, the staged residual is

```text
R_p = K p - r_p
R_u = C p + A u - r_u
R_q = D p + E u + q - r_q.
```

Its frozen-coefficient operator is block lower triangular:

```text
[ K  0  0 ] [p]   [r_p]
[ C  A  0 ] [u] = [r_u]
[ D  E  I ] [q]   [r_q].
```

`K` is a direct sum of compartment-local finite-state systems. The generated
runtime does not materialize their dense `[Nx, S, S]` representation: it
assembles the implicit rows directly from transition rates, eliminates one
state for conserved probability blocks, and uses a statically unrolled local
solve. Scalar HH-like gates and auxiliary state have the same locality even
when their update formula is closed-form rather than a finite-state solve.

`A` contains the only spatial coupling. For a single cable it is scalar
tridiagonal. For a double cable it is block tridiagonal with local `2 x 2`
intracellular/extracellular blocks and diagonal off-blocks. `C`, `D`, and `E`
are compartment-local: membrane state at one compartment does not directly
couple to another compartment.

Forward substitution in this system is exactly the retained execution order:

1. update each local membrane block matrix-free;
2. solve `A u = r_u - C p` with scalar or `2 x 2` block Thomas;
3. finalize each local membrane block.

The executable proof in `benchmark/analysis/full_step_operator.py` assembles
the reference matrix, compares this staged path with a dense solve, and uses
the production kinetic, scalar tridiagonal, and block tridiagonal solvers.

## Population axis

For a population, the full operator is

```text
L_population = diag(L_1, L_2, ..., L_Naxon).
```

There are no off-axon entries. `Naxon` is therefore a batch/direct-sum axis,
not an additional sparse coupling dimension. Axons may have different dynamic
coefficients while retaining this structure.

Materializing a sparse matrix of scale `Naxon * Nx * (S + cable_width)` would
store local state blocks, cable coefficients, and sparse indices that the
generated local update and Thomas kernels already consume in structured form.
It supplies no missing coupling information and adds allocation and memory
traffic. The benchmark reports a conservative CSR storage estimate alongside
the production-oriented core arrays; it is an accounting estimate, not a
runtime memory measurement.

## Nonlinear boundary

The membrane equations are generally nonlinear in voltage and state. There is
therefore no single constant global matrix for an entire simulation. The
matrix above is the frozen per-step operator, or equivalently the structural
Jacobian of the staged residual at known coefficients.

A different coupled solve becomes relevant only if temporal validation shows
that the present predictor/finalizer treatment is insufficient. In that case,
the model-agnostic extension is to generate complete local voltage-state
Jacobians and eliminate those local blocks with a per-compartment Schur
complement. It is not a reason to assemble one population-wide sparse matrix.
