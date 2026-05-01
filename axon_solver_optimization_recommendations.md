# Optimization Recommendations for the Nerve Fiber Simulation Solver

## Context

The current solver is already well structured for JAX: it uses `jax.lax.scan` for the time loop, tridiagonal solvers for the single-cable case, pre-extracted runtime objects, and some precomputation for extracellular stimulation. The main opportunities for further speedup are therefore not small Python-level tweaks, but changes that reduce the amount of work done per time step, reduce memory traffic, and make the JAX/XLA compilation boundary cleaner.

The recommendations below are ordered by expected impact.

---

## 1. Make sure the actual solver kernel is explicitly JIT-compiled

The solver is written in a JAX-friendly way, but the production path should separate Python-side setup from the pure JAX kernel.

A good target structure would be:

```python
@jax.jit
def solve_kernel(Vm0, gates0, state0, lower, diag, upper, stimuli, params):
    def step(carry, n):
        ...
        return carry_out, recorded_output

    _, out = jax.lax.scan(step, init_carry, jnp.arange(Nt))
    return out
```

Then `solve()` should mainly prepare arrays and call the compiled kernel.

Important static arguments should include things like:

- `Nt`
- `use_extracellular`
- `record_observables`
- `record_diagnostics`
- recording mode
- solver variant

This avoids accidental recompilation and allows XLA to eliminate unused branches.

---

## 2. Avoid the generic extracellular solver with fixed inner iterations when possible

The extracellular generic solver appears to perform a fixed nonlinear-like loop per time step, with three iterations involving:

- gate update
- ionic current evaluation
- membrane step preparation
- block-tridiagonal solve

That means each time step can pay for roughly three linear solves and three membrane evaluations.

If numerical accuracy remains acceptable, prefer the inline extracellular path in `CrankNicholson`, for example by using:

```python
axon.prefer_inline_extracellular_solver = True
```

Then benchmark against the generic extracellular path.

This is likely one of the largest speedups available, especially for long simulations.

---

## 3. Do not store the full voltage matrix by default

The current default returns `V_all` with shape:

```text
Nt × Nx
```

For long simulations, this can become memory-bandwidth bound. In many production runs, storing every compartment at every time step is unnecessary.

A better API would be:

```python
solve(..., record="none")
solve(..., record="final")
solve(..., record="stride", stride=10)
solve(..., record="indices", indices=recorded_compartments)
solve(..., record="all")
```

Recommended behavior:

- `record="none"`: return only final state or summary metrics.
- `record="final"`: return only the final voltage and state.
- `record="stride"`: store one time point every `stride` steps.
- `record="indices"`: store only selected spatial locations.
- `record="all"`: keep the current full-output behavior for debugging and plotting.

This can reduce memory usage and transfer cost by orders of magnitude.

---

## 4. Precompute intracellular stimulation, not only extracellular stimulation

The code already supports precomputing extracellular potentials over the time grid. A similar approach should be used for intracellular current injection.

Instead of calling:

```python
Iinj = inj_fun(t_mid)
```

inside every scan step, precompute:

```python
t_mid = (jnp.arange(Nt, dtype=dtype) + 0.5) * dt
Iinj_mid_all = sample_intracellular_current_density(inj_fun, t_mid)
```

Then inside the step:

```python
Iinj = Iinj_mid_all[n]
```

For localized clamps, avoid building a full one-hot basis with `jnp.eye(Nx)` if possible. Store clamp indices and amplitudes, then scatter into a zero vector or precompute the full stimulation tensor once.

This is especially useful when the stimulus uses `searchsorted`, interpolation, or multiple clamp objects.

---

## 5. Keep the dense Crank-Nicolson solver only for validation

The dense solver assembles the full matrix and calls:

```python
jnp.linalg.solve(A, rhs)
```

This is useful for clarity and testing, but it should not be used in production for large `Nx`.

For production, use the tridiagonal solver path:

```python
jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]
```

The tridiagonal version has linear memory and compute complexity with respect to the number of compartments.

---

## 6. Split solver variants into specialized kernels

The current solver handles many modes in one implementation:

- single-cable vs extracellular
- observables vs no observables
- diagnostics vs no diagnostics
- driven extracellular stimulation vs no driven extracellular stimulation
- generic vs inline extracellular behavior

If these booleans are static, JAX can optimize away unused branches. However, for maximum speed and simpler compilation, it is better to expose specialized kernels:

```text
solve_single_cable_no_record
solve_single_cable_record_stride
solve_extracellular_inline_no_record
solve_extracellular_inline_record_stride
solve_extracellular_diagnostics
```

Benefits:

- smaller XLA graphs
- fewer dead branches
- simpler carry/output tuples in `lax.scan`
- lower compile time
- less risk of accidental recompilation

---

## 7. Optimize the extracellular block-tridiagonal solve

The scalar 2×2 block-tridiagonal solver is already a good design because it avoids materializing full `(Nx, 2, 2)` block arrays inside the time loop.

The remaining cost comes from allocating and filling several length-`Nx` arrays during each solve, such as forward coefficients and intermediate right-hand sides.

Possible improvements:

1. **Pre-factorization for passive or weakly varying systems**  
   If the membrane conductance is constant or changes slowly, reuse more of the factorization.

2. **Batching across fibers or stimuli**  
   A single Thomas solve is sequential along space and does not fully utilize a GPU. If many axons, diameters, parameter sets, or stimuli are simulated, use `vmap`:

   ```python
   batched_solve = jax.jit(jax.vmap(single_solve_kernel, in_axes=(0, 0, ...)))
   ```

3. **Specialized passive extracellular solver**  
   For passive or linearized cases, create a faster path that avoids repeated nonlinear membrane preparation.

---

## 8. Reduce duplicated membrane evaluations

In the inline path, the solver does a predictor phase and then a final phase. This can involve repeated calls to:

- gate update
- current evaluation
- membrane step preparation
- final membrane state update

This is numerically reasonable, but expensive. It is worth checking whether `prepare_membrane_step` can be refactored to reuse intermediate values between predictor and corrector stages.

Possible targets:

- Avoid recomputing ionic currents when they are only needed for diagnostics.
- Return reusable conductance/current terms from the first membrane preparation.
- Skip diagnostic-related current calculations unless diagnostics are explicitly enabled.

---

## 9. Use `float32` by default for production runs

The solver should default to `float32` unless there is a strong reason to use `float64`.

Recommended approach:

- Use `float32` for production simulations.
- Use `float64` for convergence tests and validation.
- Compare biologically relevant outputs, such as spike timing and conduction velocity, between `float32` and `float64`.

On many GPUs, `float64` can be much slower than `float32`.

---

## 10. Benchmark correctly with JAX synchronization

JAX execution is asynchronous, so timings must use `.block_until_ready()`.

Use a structure like:

```python
# Compilation run
out = solve_jit(...)
out.block_until_ready()

# Timed run
t0 = time.perf_counter()
out = solve_jit(...)
out.block_until_ready()
t1 = time.perf_counter()

print(f"Runtime: {t1 - t0:.6f} s")
```

For outputs that are PyTrees, apply:

```python
jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
```

or block on a representative array.

Always report separately:

- compilation time
- execution time
- memory usage
- `Nx`
- `Nt`
- dtype
- backend: CPU, GPU, or TPU
- recording mode

---

## Highest-impact action plan

If I had to prioritize only five changes, I would do them in this order:

1. **Use the inline extracellular solver path** instead of the generic three-iteration extracellular solver when accuracy allows.
2. **Create explicit `@jax.jit` kernels** separated from Python-side runtime preparation.
3. **Stop storing `V_all` by default**; add `none`, `final`, `stride`, and `indices` recording modes.
4. **Precompute intracellular stimulation** on the solver time grid, just like extracellular stimulation.
5. **Batch simulations with `vmap`** when running multiple fibers, stimuli, diameters, or parameter sets.

Expected biggest wins:

- Fewer linear solves per time step in extracellular mode.
- Much lower memory traffic from reduced recording.
- Less per-step stimulus overhead.
- Better accelerator utilization through batching.
- Cleaner XLA graphs through specialized JIT kernels.
