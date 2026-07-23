# Artifact Cache

AxonFleet owns one persistent artifact tree rooted at `.axonfleet_cache/` by
default. Set `AXONFLEET_CACHE=/absolute/path` to relocate the complete tree or
`AXONFLEET_CACHE=off` to disable persistence. No older per-backend cache-path
variables are supported.

The layout is deterministic:

```text
.axonfleet_cache/
  model_codegen/       generated membrane runtime modules and manifests
  runtime/jax/xla/     JAX persistent compilation entries
  runtime/jax/triton/  serialized AxonFleet Triton custom calls
```

`axs.cache.inspect()` reports paths, file counts, and byte sizes without
creating the cache. `axs.cache.clean()` removes these three known sections but
preserves unknown user files under a custom root. Generated membrane modules
needed while persistence is disabled are written to a process-temporary
directory and removed at process exit.

Generated membrane identities include source and compiler contract hashes,
selected source class, static metadata, and requested runtime targets. Missing
files, corrupt manifests, contract changes, and source changes are cache
misses. Dynamic membrane parameter values do not invalidate structurally
equivalent generated code.

Only requested targets are generated. Descriptive membrane lowering requests
no runtime module; JAX execution requests its JAX and host-NumPy support, plus
Triton only when that optional runtime is available. Explicit generated-code
inspection requests only the named files.

Installation never prebuilds membrane artifacts. A built-in model is generated
on its first use by the selected runtime, then reused while its content and
compiler-contract hashes remain valid. This keeps CPU installations free of
unused GPU artifacts and avoids regenerating an already translated model.

JAX enforces a default 2 GiB persistent compilation limit. Override it with
`AXONFLEET_JAX_CACHE_MAX_SIZE_BYTES`. Its minimum compile time, minimum entry
size, and optional XLA cache families remain configurable through the
`AXONFLEET_JAX_CACHE_*` variables. Generated membrane and Triton artifacts are
content-addressed and retained until `axs.cache.clean()` or manual removal.
