# Benchmark Baselines

Baselines are independent scientific reference entry points. They are not
AxonFleet runtime paths or public APIs, and code under `src/axonfleet` must not
import them.

## ModelDB 230137

`modeldb_230137_voltage_clamp.py` runs the externally downloaded ModelDB
230137 Nav1.1-Nav1.9 mechanisms through NEURON and writes reference voltage
clamp curves. The MOD files are deliberately not vendored.

Compile the ModelDB mechanisms first, then run the script in that checkout:

```bash
nrnivmodl
python /path/to/AxonFleet/benchmark/baselines/modeldb_230137_voltage_clamp.py \
  --output modeldb_230137_voltage_clamp.json
```

Pass the resulting JSON to
`benchmark/curves/nav_isoform_voltage_clamp.py --modeldb-reference` for the
AxonFleet comparison. The baseline records its source paths and mechanism
inputs so results can be traced to the external checkout.
