from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent
CURRENT = sys.executable

SCRIPTS = [
    "001_baseline.py",
    "002_standalone_CN.py",
    "004_scipy_banded.py",
    "005_scipy_sparse.py",
    "006_scipy_factorized.py",
    "007_thomas_numpy.py",
    "008_torch.py",
    "008b_torch_compile.py",
    "009_torch_LU_fac.py",
    "010_thomas_torch.py",
    "011_jax.py",
    "012_jax_jit.py",
    "013_jax_LU_fac.py",
    "014_jax_LU_jit.py",
    "015_jax_thomas.py",
    "016_jax_thomas_jit.py",
    "017_jax_thomas_jit_optim.py",
    "018_jax_thomas_jit_optim_2.py",
    "019_jax_tridiagonal.py",
    "020_jax_tridiagonal_jitted.py",
    "021_jax_tridiagonal_jitted_optim.py",
    "021b_jax_tridiagonal_jitted_float32.py",
    "021c_jax_tridiagonal_jitted_float32_optim.py",
    "022_jax_tridiagonal_jitted_float32_gate_interp.py",
    "022b_jax_tridiagonal_jitted_float32_gate_interp2.py",
]


def main() -> None:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        subprocess.run([CURRENT, str(ROOT / script)], check=True, env=env)


if __name__ == "__main__":
    main()
