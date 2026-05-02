from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from benchmark.nrv_performance.nrv_axonscope_grid import main


if __name__ == "__main__":
    main()
