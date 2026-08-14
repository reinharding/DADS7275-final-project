"""Load function definitions out of the analysis notebook.

The notebook is the source of truth for feature engineering. These helpers let
tests execute its cells directly, so the notebook's inline definitions and
src/features.py can be compared on real data.
"""

import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(REPO_ROOT, "mcsr_playoff_prediction.ipynb")
RAW = os.path.join(REPO_ROOT, "data", "raw")
SEASONS = list(range(1, 10))


def notebook_namespace(targets: list[str]) -> dict:
    """Exec every notebook code cell defining any name in `targets`.

    Cell 2's path constants are injected rather than executed: the notebook
    resolves paths from the current working directory, which is not stable
    under pytest.
    """
    with open(NB_PATH, encoding="utf-8") as f:
        nb = json.load(f)

    ns = {"json": json, "os": os, "pd": pd, "np": np,
          "RAW": RAW, "SEASONS": SEASONS}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(f"def {t}(" in src for t in targets):
            exec(src, ns)

    missing = [t for t in targets if t not in ns]
    if missing:
        raise RuntimeError(f"Notebook did not define: {missing}")
    return ns
