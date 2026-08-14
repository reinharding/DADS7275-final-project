# Analysis Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate three divergent copies of the feature-engineering logic into one `src/features.py`, eliminating the target leakage in `src/predict_s10.py` by construction.

**Architecture:** Extract the notebook's feature functions (cells 24 and 26 — the correct, best-factored copy) into an importable module. `predict_s10.py` and `logistic_regression_eval.py` become thin consumers. The notebook keeps its readable inline definitions, guarded by a strict drift test that execs the notebook's own cells and asserts frame equality against the module across all 9 seasons.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, pytest.

## Global Constraints

- Source of truth for extraction: `mcsr_playoff_prediction.ipynb` cells 24 and 26. Not the scripts.
- `PLAYOFF_RESULTS` must never become a module-level mutable global in `features.py`. Playoff results are passed as a parameter.
- Rulings R2 and R3 are applied to `src/features.py` **and** notebook cell 24 in the same task, so the drift test stays a strict equality assertion with no column carve-outs.
- Never mix transcription and semantic change in one task. Task 2 is verbatim; semantic changes start at Task 5.
- Test player roster derives from `load_playoff_results()`, never from a hardcoded nickname list.
- Notebook edits use the NotebookEdit tool, not text substitution on the `.ipynb` JSON.
- Every regenerated PNG requires an explained delta before commit.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/features.py` | Loaders, pure `compute_*` functions, `build_player_features`. No charts, no models. |
| `tests/notebook_source.py` | Execs notebook cells into a namespace. Used by the drift test and the baseline generator. |
| `tests/conftest.py` | Module-scoped pytest fixtures: `df`, `results`, `players`, `lcq`, `deltas`. |
| `tests/test_features_baseline.py` | The verbatim gate — `features.py` vs the committed fixture CSV. |
| `tests/test_features.py` | Unit tests on the pure functions, including the leakage regression test. |
| `tests/test_notebook_drift.py` | Strict frame equality, notebook vs module, seasons 1–9. |
| `tests/fixtures/features_baseline.csv` | Golden output. Regenerated deliberately at Tasks 5 and 6; its diff is the record of each ruling. |
| `scripts/make_baseline.py` | Regenerates the fixture from notebook code. |

**Known necessary deviations from verbatim** (expected, not defects):

1. Notebook cell 2 sets `BASE_DIR = os.path.abspath('.')`. `features.py` uses `os.path.dirname(os.path.abspath(__file__))` instead.
2. `build_player_features_pred` is renamed to `build_player_features` in the module.

Nothing else may differ in Task 2.

---

### Task 1: Test scaffolding and baseline fixture

**Files:**
- Create: `tests/notebook_source.py`
- Create: `scripts/make_baseline.py`
- Create: `tests/fixtures/features_baseline.csv` (generated)
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `notebook_namespace(targets: list[str]) -> dict` — execs every notebook code cell that defines any name in `targets`, returns the namespace. Injects `json`, `os`, `pd`, `np`, `RAW`, `SEASONS`.

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
pytest>=8.0
```

- [ ] **Step 2: Write the notebook source loader**

Create `tests/notebook_source.py`:

```python
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
```

- [ ] **Step 3: Verify the loader finds every target**

Run:

```bash
python -c "import sys; sys.path.insert(0,'tests'); from notebook_source import notebook_namespace; ns=notebook_namespace(['load_all_matches','load_playoff_results','load_lcq_by_season','load_delta_by_season','compute_win_rate','compute_finish_stats','compute_recent_form','compute_current_elo','compute_elo_momentum','compute_pedigree','build_player_features_pred']); print('all targets resolved')"
```

Expected: `all targets resolved`. If it raises `RuntimeError`, a target name is misspelled or its cell also depends on an un-injected global — add the global to `ns` in `notebook_namespace`.

- [ ] **Step 4: Write the baseline generator**

Create `scripts/make_baseline.py`:

```python
"""Regenerate tests/fixtures/features_baseline.csv from the notebook's code.

Run this only when a ruling deliberately changes feature semantics. The diff on
the generated CSV is the visible record of what that ruling changed.
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

from notebook_source import notebook_namespace  # noqa: E402

TARGETS = [
    "load_all_matches", "load_playoff_results",
    "load_lcq_by_season", "load_delta_by_season",
    "compute_win_rate", "compute_finish_stats", "compute_recent_form",
    "compute_current_elo", "compute_elo_momentum", "compute_pedigree",
    "build_player_features_pred",
]

OUT = os.path.join(REPO_ROOT, "tests", "fixtures", "features_baseline.csv")


def roster(results: dict) -> list[str]:
    """Every player appearing in any bracket tier, across all seasons."""
    names = set()
    for season in results.values():
        for value in season.values():
            if isinstance(value, str):
                names.add(value)
            elif isinstance(value, list):
                names.update(value)
    return sorted(names)


def main():
    ns = notebook_namespace(TARGETS)
    df = ns["load_all_matches"]()
    results = ns["load_playoff_results"]()
    lcq = ns["load_lcq_by_season"]()
    deltas = ns["load_delta_by_season"]()
    players = roster(results)

    frames = []
    for season in range(1, 10):
        feat = ns["build_player_features_pred"](
            df, players, results, season,
            lcq_by_season=lcq, delta_by_season=deltas,
        )
        feat.insert(0, "season", season)
        frames.append(feat)

    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows x {len(out.columns)} cols -> {OUT}")
    print(f"players: {len(players)}  seasons: 1-9")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the fixture**

Run:

```bash
python scripts/make_baseline.py
```

Expected: `wrote 477 rows x 15 cols` (53 players × 9 seasons). If the row count differs, record the actual number — it is the baseline, not an error, as long as it equals `len(players) * 9`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/notebook_source.py scripts/make_baseline.py tests/fixtures/features_baseline.csv
git commit -m "test: pin notebook feature output as a golden fixture

Captures build_player_features_pred output for all 53 bracket players
across seasons 1-9, straight from the notebook's own cells. This is the
reference that the extracted src/features.py must reproduce exactly
before any semantic change is applied."
```

---

### Task 2: Extract features.py verbatim

**Files:**
- Create: `src/features.py`
- Create: `tests/conftest.py`
- Create: `tests/test_features_baseline.py`

**Interfaces:**
- Consumes: `tests/fixtures/features_baseline.csv` from Task 1.
- Produces:
  - `features.load_all_matches() -> pd.DataFrame`
  - `features.load_playoff_results() -> dict[int, dict]`
  - `features.load_lcq_by_season() -> dict[int, set[str]]`
  - `features.load_delta_by_season() -> dict[int, dict[str, int]]`
  - `features.load_h2h_csv() -> pd.DataFrame`
  - `features.build_h2h_lookup(df, season_filter) -> dict[tuple[str, str], float]`
  - `features.compute_win_rate(as_p1, as_p2) -> tuple[float, int, int]`
  - `features.compute_finish_stats(as_p1, as_p2) -> tuple[float, float, float]`
  - `features.compute_recent_form(as_p1, as_p2, fallback_wr) -> float`
  - `features.compute_current_elo(as_p1, as_p2, default=1500.0) -> float`
  - `features.compute_elo_momentum(as_p1, as_p2) -> float`
  - `features.compute_pedigree(nick, ped) -> tuple[int, int, int]`
  - `features.build_player_features(df, players, playoff_results, season_filter, pedigree_cutoff=None, lcq_by_season=None, delta_by_season=None, lcq_season=None) -> pd.DataFrame`

- [ ] **Step 1: Create src/features.py**

Copy the following notebook cells verbatim into `src/features.py`, in this order:

| Notebook cell | Functions |
|---|---|
| 4 | `load_all_matches`, `load_playoff_results`, `load_h2h_csv` |
| 25 | `load_lcq_by_season`, `load_delta_by_season`, `build_h2h_lookup` |
| 24 | the six `compute_*` functions |
| 26 | `build_player_features_pred` |

Do not copy the trailing module-level calls in cells 4 and 25 (`df_matches = load_all_matches()`, `lcq_by_season = load_lcq_by_season()`, the `print` statements). Functions only.

Prepend this header, which replaces the cwd-relative constants from cell 2:

```python
"""Feature engineering for MCSR playoff prediction.

Single source of truth, extracted from mcsr_playoff_prediction.ipynb cells 4,
24, 25 and 26. The notebook keeps its own inline copies for readability;
tests/test_notebook_drift.py asserts the two cannot diverge.

Playoff results are always passed as a parameter. They are deliberately not a
module global: pedigree features must only ever see the dict they were handed,
already filtered to pedigree_cutoff.
"""

import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO_ROOT, "data", "raw")
SEASONS = list(range(1, 10))
```

Apply exactly two changes to the copied code, both listed in File Structure above:

1. Rename `build_player_features_pred` to `build_player_features`.
2. Nothing else. Paths already resolve through the module-level `RAW`.

- [ ] **Step 2: Write the shared fixtures**

Create `tests/conftest.py`:

```python
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import features  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return features.load_all_matches()


@pytest.fixture(scope="session")
def results():
    return features.load_playoff_results()


@pytest.fixture(scope="session")
def lcq():
    return features.load_lcq_by_season()


@pytest.fixture(scope="session")
def deltas():
    return features.load_delta_by_season()


@pytest.fixture(scope="session")
def players(results):
    """Every player appearing in any bracket tier, across all seasons.

    Derived from bracket data rather than a hardcoded roster: the hardcoded
    lists in scraper.py hold nicknames as announced at the time, which the API
    no longer agrees with for at least Season 6.
    """
    names = set()
    for season in results.values():
        for value in season.values():
            if isinstance(value, str):
                names.add(value)
            elif isinstance(value, list):
                names.update(value)
    return sorted(names)
```

- [ ] **Step 3: Write the verbatim gate test**

Create `tests/test_features_baseline.py`:

```python
"""The verbatim gate.

Proves src/features.py reproduces the notebook's feature output exactly. Run
before any semantic ruling is applied, so that "did I transcribe faithfully?"
is answered independently of "did I change the math?".
"""

import os

import pandas as pd
import pytest

import features

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "features_baseline.csv")


@pytest.fixture(scope="module")
def baseline():
    return pd.read_csv(FIXTURE)


def test_matches_baseline(df, players, results, lcq, deltas, baseline):
    frames = []
    for season in range(1, 10):
        feat = features.build_player_features(
            df, players, results, season,
            lcq_by_season=lcq, delta_by_season=deltas,
        )
        feat.insert(0, "season", season)
        frames.append(feat)
    actual = pd.concat(frames, ignore_index=True)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        baseline.reset_index(drop=True),
        check_dtype=False,
    )
```

- [ ] **Step 4: Run the gate**

Run:

```bash
python -m pytest tests/test_features_baseline.py -v
```

Expected: PASS. A failure here means transcription drifted — diff the failing columns against the notebook cell and correct `features.py`. Do not regenerate the fixture to make this pass; the fixture is the reference.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/conftest.py tests/test_features_baseline.py
git commit -m "feat: extract feature engineering into src/features.py

Verbatim extraction from notebook cells 4, 24, 25 and 26 — the correct
and best-factored of the three copies in this repo. Verified to reproduce
the golden fixture exactly; no semantic change in this commit.

Playoff results are a parameter, not a module global. That is the
structural change that makes the predict_s10.py leakage unrepresentable:
pedigree can only read the dict it was handed."
```

---

### Task 3: Notebook drift test

**Files:**
- Create: `tests/test_notebook_drift.py`

**Interfaces:**
- Consumes: `notebook_namespace` (Task 1), `features.build_player_features` (Task 2), conftest fixtures.
- Produces: nothing consumed by later tasks. This test is a guard that must keep passing.

- [ ] **Step 1: Write the drift test**

Create `tests/test_notebook_drift.py`:

```python
"""Guarantee the notebook's inline copy and src/features.py cannot diverge.

Three copies of this logic existed in the repo and they disagreed; one of them
leaked future-season pedigree into training features. This test is what stops
that recurring. Strict equality, no column carve-outs — carve-outs are exactly
where the next drift would hide.
"""

import pandas as pd
import pytest

import features
from notebook_source import notebook_namespace

TARGETS = [
    "compute_win_rate", "compute_finish_stats", "compute_recent_form",
    "compute_current_elo", "compute_elo_momentum", "compute_pedigree",
    "build_player_features_pred",
]


@pytest.fixture(scope="module")
def nb():
    return notebook_namespace(TARGETS)


@pytest.mark.parametrize("season", range(1, 10))
def test_no_drift(nb, df, players, results, lcq, deltas, season):
    expected = nb["build_player_features_pred"](
        df, players, results, season,
        lcq_by_season=lcq, delta_by_season=deltas,
    )
    actual = features.build_player_features(
        df, players, results, season,
        lcq_by_season=lcq, delta_by_season=deltas,
    )
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
```

- [ ] **Step 2: Add tests dir to the import path**

Append to `tests/conftest.py` (after the existing `sys.path.insert` for `src`):

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 3: Run the drift test**

Run:

```bash
python -m pytest tests/test_notebook_drift.py -v
```

Expected: 9 PASSED. A failure now means the Task 2 extraction was not faithful — fix `features.py`, not the notebook.

- [ ] **Step 4: Commit**

```bash
git add tests/test_notebook_drift.py tests/conftest.py
git commit -m "test: assert notebook and features.py cannot diverge

Execs the notebook's own cells and compares frame-for-frame across all
nine seasons. This is the regression guard for the class of bug that
produced the leakage: three copies of one function, silently disagreeing."
```

---

### Task 4: Unit tests for the pure functions

**Files:**
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: the six `compute_*` functions from Task 2.
- Produces: nothing. Tests only.

These test current behaviour and must all pass before Tasks 5 and 6 change anything.

- [ ] **Step 1: Write the unit tests**

Create `tests/test_features.py`:

```python
"""Unit tests for the pure feature functions.

No mocking and no fixtures on disk — these functions take DataFrames of match
rows and return numbers.
"""

import numpy as np
import pandas as pd
import pytest

import features


def rows(**cols) -> pd.DataFrame:
    """Build a match-rows frame with the schema build_player_features expects."""
    n = len(next(iter(cols.values()))) if cols else 0
    base = {
        "date": list(range(n)),
        "p1_won": [True] * n,
        "forfeited": [False] * n,
        "win_time_ms": [500_000.0] * n,
        "p1_elo": [1500.0] * n,
        "p2_elo": [1500.0] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


EMPTY = rows()


class TestComputeWinRate:
    def test_no_matches_returns_nan(self):
        rate, wins, total = features.compute_win_rate(EMPTY, EMPTY)
        assert np.isnan(rate)
        assert wins == 0
        assert total == 0

    def test_all_wins_as_p1(self):
        rate, wins, total = features.compute_win_rate(
            rows(p1_won=[True, True, True]), EMPTY
        )
        assert rate == 1.0
        assert (wins, total) == (3, 3)

    def test_counts_wins_from_both_sides(self):
        # 1 win as p1, 1 win as p2 (p1_won False means this player won), 2 losses
        rate, wins, total = features.compute_win_rate(
            rows(p1_won=[True, False]), rows(p1_won=[False, True])
        )
        assert rate == 0.5
        assert (wins, total) == (2, 4)


class TestComputeFinishStats:
    def test_no_times_returns_nan(self):
        avg, best, consistency = features.compute_finish_stats(EMPTY, EMPTY)
        assert np.isnan(avg) and np.isnan(best) and np.isnan(consistency)

    def test_single_time_has_no_consistency(self):
        avg, best, consistency = features.compute_finish_stats(
            rows(p1_won=[True], win_time_ms=[400_000.0]), EMPTY
        )
        assert avg == 400_000.0
        assert best == 400_000.0
        assert np.isnan(consistency)

    def test_best_is_the_minimum(self):
        avg, best, _ = features.compute_finish_stats(
            rows(p1_won=[True, True], win_time_ms=[400_000.0, 600_000.0]), EMPTY
        )
        assert best == 400_000.0
        assert avg == 500_000.0

    def test_forfeited_wins_excluded(self):
        avg, best, _ = features.compute_finish_stats(
            rows(p1_won=[True, True], forfeited=[False, True],
                 win_time_ms=[400_000.0, 100_000.0]), EMPTY
        )
        assert best == 400_000.0


class TestComputePedigree:
    PED = {
        1: {"champion": "alice", "finalist": "bob",
            "top4": ["carol"], "qf_exit": ["dave"]},
        2: {"champion": "bob", "finalist": "alice",
            "top4": ["dave"], "qf_exit": ["carol"]},
    }

    def test_champion_and_finalist_counts(self):
        deep, champ, fin = features.compute_pedigree("alice", self.PED)
        assert (champ, fin) == (1, 1)

    def test_weighted_score(self):
        # alice: 1 champion (4) + 1 finalist (3) = 7
        deep, _, _ = features.compute_pedigree("alice", self.PED)
        assert deep == 7
        # carol: 1 top4 (2) + 1 qf (1) = 3
        deep, _, _ = features.compute_pedigree("carol", self.PED)
        assert deep == 3

    def test_unknown_player_scores_zero(self):
        assert features.compute_pedigree("nobody", self.PED) == (0, 0, 0)

    def test_empty_history_scores_zero(self):
        assert features.compute_pedigree("alice", {}) == (0, 0, 0)


class TestComputeEloMomentum:
    def test_fewer_than_two_points_is_zero(self):
        assert features.compute_elo_momentum(EMPTY, EMPTY) == 0.0
        assert features.compute_elo_momentum(rows(p1_elo=[1500.0]), EMPTY) == 0.0

    def test_rising_elo_is_positive(self):
        assert features.compute_elo_momentum(
            rows(p1_elo=[1400.0, 1500.0]), EMPTY
        ) > 0

    def test_falling_elo_is_negative(self):
        assert features.compute_elo_momentum(
            rows(p1_elo=[1500.0, 1400.0]), EMPTY
        ) < 0


class TestComputeRecentForm:
    def test_falls_back_when_no_matches(self):
        assert features.compute_recent_form(EMPTY, EMPTY, 0.42) == 0.42

    def test_uses_only_last_twenty(self):
        # 20 losses then 5 wins -> last 20 contains 15 losses and 5 wins
        played = rows(p1_won=[False] * 20 + [True] * 5)
        assert features.compute_recent_form(played, EMPTY, 0.0) == pytest.approx(0.25)
```

- [ ] **Step 2: Run them**

Run:

```bash
python -m pytest tests/test_features.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test: unit-test the extracted pure feature functions"
```

---

### Task 5: Ruling R3 — fix the zero-variance consistency inversion

**Files:**
- Modify: `src/features.py` (`compute_finish_stats`)
- Modify: `mcsr_playoff_prediction.ipynb` cell 24 (`compute_finish_stats`)
- Modify: `tests/test_features.py`
- Regenerate: `tests/fixtures/features_baseline.csv`

**Interfaces:**
- Consumes: `features.compute_finish_stats` from Task 2.
- Produces: unchanged signature. Behaviour changes only at `std_time == 0.0`.

- [ ] **Step 1: Write the failing test**

Add to `TestComputeFinishStats` in `tests/test_features.py`:

```python
    def test_zero_variance_gives_maximum_consistency(self):
        """Identical finish times are perfectly consistent, not unmeasurable.

        The original guard read `if std_time and not np.isnan(std_time)`, which
        is falsy at exactly std_time == 0.0 — so the best possible value fell
        through to NaN.
        """
        _, _, consistency = features.compute_finish_stats(
            rows(p1_won=[True, True], win_time_ms=[400_000.0, 400_000.0]), EMPTY
        )
        assert consistency == 1.0
```

- [ ] **Step 2: Run it to confirm it fails**

Run:

```bash
python -m pytest tests/test_features.py::TestComputeFinishStats::test_zero_variance_gives_maximum_consistency -v
```

Expected: FAIL — `assert nan == 1.0`.

- [ ] **Step 3: Fix features.py**

In `src/features.py`, in `compute_finish_stats`, replace:

```python
    consistency = 1.0 / (std_time / 1000 + 1) if std_time and not np.isnan(std_time) else np.nan
```

with:

```python
    consistency = (1.0 / (std_time / 1000 + 1)
                   if std_time is not None and not np.isnan(std_time) else np.nan)
```

- [ ] **Step 4: Run the test to confirm it passes**

Run:

```bash
python -m pytest tests/test_features.py -v
```

Expected: all PASS.

- [ ] **Step 5: Apply the identical fix to the notebook**

Using the NotebookEdit tool, edit cell 24 of `mcsr_playoff_prediction.ipynb`. Replace the same line in its `compute_finish_stats` with the same corrected expression, preserving the surrounding comments.

- [ ] **Step 6: Confirm the drift test still passes**

Run:

```bash
python -m pytest tests/test_notebook_drift.py -v
```

Expected: 9 PASSED. A failure means the notebook and module were not changed identically — reconcile before continuing.

- [ ] **Step 7: Regenerate the fixture**

Run:

```bash
python scripts/make_baseline.py && python -m pytest tests/test_features_baseline.py -v
```

Expected: the CSV changes, then PASS. Inspect `git diff --stat tests/fixtures/features_baseline.csv` — if zero rows changed, no player in the dataset had zero finish-time variance, which is a valid outcome. Record which it was in the commit message.

- [ ] **Step 8: Commit**

```bash
git add src/features.py mcsr_playoff_prediction.ipynb tests/test_features.py tests/fixtures/features_baseline.csv
git commit -m "fix: zero-variance finish times score maximum consistency

The guard read 'if std_time and not np.isnan(std_time)', which is falsy
at exactly std_time == 0.0 — so a player with perfectly consistent finish
times received NaN instead of the maximum score of 1.0. The condition
inverted at the best possible value.

Applied to src/features.py and notebook cell 24 together so the drift
test stays a strict equality assertion. Fixture regenerated; the CSV diff
records the effect."
```

---

### Task 6: Ruling R2 — zero-match players get NaN Elo

**Files:**
- Modify: `src/features.py` (`compute_current_elo`)
- Modify: `mcsr_playoff_prediction.ipynb` cell 24 (`compute_current_elo`)
- Modify: `tests/test_features.py`
- Regenerate: `tests/fixtures/features_baseline.csv`

**Interfaces:**
- Consumes: `features.compute_current_elo` from Task 2.
- Produces: `compute_current_elo(as_p1, as_p2, default=np.nan) -> float`. The `default` parameter is kept so callers can opt back in, but the default value changes from `1500.0` to `np.nan`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features.py`:

```python
class TestComputeCurrentElo:
    def test_uses_latest_appearance_as_p1(self):
        assert features.compute_current_elo(
            rows(p1_elo=[1400.0, 1600.0]), EMPTY
        ) == 1600.0

    def test_falls_back_to_p2_side(self):
        assert features.compute_current_elo(EMPTY, rows(p2_elo=[1700.0])) == 1700.0

    def test_no_matches_returns_nan(self):
        """A player with no matches is missing, not average.

        Returning 1500.0 bypassed the NaN policy every other feature follows:
        the row entered the model with a plausible Elo and NaN everywhere else,
        so the imputer never learned it was missing. Season 6's ogurikappa is
        the known instance.
        """
        assert np.isnan(features.compute_current_elo(EMPTY, EMPTY))

    def test_explicit_default_still_honoured(self):
        assert features.compute_current_elo(EMPTY, EMPTY, default=1500.0) == 1500.0
```

- [ ] **Step 2: Run it to confirm it fails**

Run:

```bash
python -m pytest tests/test_features.py::TestComputeCurrentElo -v
```

Expected: `test_no_matches_returns_nan` FAILS with `assert False` (1500.0 is not NaN). The other three PASS.

- [ ] **Step 3: Fix features.py**

In `src/features.py`, change the signature of `compute_current_elo`:

```python
def compute_current_elo(as_p1, as_p2, default=np.nan):
```

Leave the body unchanged — it already returns `default` when neither side has rows.

- [ ] **Step 4: Run the tests**

Run:

```bash
python -m pytest tests/test_features.py -v
```

Expected: all PASS.

- [ ] **Step 5: Apply the identical change to the notebook**

Using the NotebookEdit tool, change cell 24's `compute_current_elo` signature to `default=np.nan` and update its trailing comment from `# Default to 1500 if no data` to `# NaN if no data — let the imputer handle it`.

- [ ] **Step 6: Confirm the drift test still passes**

Run:

```bash
python -m pytest tests/test_notebook_drift.py -v
```

Expected: 9 PASSED.

- [ ] **Step 7: Regenerate the fixture and inspect the delta**

Run:

```bash
python scripts/make_baseline.py && git diff --stat tests/fixtures/features_baseline.csv
```

Then confirm the change is confined to zero-match players:

```bash
python -c "import pandas as pd; b=pd.read_csv('tests/fixtures/features_baseline.csv'); n=b[b['elo'].isna()]; print(n[['season','nickname']].to_string()); print(f'{len(n)} rows now NaN elo')"
```

Expected: only players with no matches up to that season appear. Any player with match history showing NaN Elo is a defect — stop and investigate.

- [ ] **Step 8: Run the full suite and commit**

```bash
python -m pytest tests/ -v
```

Expected: all PASS.

```bash
git add src/features.py mcsr_playoff_prediction.ipynb tests/test_features.py tests/fixtures/features_baseline.csv
git commit -m "fix: zero-match players get NaN Elo, not a hardcoded 1500

compute_current_elo defaulted to 1500.0, silently bypassing the NaN
policy every other feature follows. A player with no matches entered the
model with a plausible average Elo and NaN everywhere else, so the
SimpleImputer never learned the row was missing.

Season 6's ogurikappa is the known instance: absent from all 1351 S6
matches and from 15 of the 120 expected H2H pairs, yet carrying a
confident 1500 Elo into training.

Applied to features.py and notebook cell 24 together."
```

---

### Task 7: Leakage regression test

**Files:**
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `features.build_player_features` from Task 2, conftest fixtures.
- Produces: nothing. This is the test that would have caught the original defect.

- [ ] **Step 1: Write the test**

Add to `tests/test_features.py`:

```python
class TestPedigreeLeakage:
    """The defect this whole change exists to prevent.

    src/predict_s10.py filtered match data by season_filter but read pedigree
    from an unfiltered module global, so the feature row for season s included
    championships won in seasons s+1..9.
    """

    def test_pedigree_excludes_the_target_season_by_default(self, df, results):
        """A player's own win in the season being predicted is never a feature.

        Season 9's champion also won Season 8, so the correct count under the
        default cutoff is 1, not 0. What must hold is that the Season 9 title
        itself is absent — proven by comparing against an explicit cutoff that
        does include it.
        """
        champion = results[9]["champion"]
        through_8 = sum(1 for s, r in results.items()
                        if s <= 8 and r.get("champion") == champion)
        through_9 = sum(1 for s, r in results.items()
                        if s <= 9 and r.get("champion") == champion)
        assert through_9 == through_8 + 1, "precondition: S9 champion won S9"

        default = features.build_player_features(
            df, [champion], results, season_filter=9
        )
        assert default.iloc[0]["champion_count"] == through_8

        explicit = features.build_player_features(
            df, [champion], results, season_filter=9, pedigree_cutoff=9
        )
        assert explicit.iloc[0]["champion_count"] == through_9

    def test_pedigree_counts_strictly_earlier_seasons(self, df, results):
        """A season-3 champion is visible from season 4 onward, never before."""
        champion = results[3]["champion"]
        before = features.build_player_features(df, [champion], results, season_filter=3)
        after = features.build_player_features(df, [champion], results, season_filter=4)
        assert before.iloc[0]["champion_count"] == 0
        assert after.iloc[0]["champion_count"] >= 1

    def test_explicit_cutoff_overrides_the_default(self, df, results):
        """The hold-out path passes season_filter explicitly to include it."""
        champion = results[9]["champion"]
        feat = features.build_player_features(
            df, [champion], results, season_filter=9, pedigree_cutoff=9
        )
        assert feat.iloc[0]["champion_count"] >= 1

    def test_future_seasons_never_leak(self, df, results):
        """No season's feature row may reflect any later season's bracket."""
        champion = results[9]["champion"]
        for season in range(1, 9):
            feat = features.build_player_features(
                df, [champion], results, season_filter=season
            )
            counted = feat.iloc[0]["champion_count"]
            earlier = sum(1 for s, r in results.items()
                          if s <= season - 1 and r.get("champion") == champion)
            assert counted == earlier, f"season {season}: {counted} != {earlier}"
```

- [ ] **Step 2: Run it**

Run:

```bash
python -m pytest tests/test_features.py::TestPedigreeLeakage -v
```

Expected: all PASS — `features.py` inherited the notebook's correct cutoff logic.

- [ ] **Step 3: Verify the test actually detects the defect**

Temporarily break `src/features.py` by changing the pedigree filter to ignore the cutoff:

```python
    ped = playoff_results   # TEMPORARY - reproduces the predict_s10.py defect
```

Run:

```bash
python -m pytest tests/test_features.py::TestPedigreeLeakage -v
```

Expected: FAIL. This confirms the test has teeth. **Revert the temporary change** and re-run to confirm PASS before committing.

- [ ] **Step 4: Commit**

```bash
git add tests/test_features.py
git commit -m "test: regression test for future-season pedigree leakage

Verified to fail when the pedigree_cutoff filter is removed, which is
exactly the shape of the predict_s10.py defect."
```

---

### Task 8: Rewrite predict_s10.py as a thin driver

**Files:**
- Modify: `src/predict_s10.py` (833 lines → roughly 200)

**Interfaces:**
- Consumes: every function from `features.py` (Task 2).
- Produces: five PNGs in `data/processed/`. No importable API — this is a script.

- [ ] **Step 1: Delete the duplicated data layer**

Remove from `src/predict_s10.py`:

- `load_playoff_results` (line 53)
- `load_all_matches` (line 142)
- `build_player_features` (line 182) — the leaky one
- the module-level `PLAYOFF_RESULTS: dict = {}` (line 92) and the `global PLAYOFF_RESULTS` statement in `main` (line 515)

Keep: `build_matchup_features`, `build_training_data`, `build_player_outcome_data`, `train_lda`, `topk_accuracy`, `upset_detection_rate`, `predict_h2h`, `simulate_bracket`, `_make_charts`, `S10_POOL`, `SEASON_WEIGHTS`.

- [ ] **Step 2: Import from features**

Add below the existing imports:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (
    build_player_features,
    load_all_matches,
    load_playoff_results,
    load_lcq_by_season,
    load_delta_by_season,
)
```

- [ ] **Step 3: Thread playoff results as a parameter**

Every function that read the `PLAYOFF_RESULTS` global now takes it as an argument. Change these signatures and update their call sites in `main`:

```python
def build_training_data(df, feat_by_season, playoff_results):
def build_player_outcome_data(feat_by_season, playoff_results):
```

Inside each, replace `for season, results in PLAYOFF_RESULTS.items():` with `for season, results in playoff_results.items():`.

- [ ] **Step 4: Rebuild the feature construction in main**

Replace the `feat_by_season` construction (line 531) with:

```python
    playoff_results = load_playoff_results()
    lcq = load_lcq_by_season()
    deltas = load_delta_by_season()

    df = load_all_matches()

    players = sorted({
        name
        for season in playoff_results.values()
        for value in season.values()
        for name in ([value] if isinstance(value, str) else value)
    })

    feat_by_season = {
        s: build_player_features(df, players, playoff_results, season_filter=s,
                                 lcq_by_season=lcq, delta_by_season=deltas)
        for s in range(1, 10)
    }

    missing = {
        s: sorted(f[f["elo"].isna()]["nickname"])
        for s, f in feat_by_season.items()
    }
    total_missing = sum(len(v) for v in missing.values())
    if total_missing:
        print(f"\nWARNING: {total_missing} player-seasons have no match data:")
        for s, names in missing.items():
            if names:
                print(f"  S{s}: {', '.join(names)}")
```

- [ ] **Step 5: Verify it runs**

Run:

```bash
python src/predict_s10.py
```

Expected: completes without error, prints the missing-data warning, writes five PNGs. Do not commit the PNGs yet — Task 10 handles them under the verification gate.

- [ ] **Step 6: Confirm the test suite still passes**

Run:

```bash
python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit the code only**

```bash
git add src/predict_s10.py
git checkout -- data/processed/
git commit -m "refactor: predict_s10.py becomes a thin driver over features.py

Deletes its own loaders and its leaky build_player_features, importing
from src/features.py instead. Playoff results are threaded as a parameter
rather than read from a module global — the global was what let pedigree
escape the season filter.

Also reports player-seasons with no match data at the end of a run,
instead of letting them pass silently.

Charts intentionally not regenerated in this commit; that happens under
the verification gate."
```

---

### Task 9: Repoint logistic_regression_eval.py

**Files:**
- Modify: `src/logistic_regression_eval.py` (894 lines → roughly 450)

**Interfaces:**
- Consumes: `features.py` (Task 2).
- Produces: three PNGs. No importable API.

- [ ] **Step 1: Delete its duplicated data layer**

Remove from `src/logistic_regression_eval.py`:

- `load_lcq_by_season` (line 75)
- `load_delta_by_season` (line 101)
- `load_playoff_results` (line 135)
- `load_all_matches` (line 187)
- `build_player_features` (line 229)
- `build_h2h_lookup` (line 54)

Keep: `build_matchup_vector`, `_expand_tier`, `build_pairwise_data`, `compute_upset_detection`, `print_metrics`, `plot_confusion_matrix`, `plot_feature_importance`, `plot_season_coverage`, `main`, `PLAYER_FEAT_COLS`.

- [ ] **Step 2: Import from features**

Add below the existing imports:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (
    build_h2h_lookup,
    build_player_features,
    load_all_matches,
    load_delta_by_season,
    load_lcq_by_season,
    load_playoff_results,
)
```

- [ ] **Step 3: Verify it runs**

Run:

```bash
python src/logistic_regression_eval.py
```

Expected: completes without error and reports its S9 hold-out metrics. Note the reported accuracy — Task 10 compares against it.

- [ ] **Step 4: Confirm the suite passes**

Run:

```bash
python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 5: Commit the code only**

```bash
git add src/logistic_regression_eval.py
git checkout -- data/processed/
git commit -m "refactor: logistic_regression_eval.py imports from features.py

Its pedigree handling was already correct, so this removes duplication
rather than fixing a defect. All three copies of the feature logic now
resolve to one."
```

---

### Task 10: Regenerate charts under the verification gate

**Files:**
- Regenerate: 5 PNGs from `predict_s10.py`, 3 PNGs from `logistic_regression_eval.py`
- Modify: `README.md` (Project Structure section)

**Interfaces:**
- Consumes: Tasks 8 and 9.
- Produces: the final committed state.

- [ ] **Step 1: Capture the before state**

Run:

```bash
python -c "import hashlib,glob; [print(hashlib.md5(open(f,'rb').read()).hexdigest()[:12], f) for f in sorted(glob.glob('data/processed/*.png'))]"
```

Record the output.

- [ ] **Step 2: Regenerate everything**

Run:

```bash
python src/predict_s10.py && python src/logistic_regression_eval.py
```

- [ ] **Step 3: Identify what moved**

Run:

```bash
git status --short data/processed/
```

Expected to change: `s10_champion_probs.png`, `s10_feature_importance.png`, `s10_elo_vs_pedigree.png`, `s10_lda_outcomes.png`, `lda_pca_outcomes.png`, `lr_confusion_matrix.png`, `lr_feature_importance.png`, `lr_season_coverage.png`.

Expected **not** to change: the 9 PNGs and 5 CSVs written by `analysis.py`. If any of those changed, `analysis.py` was touched by mistake — revert it.

- [ ] **Step 4: The verification gate**

Compare the championship probabilities printed by `predict_s10.py` against the pre-change `data/processed/s10_champion_probs.png`.

The gate: **players whose standing was inflated by future-season pedigree must fall.** Concretely, a player who won a title in a season *later* than one they were being scored in previously carried that title as a feature; they should now rank lower.

Write the observed movement into the commit message — which players moved, in which direction, and why leakage explains it.

If a movement appears that leakage plus R2/R3 cannot explain, **stop**. Do not commit. Investigate the discrepancy first.

For `logistic_regression_eval.py`'s three charts, the only expected driver is R2/R3, since its pedigree was already correct. A large shift there is a defect.

- [ ] **Step 5: Update the README project structure**

In `README.md`, in the Project Structure block, add the two new entries:

```
├── src/
│   ├── features.py                 # Shared feature engineering (single source of truth)
│   ├── scraper.py                  # MCSR Ranked API scraper (S1-S9)
...
├── tests/                          # Unit tests + notebook drift test
├── scripts/
│   └── make_baseline.py            # Regenerates the feature fixture
```

- [ ] **Step 6: Run the full suite one final time**

Run:

```bash
python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add data/processed/ README.md
git commit -m "chore: regenerate charts from de-leaked feature pipeline

<Replace this line with the observed movement from Step 4: which players
moved, which direction, and why leakage explains it.>

The five predict_s10.py charts change for two reasons: pedigree no longer
leaks future seasons, and rulings R2/R3. The three logistic_regression_eval.py
charts change for R2/R3 only, since its pedigree handling was already correct."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `src/features.py` three-layer module | 2 |
| `PLAYOFF_RESULTS` demoted from global to parameter | 2, 8 |
| R1 — NaN missing-data policy | 2 (inherited from the notebook), 8 (0.5 retired with the file) |
| R2 — `compute_current_elo` returns NaN | 6 |
| R3 — zero-variance consistency | 5 |
| Unit tests | 4, 7 |
| Notebook drift test | 3 |
| `pytest` in requirements | 1 |
| Error handling: NaN rows + driver warning count | 6 (test), 8 (warning) |
| Sequencing steps 1–5 | 1, 2, 5–6, 8–9, 10 |
| Step 5 verification gate | 10 |
| Definition of done | 10 |

**Additions discovered during planning, not in the spec:**

- `load_h2h_csv` and `build_h2h_lookup` are included in `features.py`. The spec listed four loaders; `logistic_regression_eval.py` needs these two as well, and leaving them behind would preserve exactly the duplication this work removes.
- The path-resolution deviation (`os.path.abspath('.')` → `__file__`) is documented in File Structure so the Task 2 verbatim gate does not fail for the wrong reason.
- The test roster derives from `load_playoff_results()` rather than any hardcoded list, because `scraper.py`'s `PLAYOFF_PLAYERS[6]` disagrees with the API's own Season 6 bracket.

**Type consistency:** `build_player_features` is the module name throughout Tasks 2–10; `build_player_features_pred` appears only where the notebook's own symbol is referenced (Tasks 1 and 3). `compute_current_elo`'s `default` parameter is `1500.0` through Task 5 and `np.nan` from Task 6 onward, which Task 6 states explicitly.
