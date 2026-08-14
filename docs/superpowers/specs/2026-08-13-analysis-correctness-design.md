# Analysis Correctness: Consolidate Feature Engineering into `src/features.py`

**Date:** 2026-08-13
**Scope:** Subsystem A of three (A: analysis correctness, B: collection hardening, C: repo infrastructure)
**Status:** Approved, ready for planning

## Problem

Three implementations of the same feature-engineering logic exist in this repo, and
they disagree.

| Location | Pedigree season-filtered? |
|---|---|
| `mcsr_playoff_prediction.ipynb` cell 26, `build_player_features_pred` | Yes |
| `src/logistic_regression_eval.py:229`, `build_player_features` | Yes |
| `src/predict_s10.py:182`, `build_player_features` | **No** |

Measured line-level similarity between the two script copies:

```
load_all_matches         lr= 42  s10= 40  similarity= 88%
load_playoff_results     lr= 52  s10= 89  similarity= 37%
build_player_features    lr=155  s10= 93  similarity= 26%
```

These are not copy-paste duplicates. They have diverged into different functions
sharing a name.

### The defect

`src/predict_s10.py:182` filters *match* data by `season_filter` but computes
tournament pedigree from the module-level `PLAYOFF_RESULTS` global with no season
filter (`predict_s10.py:246-250`). That global is populated by
`main()` via a `global` statement (`predict_s10.py:515`).

Consequence: the feature row for season *s* includes championships won in seasons
*s+1 … 9*. This is target leakage.

It reaches a reported metric. `predict_s10.py:682` evaluates Season 9 top-3
accuracy against `actual_champion="hackingnoises"` using `feat_by_season[9]`, whose
`champion_count` for that player already includes the Season 9 championship being
predicted.

### Blast radius

`feat_s10` (`predict_s10.py:604`) passes no `season_filter`, so counting all nine
completed seasons is legitimate there. **The Season 10 predictions are sound; the
evidence offered for them is not.**

The notebook is unaffected — it carries its own correct implementation and imports
nothing from `src/`. But the notebook saves no PNGs (it renders inline), so all five
of these committed images were produced by the leaky script, including the README
hero image:

- `s10_champion_probs.png` ← README hero image
- `s10_feature_importance.png`
- `s10_elo_vs_pedigree.png`
- `s10_lda_outcomes.png`
- `lda_pca_outcomes.png`

Unaffected: all 9 PNGs and 5 CSVs from `analysis.py`, all 3 PNGs from
`logistic_regression_eval.py`, everything in `data/raw/`.

## Goals

1. One implementation of feature engineering, in `src/features.py`.
2. Leakage eliminated by construction, and a regression test that proves it.
3. Notebook keeps its readable inline definitions, protected by a drift test.
4. The five PNGs regenerated from validated code.

### Non-goals

Deferred to subsystem B: UUID keying, scraper failure counting, retry
consolidation, incremental fetching, and the `_parse_inner` / `get()` /
`flatten_versus` unit tests. Deferred to subsystem C: Dockerfile, GitHub Actions,
README tradeoffs section.

## Design

### Module boundary

`src/features.py`, three layers, dependencies pointing one direction:

```
loaders      load_all_matches, load_playoff_results,
             load_lcq_by_season, load_delta_by_season
               read data/raw/*.json; depend on nothing in this repo

compute_*    compute_win_rate, compute_finish_stats, compute_recent_form,
             compute_current_elo, compute_elo_momentum, compute_pedigree
               pure functions over one player's match rows
               pandas/numpy only; no file I/O, no globals

build_player_features(df, players, playoff_results, season_filter,
                      pedigree_cutoff=None, lcq_by_season=None,
                      delta_by_season=None, lcq_season=None)
               composes compute_*; depends on compute_* only
```

Source of truth for the extraction is the notebook (cells 24 and 26), which is both
correct and the best-factored of the three copies.

`PLAYOFF_RESULTS` ceases to exist as a mutable module global. Playoff results are
passed as a parameter. This is the structural change that makes the leakage
unrepresentable: pedigree can only read the dict it was handed, already filtered to
`pedigree_cutoff`.

### Consumers

| File | Before | After |
|---|---|---|
| `src/predict_s10.py` | 833 lines; own loaders, own leaky features | ~200 lines: import, LDA, Monte Carlo, 5 charts |
| `src/logistic_regression_eval.py` | 894 lines; own loaders, own features | ~450 lines: import, pairwise training, S9 hold-out, 3 charts |
| notebook | inline definitions | unchanged except rulings 2b/2c below |

`src/analysis.py` is out of scope. Its `build_player_features` computes a different
thing (per-season EDA aggregates, not leakage-sensitive model inputs) and none of
its outputs are affected.

### Semantic rulings

Three decisions that cannot be settled by transcription.

**R1 — Missing-data policy: NaN.** A player with no matches yields `np.nan`, imputed
downstream by `SimpleImputer`. The notebook and `logistic_regression_eval.py`
already agree on this. `predict_s10.py`'s hardcoded `0.5` is the outlier and is
retired with the file.

**R2 — `compute_current_elo` returns `np.nan`, not `1500.0`.** The notebook's
`default=1500.0` silently bypasses R1: a zero-match player enters the model with a
hardcoded average Elo and `NaN` for every other feature, so the imputer never learns
the row was missing. Returning `np.nan` makes the treatment uniform.

This changes model inputs for zero-match players. Season 6's `ogurikappa` is the
known instance.

**R3 — Fix the zero-variance inversion in `compute_finish_stats`.**

```python
consistency = 1.0 / (std_time / 1000 + 1) if std_time and not np.isnan(std_time) else np.nan
```

`if std_time` is falsy at `std_time == 0.0`, so a player with perfectly consistent
finish times receives `NaN` instead of the maximum consistency score of `1.0`. The
condition inverts at exactly the best possible value. Corrected form:

```python
consistency = 1.0 / (std_time / 1000 + 1) if std_time is not None and not np.isnan(std_time) else np.nan
```

R2 and R3 are applied to `src/features.py` **and** to notebook cell 24 together, so
one semantic exists everywhere and the drift test can remain a strict equality
assertion.

### Testing

`tests/test_features.py` — unit tests on the pure functions. No mocking, no fixtures.

| Function | Cases |
|---|---|
| `compute_finish_stats` | `std_time == 0` → `1.0` (R3); no times → `NaN`; single time → `NaN` consistency |
| `compute_pedigree` | champion / finalist / top4 / qf counts; weighted score `4c+3f+2t+q`; unknown nickname → zeros |
| `compute_win_rate` | zero matches → `NaN`; all wins → `1.0` |
| `compute_elo_momentum` | fewer than 2 points → `0.0` |
| `build_player_features` | `pedigree_cutoff` excludes the target season — the leakage regression test |

`tests/test_notebook_drift.py` — guarantees the notebook's inline copy and
`features.py` cannot diverge again:

```python
TARGETS = ["compute_win_rate", "compute_finish_stats", "compute_recent_form",
           "compute_current_elo", "compute_elo_momentum", "compute_pedigree",
           "build_player_features_pred"]

def _notebook_namespace():
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    ns = {"pd": pd, "np": np}
    for cell in nb["cells"]:                       # cells 24 and 26 both match
        src = "".join(cell["source"])
        if any(f"def {t}(" in src for t in TARGETS):
            exec(src, ns)
    return ns

@pytest.mark.parametrize("season", range(1, 10))
def test_no_drift(season, df, players, results):   # fixtures load once, module-scoped
    pd.testing.assert_frame_equal(
        _notebook_namespace()["build_player_features_pred"](df, players, results, season),
        features.build_player_features(df, players, results, season),
    )
```

`df`, `players`, and `results` come from module-scoped pytest fixtures backed by
`features.py`'s own loaders reading `data/raw/`, so the comparison runs on the real
dataset rather than synthetic input.

Strict equality, no column carve-outs. Carve-outs are where the next drift would
hide.

`pytest` is added to `requirements.txt`.

### Error handling

`build_player_features` emits a full row of `NaN`s for a player with no matching
rows rather than skipping them, so the player count stays stable and the absence is
visible downstream. The driver script logs a count of such players at the end of a
run. This surfaces the `ogurikappa` case without pulling subsystem B's full roster
contract into this spec.

## Sequencing

The order keeps "did I transcribe faithfully?" separate from "did I change the
math?" — the two must never mix in one step.

1. **Snapshot the baseline.** Execute the notebook's cell 24/26 code as-is over
   seasons 1–9. Dump to `tests/fixtures/features_baseline.csv`. Commit.
2. **Extract verbatim.** Write `src/features.py` with no rulings applied. Assert it
   reproduces the fixture exactly. Proves faithful transcription and nothing else.
3. **Apply R2 and R3** to `src/features.py` and notebook cell 24 together.
   Regenerate the fixture and commit the diff — the fixture diff is the visible
   record of what the rulings changed.
4. **Rewrite consumers.** `predict_s10.py` becomes a thin driver; point
   `logistic_regression_eval.py` at `features.py`.
5. **Regenerate the five PNGs** and inspect the deltas.

### Verification gate at step 5

The championship odds will move, because pedigree stops leaking. Before committing
new images, confirm the shift is explicable: players whose standing was inflated by
future-season pedigree should fall. If a movement appears that leakage does not
explain, stop and investigate rather than commit.

## Risks

| Risk | Mitigation |
|---|---|
| Transcription error during extraction | Step 2's verbatim gate against the committed fixture |
| Notebook cells depend on globals defined in earlier cells | Drift test execs only the function-defining cells and injects `pd`/`np`; if other globals surface, they become explicit parameters |
| Regenerated PNGs differ for unexplained reasons | Step 5 verification gate; do not commit until explained |
| `logistic_regression_eval.py` rewrite changes its 3 PNGs | Its pedigree handling is already correct, so leakage is not a factor — but R2 and R3 do change its inputs. Expect movement attributable to those two rulings only; anything larger is a defect. Regenerate and inspect alongside step 5. |

## Definition of done

- `src/features.py` exists; `predict_s10.py` and `logistic_regression_eval.py`
  import from it and define no feature logic of their own.
- `pytest` passes: unit tests plus the 9-season drift test.
- The leakage regression test fails if `pedigree_cutoff` is removed.
- Five PNGs from `predict_s10.py` regenerated, deltas explained by leakage removal
  plus R2/R3.
- Three PNGs from `logistic_regression_eval.py` regenerated, deltas explained by
  R2/R3 alone.
- README project-structure section lists `src/features.py` and `tests/`.
