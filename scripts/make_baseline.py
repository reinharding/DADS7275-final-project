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
