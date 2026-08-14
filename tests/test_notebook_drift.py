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
