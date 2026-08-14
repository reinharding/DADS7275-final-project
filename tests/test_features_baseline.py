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
