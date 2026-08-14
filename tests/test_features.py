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
