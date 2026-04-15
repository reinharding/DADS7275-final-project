"""
Logistic Regression – Seasonal Train / Test Evaluation
=======================================================
Train  : Seasons 1–8 playoff pairwise outcome comparisons
Test   : Season 9 playoff pairwise outcome comparisons (strict hold-out)
Features computed from match history *up to but not including Season 9*
to avoid data leakage.

Metrics reported
----------------
- Overall accuracy
- Per-class precision, recall, F1 (sklearn classification_report)
- Upset Detection Rate  (lower-Elo player advances further in the bracket)
- Confusion matrix plot
- Feature importance (LR coefficients) plot
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# NEW FEATURE HELPERS
# ---------------------------------------------------------------------------

def build_h2h_lookup(df: pd.DataFrame, season_filter: int) -> dict:
    """
    Build a head-to-head win-rate lookup from ranked match history.

    Returns {(nick_a, nick_b): nick_a_win_rate} for every ordered pair that
    appears in matches from seasons 1..season_filter.  Both directions are
    stored so callers can do a simple lookup without knowing which ordering
    was used in the raw data.
    """
    sub = df[df["season"] <= season_filter][["p1_nick", "p2_nick", "p1_won"]].dropna()
    wins  = {}
    total = {}
    for row in sub.itertuples(index=False):
        p1, p2, p1_won = row.p1_nick, row.p2_nick, row.p1_won
        total[(p1, p2)] = total.get((p1, p2), 0) + 1
        wins[ (p1, p2)] = wins.get( (p1, p2), 0) + int(p1_won)
        total[(p2, p1)] = total.get((p2, p1), 0) + 1
        wins[ (p2, p1)] = wins.get( (p2, p1), 0) + int(not p1_won)
    return {k: wins[k] / total[k] for k in total}


def load_lcq_by_season() -> dict:
    """
    Parse each season's playoffs JSON and return {season: set_of_lcq_nicks}.

    Players with seedNumber >= 12 occupy the four LCQ spots.  Early seasons
    that lack a seedNumber field default to an empty set.
    """
    lcq = {}
    for s in range(1, 10):
        path = os.path.join(RAW, f"season_{s}_playoffs.json")
        if not os.path.exists(path):
            lcq[s] = set()
            continue
        try:
            with open(path) as f:
                data = json.load(f)["data"]["data"]
            lcq[s] = {
                p["nickname"]
                for p in data.get("players", [])
                if p.get("seedNumber", 0) >= 12
            }
        except (KeyError, TypeError, json.JSONDecodeError):
            lcq[s] = set()
    return lcq


def load_delta_by_season() -> dict:
    """
    Parse each season's playoffs JSON and return {season: {nick: delta}}.

    delta = seed_rank - actual_place  (positive = player outperformed seeding).
    seed_rank is seedNumber + 1 (0-indexed → 1-indexed).
    """
    deltas = {}
    for s in range(1, 10):
        path = os.path.join(RAW, f"season_{s}_playoffs.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)["data"]["data"]
            idx_to_nick = {p["seedNumber"]: p["nickname"]
                           for p in data.get("players", [])}
            season_deltas = {}
            for r in data.get("results", []):
                idx          = r.get("player")
                actual_place = r.get("place")
                nick         = idx_to_nick.get(idx)
                if nick is None or actual_place is None:
                    continue
                seed_rank            = idx + 1          # 0-indexed → 1-indexed
                season_deltas[nick]  = seed_rank - actual_place
            deltas[s] = season_deltas
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return deltas

# ---------------------------------------------------------------------------
# Ground-truth playoff results (from scraped JSON + confirmed overrides)
# ---------------------------------------------------------------------------
def load_playoff_results() -> dict:
    """Load scraped playoff results and apply confirmed overrides."""
    path = os.path.join(RAW, "all_playoff_results.json")
    with open(path) as f:
        raw = json.load(f)
    results = {int(k): v for k, v in raw.items()}

    # Confirmed S9 results (override any scraper errors)
    results[9] = {
        "champion": "hackingnoises",
        "finalist": "doogile",
        "top4":     ["Pinne", "Infume"],
        "qf_exit":  ["steez", "Aquacorde", "lowk3y_", "BlazeMind"],
        "r1_exit":  ["edcr", "Feinberg", "nhb_", "silverrruns",
                     "BeefSalad", "nahhann", "HDMICables", "bing_pigs"],
    }
    return results


# ---------------------------------------------------------------------------
# Recency weights for training seasons
# ---------------------------------------------------------------------------
SEASON_WEIGHTS = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4, 8: 6}

# ---------------------------------------------------------------------------
# Feature names (11-dimensional matchup diff vector)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Elo diff",
    "WinRate diff",
    "RecentWR diff",
    "Consistency diff",
    "AvgTime diff (s)",
    "DeepRun diff",
    "Champion diff",
    "Finalist diff",
    "BestTime diff (s)",
    "ForfeitRate diff",
    "EloMomentum diff",
    "LCQ Flag diff",
    "Tournament Delta diff",
    "H2H WinRate",
]

# Playoff tier ordering (best → worst)
TIER_ORDER = ["champion", "finalist", "top4", "qf_exit", "r1_exit"]


# =============================================================================
# 1.  LOAD MATCH DATA
# =============================================================================

def load_all_matches() -> pd.DataFrame:
    """Load all ranked match rows for seasons 1-9 into a flat DataFrame."""
    frames = []
    for s in range(1, 10):
        fpath = os.path.join(RAW, f"season_{s}_matches.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            matches = json.load(f)
        for m in matches:
            players     = m.get("players", [])
            result      = m.get("result") or {}
            winner_uuid = result.get("uuid")
            win_time    = result.get("time")
            forfeited   = m.get("forfeited", False)
            if len(players) < 2:
                continue
            p1, p2 = players[0], players[1]
            frames.append({
                "match_id":    m.get("id"),
                "season":      m.get("season", s),
                "date":        m.get("date"),
                "p1_nick":     p1.get("nickname"),
                "p1_uuid":     p1.get("uuid"),
                "p1_elo":      p1.get("eloRate"),
                "p2_nick":     p2.get("nickname"),
                "p2_uuid":     p2.get("uuid"),
                "p2_elo":      p2.get("eloRate"),
                "winner_uuid": winner_uuid,
                "win_time_ms": win_time,
                "forfeited":   forfeited,
            })

    df = pd.DataFrame(frames).drop_duplicates("match_id")
    df["p1_won"] = df["winner_uuid"] == df["p1_uuid"]
    return df


# =============================================================================
# 2.  PLAYER FEATURE ENGINEERING (11 features per player)
# =============================================================================

def build_player_features(
    df: pd.DataFrame,
    players: list,
    playoff_results: dict,
    season_filter: int,
    pedigree_cutoff: int = None,
    lcq_by_season: dict = None,
    delta_by_season: dict = None,
    lcq_season: int = None,
) -> pd.DataFrame:
    """
    Compute 11 per-player features using only match data up to `season_filter`.

    Parameters
    ----------
    df              : full match DataFrame (all seasons)
    players         : list of nicknames to build features for
    playoff_results : dict[season -> results]
    season_filter   : use match data from seasons 1..season_filter only
    pedigree_cutoff : count playoff pedigree from seasons 1..pedigree_cutoff
                      (defaults to season_filter - 1, i.e. the season before
                      the current playoffs — correct for training seasons;
                      pass season_filter explicitly for the hold-out test
                      so that the most-recently completed season is included)
    """
    sub = df[df["season"] <= season_filter]
    if pedigree_cutoff is None:
        pedigree_cutoff = season_filter - 1   # training default: exclude current season
    pedigree_seasons = {k: v for k, v in playoff_results.items()
                        if k <= pedigree_cutoff}

    records = []
    for nick in players:
        as_p1 = sub[sub["p1_nick"] == nick]
        as_p2 = sub[sub["p2_nick"] == nick]

        wins   = int(as_p1["p1_won"].sum() + (~as_p2["p1_won"]).sum())
        losses = int((~as_p1["p1_won"]).sum() + as_p2["p1_won"].sum())
        total  = wins + losses
        win_rate = wins / total if total > 0 else np.nan

        # Completion times (non-forfeited wins)
        times_p1 = as_p1[as_p1["p1_won"] & ~as_p1["forfeited"]]["win_time_ms"]
        times_p2 = as_p2[~as_p2["p1_won"] & ~as_p2["forfeited"]]["win_time_ms"]
        all_times = pd.concat([times_p1, times_p2]).dropna()

        avg_time   = float(all_times.mean())  if len(all_times) > 0 else np.nan
        best_time  = float(all_times.min())   if len(all_times) > 0 else np.nan
        std_time   = float(all_times.std())   if len(all_times) > 1 else np.nan
        if not (np.isnan(std_time) if std_time is not np.nan else True):
            consistency = 1.0 / (std_time / 1000 + 1)
        else:
            consistency = np.nan

        # Recent form (last 20 matches)
        all_m = pd.concat([
            as_p1[["date", "p1_won"]].rename(columns={"p1_won": "won"}),
            as_p2[["date", "p1_won"]].rename(columns={"p1_won": "won"}).assign(
                won=lambda x: ~x["won"]
            ),
        ]).sort_values("date").tail(20)
        recent_wr = float(all_m["won"].mean()) if len(all_m) > 0 else win_rate

        # Forfeit rate
        total_rows   = len(as_p1) + len(as_p2)
        forfeit_rate = float(
            (as_p1["forfeited"].sum() + as_p2["forfeited"].sum()) / total_rows
        ) if total_rows > 0 else 0.0

        # Elo momentum (slope of Elo over last 20 appearances)
        elo_ts = pd.concat([
            as_p1[["date", "p1_elo"]].rename(columns={"p1_elo": "elo"}),
            as_p2[["date", "p2_elo"]].rename(columns={"p2_elo": "elo"}),
        ]).sort_values("date")["elo"].dropna()
        recent_elo   = elo_ts.tail(20)
        elo_momentum = float(
            (recent_elo.iloc[-1] - recent_elo.iloc[0]) / len(recent_elo)
        ) if len(recent_elo) > 1 else 0.0

        # Current Elo
        elo_p1 = as_p1.sort_values("date").tail(1)["p1_elo"]
        elo_p2 = as_p2.sort_values("date").tail(1)["p2_elo"]
        last_elo = float(elo_p1.values[-1]) if len(elo_p1) else (
                   float(elo_p2.values[-1]) if len(elo_p2) else 1500.0)

        # Tournament pedigree (completed seasons before season_filter)
        champion_count = sum(
            1 for v in pedigree_seasons.values() if v.get("champion") == nick
        )
        finalist_count = sum(
            1 for v in pedigree_seasons.values() if v.get("finalist") == nick
        )
        top4_count = sum(
            1 for v in pedigree_seasons.values() if nick in v.get("top4", [])
        )
        qf_count = sum(
            1 for v in pedigree_seasons.values() if nick in v.get("qf_exit", [])
        )
        deep_run_score = champion_count * 4 + finalist_count * 3 + top4_count * 2 + qf_count

        # LCQ qualifier flag — 1 if this player entered the target playoffs via
        # the Last-Chance Qualifier, 0 if they qualified directly.
        # lcq_season overrides season_filter so the S9 test set can use S9 LCQ
        # data (known before the tournament starts, so not leakage) while still
        # restricting match stats to S8.
        _lcq_season = lcq_season if lcq_season is not None else season_filter
        lcq_flag = 0
        if lcq_by_season is not None:
            lcq_flag = 1 if nick in lcq_by_season.get(_lcq_season, set()) else 0

        # Historical tournament performance delta — average of (seed_rank - actual_place)
        # across all playoff appearances up to pedigree_cutoff.
        # Positive = player historically outperforms their seeding.
        avg_delta = np.nan
        if delta_by_season is not None:
            past = [
                delta_by_season[s][nick]
                for s in range(1, pedigree_cutoff + 1)
                if s in delta_by_season and nick in delta_by_season[s]
            ]
            if past:
                avg_delta = float(np.mean(past))

        records.append({
            "nickname":              nick,
            "elo":                   last_elo,
            "win_rate":              win_rate,
            "recent_wr":             recent_wr,
            "consistency":           consistency,
            "avg_time_ms":           avg_time,
            "best_time_ms":          best_time,
            "forfeit_rate":          forfeit_rate,
            "elo_momentum":          elo_momentum,
            "champion_count":        champion_count,
            "finalist_count":        finalist_count,
            "deep_run_score":        deep_run_score,
            "lcq_flag":              lcq_flag,
            "avg_tournament_delta":  avg_delta,
        })

    return pd.DataFrame(records)


# =============================================================================
# 3.  MATCHUP FEATURE VECTOR (11-dim diff)
# =============================================================================

PLAYER_FEAT_COLS = [
    "elo", "win_rate", "recent_wr", "consistency", "avg_time_ms",
    "deep_run_score", "champion_count", "finalist_count",
    "best_time_ms", "forfeit_rate", "elo_momentum",
    "lcq_flag", "avg_tournament_delta",
]


def build_matchup_vector(feat: pd.DataFrame, p1: str, p2: str,
                         h2h: dict = None) -> np.ndarray:
    """Return the 14-dim feature diff vector (positive = p1 advantage).

    The three new dimensions (indices 11-13) are:
      11 — LCQ Flag diff       (1 = p1 is LCQ qualifier, -1 = p2, 0 = same)
      12 — Tournament Delta diff (p1 avg delta minus p2 avg delta)
      13 — H2H WinRate          (p1 career win rate vs p2, centered at 0)
    """
    r1 = feat[feat["nickname"] == p1]
    r2 = feat[feat["nickname"] == p2]
    if r1.empty or r2.empty:
        return np.full(len(FEATURE_NAMES), np.nan)

    r1, r2 = r1.iloc[0], r2.iloc[0]

    avg_time_diff  = 0.0
    best_time_diff = 0.0
    if not (np.isnan(r1["avg_time_ms"])  if pd.isna(r1["avg_time_ms"])  else False) and \
       not (np.isnan(r2["avg_time_ms"])  if pd.isna(r2["avg_time_ms"])  else False):
        avg_time_diff  = (r2["avg_time_ms"]  - r1["avg_time_ms"])  / 1000
    if not (pd.isna(r1["best_time_ms"])) and not (pd.isna(r2["best_time_ms"])):
        best_time_diff = (r2["best_time_ms"] - r1["best_time_ms"]) / 1000

    # LCQ flag diff — direct subtraction (0/1 values, never NaN)
    lcq_diff = float(r1.get("lcq_flag", 0)) - float(r2.get("lcq_flag", 0))

    # Tournament delta diff — NaN-safe
    d1 = r1.get("avg_tournament_delta", np.nan)
    d2 = r2.get("avg_tournament_delta", np.nan)
    delta_diff = float(d1 - d2) if not (pd.isna(d1) or pd.isna(d2)) else np.nan

    # H2H win rate centered at 0 — positive means p1 has beaten p2 more often
    h2h_val = 0.0
    if h2h is not None:
        wr = h2h.get((p1, p2), None)
        if wr is not None:
            h2h_val = wr - 0.5

    return np.array([
        r1["elo"]            - r2["elo"],
        r1["win_rate"]       - r2["win_rate"],
        r1["recent_wr"]      - r2["recent_wr"],
        r1["consistency"]    - r2["consistency"],
        avg_time_diff,
        r1["deep_run_score"] - r2["deep_run_score"],
        r1["champion_count"] - r2["champion_count"],
        r1["finalist_count"] - r2["finalist_count"],
        best_time_diff,
        r1["forfeit_rate"]   - r2["forfeit_rate"],
        r1["elo_momentum"]   - r2["elo_momentum"],
        lcq_diff,
        delta_diff,
        h2h_val,
    ], dtype=float)


# =============================================================================
# 4.  BUILD PAIRWISE DATASETS
# =============================================================================

def _expand_tier(tier_val):
    """Return a list of player names from a tier value (str or list)."""
    if tier_val is None:
        return []
    if isinstance(tier_val, str):
        return [tier_val]
    return list(tier_val)


def build_pairwise_data(
    feat_lookup: dict,      # season → DataFrame of player features
    results: dict,          # playoff_results dict
    seasons: list,
    season_weights: dict,
    all_players_set: set = None,
    h2h_by_season: dict = None,  # season → h2h lookup dict
):
    """
    Build cross-tier pairwise (p_better, p_worse) matchup vectors.

    For every (higher_tier, lower_tier) pair we add:
      - fv       → y=1  (p_better won)
      - -fv      → y=0  (p_worse won, from the other player's POV)

    Returns X, y, w, meta_list.
    meta_list contains dicts with keys: season, p1, p2, elo_diff, y
    """
    X, y, w, meta = [], [], [], []

    for season in seasons:
        if season not in results or season not in feat_lookup:
            continue

        feat   = feat_lookup[season]
        known  = set(feat["nickname"].dropna().values)
        weight = season_weights.get(season, 1)
        res    = results[season]
        h2h    = h2h_by_season.get(season) if h2h_by_season else None

        tiers = [_expand_tier(res.get(t)) for t in TIER_ORDER]

        for i, better_tier in enumerate(tiers):
            for j in range(i + 1, len(tiers)):
                worse_tier = tiers[j]
                for p_b in better_tier:
                    for p_w in worse_tier:
                        if p_b not in known or p_w not in known:
                            continue
                        fv = build_matchup_vector(feat, p_b, p_w, h2h=h2h)
                        elo_diff = float(fv[0])

                        # p_b (better tier) is the "winner" → y=1
                        X.append(fv)
                        y.append(1)
                        w.append(weight)
                        meta.append({"season": season, "p1": p_b, "p2": p_w,
                                     "elo_diff": elo_diff, "y": 1})

                        # Reverse direction → y=0
                        X.append(-fv)
                        y.append(0)
                        w.append(weight)
                        meta.append({"season": season, "p1": p_w, "p2": p_b,
                                     "elo_diff": -elo_diff, "y": 0})

    return (np.array(X, dtype=float),
            np.array(y),
            np.array(w, dtype=float),
            pd.DataFrame(meta))


# =============================================================================
# 5.  METRICS
# =============================================================================

def compute_upset_detection(X_test, y_test, y_pred):
    """
    Upset = lower-Elo player finishes in a BETTER playoff position.
    In our pairwise frame: y=1 (p1 finished better) but elo_diff < 0 (p1 had lower Elo).
    Equivalently: y=0 (p1 finished worse) but elo_diff > 0 (p1 had higher Elo).

    We capture upsets from both perspectives to avoid missing the mirror pair.
    """
    elo_diff = X_test[:, 0]

    # p1 had higher Elo but p2 placed higher (p1 lost → y=0)
    upset_mask_a = (elo_diff > 0) & (y_test == 0)
    # p1 had lower Elo but p1 placed higher (p1 won → y=1)
    upset_mask_b = (elo_diff < 0) & (y_test == 1)
    upset_mask   = upset_mask_a | upset_mask_b

    n_upsets = upset_mask.sum()
    if n_upsets == 0:
        print("\n-- Upset Detection: no upsets in test set --")
        return 0.0, 0, 0

    # For mask_a: correct detection means predicting 0 (p1 loses)
    correct_a = ((y_pred == 0) & upset_mask_a).sum()
    # For mask_b: correct detection means predicting 1 (p1 wins despite lower Elo)
    correct_b = ((y_pred == 1) & upset_mask_b).sum()
    correct_total = int(correct_a + correct_b)

    rate = correct_total / n_upsets
    return rate, int(n_upsets), correct_total


def print_metrics(y_test, y_pred, X_test):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION — SEASON 9 HOLD-OUT EVALUATION")
    print("=" * 60)
    print(f"  Test samples  : {len(y_test)}")
    print(f"  Class balance : {y_test.mean():.2f} (fraction where p1 wins)")
    print()
    print(f"  Accuracy      : {acc:.4f}  ({acc:.1%})")
    print(f"  Precision     : {prec:.4f}  (when model says 'p1 wins', how often correct)")
    print(f"  Recall        : {rec:.4f}  (of actual p1 wins, how many model caught)")
    print(f"  F1 Score      : {f1:.4f}")
    print()
    print("  Full classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["p2 wins (0)", "p1 wins (1)"],
                                zero_division=0))

    rate, n_upsets, n_correct = compute_upset_detection(X_test, y_test, y_pred)
    print("-- Upset Detection --")
    print(f"  Upsets in test set  : {n_upsets}")
    print(f"  Correctly predicted : {n_correct}")
    print(f"  Upset detection rate: {rate:.4f}  ({rate:.1%})")
    print("=" * 60)

    return {
        "accuracy":             acc,
        "precision":            prec,
        "recall":               rec,
        "f1":                   f1,
        "upset_detection_rate": rate,
        "n_upsets":             n_upsets,
    }


# =============================================================================
# 6.  VISUALIZATIONS
# =============================================================================

def plot_confusion_matrix(y_test, y_pred, out_path):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["p2 wins", "p1 wins"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Logistic Regression – Confusion Matrix\n"
                 "Train: S1–8  |  Test: S9 playoffs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {os.path.basename(out_path)}")


def plot_feature_importance(coef_series, out_path):
    coef = coef_series.sort_values(key=abs, ascending=True)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in coef.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(coef.index, coef.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("Feature Importance – Logistic Regression\n"
                 "Red = favors p1 (higher value → p1 wins more)  |  "
                 "Blue = favors p2")

    for bar, val in zip(bars, coef.values):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=8)

    red_patch  = mpatches.Patch(color="#e74c3c", label="Favors p1 (higher diff = p1 advantage)")
    blue_patch = mpatches.Patch(color="#3498db", label="Favors p2 (negative coef)")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {os.path.basename(out_path)}")


def plot_season_coverage(meta_train, meta_test, out_path):
    """Bar chart: training samples per season, plus test season."""
    train_counts = meta_train.groupby("season").size()
    test_count   = len(meta_test)

    fig, ax = plt.subplots(figsize=(9, 4))
    seasons = list(train_counts.index)
    counts  = list(train_counts.values)
    colors  = ["#3498db"] * len(seasons) + ["#e74c3c"]
    all_s   = seasons + [9]
    all_c   = counts  + [test_count]

    bars = ax.bar([str(s) for s in all_s], all_c, color=colors, edgecolor="white")
    ax.set_xlabel("Season")
    ax.set_ylabel("Pairwise Samples")
    ax.set_title("Pairwise Training Samples per Season\n"
                 "Blue = training (S1–8)  |  Red = test (S9)")

    blue_p = mpatches.Patch(color="#3498db", label="Training seasons (S1–8)")
    red_p  = mpatches.Patch(color="#e74c3c", label="Test season (S9)")
    ax.legend(handles=[blue_p, red_p])

    for bar, cnt in zip(bars, all_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {os.path.basename(out_path)}")


# =============================================================================
# 7.  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Loading playoff results...")
    playoff_results = load_playoff_results()
    for s in sorted(playoff_results):
        r = playoff_results[s]
        print(f"  S{s}: champ={r['champion']!s:<16} finalist={r['finalist']}")

    print("\nLoading match data...")
    df = load_all_matches()
    print(f"  {len(df)} unique matches (S1–S9)")

    # ------------------------------------------------------------------
    # Load new feature data (LCQ flags, tournament deltas, H2H lookups)
    # ------------------------------------------------------------------
    print("\nLoading LCQ qualifiers and tournament deltas...")
    lcq_by_season   = load_lcq_by_season()
    delta_by_season = load_delta_by_season()
    print(f"  LCQ players per season: { {s: len(v) for s, v in lcq_by_season.items()} }")

    print("Building per-season H2H lookups (no leakage)...")
    h2h_by_season_train = {s: build_h2h_lookup(df, s) for s in range(1, 9)}
    h2h_s9_test         = build_h2h_lookup(df, 8)   # S9 test uses only data through S8
    print("  H2H lookups built for S1–S8 training and S9 test hold-out")

    # ------------------------------------------------------------------
    # Collect all player names that appear in any S1-9 playoff
    # ------------------------------------------------------------------
    all_playoff_players = set()
    for v in playoff_results.values():
        for tier in TIER_ORDER:
            all_playoff_players.update(_expand_tier(v.get(tier)))

    print(f"\n  Unique playoff players found : {len(all_playoff_players)}")

    # ------------------------------------------------------------------
    # Build feature snapshots for each training season
    # (features computed from match data up to and INCLUDING that season)
    # This is what the model would have seen *before* season s playoffs.
    # For season s training label, we use features from data through season s-1
    # (pedigree) but match stats through season s.
    # ------------------------------------------------------------------
    print("\nBuilding per-season player feature snapshots (S1–S8)...")
    feat_by_season_train = {}
    for s in range(1, 9):
        feat_by_season_train[s] = build_player_features(
            df=df,
            players=list(all_playoff_players),
            playoff_results=playoff_results,
            season_filter=s,
            lcq_by_season=lcq_by_season,
            delta_by_season=delta_by_season,
        )
        n_known = feat_by_season_train[s]["nickname"].notna().sum()
        print(f"  S{s}: {n_known} players with features")

    # S9 test features: use data only through S8 (no leakage)
    # pedigree_cutoff=8 so S8's completed playoff results are visible
    print("\nBuilding S9 test player features (using data through S8 only)...")
    s9_players = _expand_tier(playoff_results[9].get("champion")) + \
                 _expand_tier(playoff_results[9].get("finalist")) + \
                 _expand_tier(playoff_results[9].get("top4")) + \
                 _expand_tier(playoff_results[9].get("qf_exit")) + \
                 _expand_tier(playoff_results[9].get("r1_exit"))

    feat_s9_test = build_player_features(
        df=df,
        players=s9_players,
        playoff_results=playoff_results,
        season_filter=8,        # NO Season 9 match data → strict hold-out
        pedigree_cutoff=8,      # S8 results ARE known before S9 playoffs
        lcq_by_season=lcq_by_season,
        delta_by_season=delta_by_season,
        lcq_season=9,           # S9 bracket seeding is public before the tournament
    )
    print(f"  S9 test players: {len(feat_s9_test)}")
    print(f"\n  S9 player snapshot (sorted by Elo):")
    print(f"  {'Player':<22} {'Elo':>5}  {'WR':>6}  {'Consistency':>11}  "
          f"{'DeepRun':>7}  {'Champ':>5}  {'Matches':>7}")
    print("  " + "-" * 72)
    for _, row in feat_s9_test.sort_values("elo", ascending=False).iterrows():
        wr_str = f"{row['win_rate']:.1%}" if not pd.isna(row["win_rate"]) else "  N/A"
        cs_str = f"{row['consistency']:.4f}" if not pd.isna(row["consistency"]) else "     N/A"
        print(f"  {row['nickname']:<22} {row['elo']:>5.0f}  {wr_str:>6}  "
              f"{cs_str:>11}  {row['deep_run_score']:>7}  "
              f"{row['champion_count']:>5}  {row['win_rate'] if not pd.isna(row['win_rate']) else 0:>7.0%}")

    # ------------------------------------------------------------------
    # Build pairwise training data (S1–S8)
    # ------------------------------------------------------------------
    print("\nBuilding pairwise training data (S1–S8)...")
    X_train, y_train, w_train, meta_train = build_pairwise_data(
        feat_lookup=feat_by_season_train,
        results=playoff_results,
        seasons=list(range(1, 9)),
        season_weights=SEASON_WEIGHTS,
        h2h_by_season=h2h_by_season_train,
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Class balance    : {y_train.mean():.2f}")

    # Count pairs per season
    print(f"  Samples by season (after both-direction expansion):")
    for s, cnt in meta_train.groupby("season").size().items():
        print(f"    S{s}: {cnt} pairs  (weight={SEASON_WEIGHTS.get(s, 1)})")

    # ------------------------------------------------------------------
    # Build pairwise test data (S9 only)
    # ------------------------------------------------------------------
    print("\nBuilding pairwise test data (S9)...")
    # For test, we use a single feat snapshot (features through S8)
    feat_test_lookup = {9: feat_s9_test}
    X_test, y_test, _, meta_test = build_pairwise_data(
        feat_lookup=feat_test_lookup,
        results={9: playoff_results[9]},
        seasons=[9],
        season_weights={9: 1},
        h2h_by_season={9: h2h_s9_test},
    )
    print(f"  Test samples : {len(X_test)}")
    print(f"  Class balance: {y_test.mean():.2f}")
    print(f"  Pairs excluded (player not in S1–S8 data): "
          f"{len(s9_players)**2 // 2 - len(X_test) // 2} potential pairs skipped")

    if len(X_test) == 0:
        print("\nERROR: No test samples generated — check S9 player features.")
        return

    # ------------------------------------------------------------------
    # Train Logistic Regression
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression (with recency sample weights)...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=2000, C=0.5,
                                       class_weight="balanced",
                                       random_state=42)),
    ])
    pipeline.fit(X_train, y_train, clf__sample_weight=w_train)
    print("  Model trained.")
    print(f"  Training accuracy (in-sample): "
          f"{accuracy_score(y_train, pipeline.predict(X_train)):.1%}")

    # ------------------------------------------------------------------
    # Evaluate on S9 test set
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    metrics = print_metrics(y_test, y_pred, X_test)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    coef = pd.Series(
        pipeline.named_steps["clf"].coef_[0],
        index=FEATURE_NAMES,
    )
    print("\n-- Feature Importance (LR Coefficients) --")
    for name, val in coef.sort_values(key=abs, ascending=False).items():
        direction = "favors higher" if val > 0 else "favors lower"
        bar = "#" * int(abs(val) * 20)
        print(f"  {name:<26}  coef={val:+.4f}  ({direction})  {bar}")

    # ------------------------------------------------------------------
    # Inspect specific S9 matchup predictions
    # ------------------------------------------------------------------
    print("\n-- Sample S9 Matchup Predictions --")
    print(f"  {'P1':<22} {'P2':<22} {'True':>5}  {'Pred':>5}  {'P(p1 wins)':>11}  {'Elo diff':>9}")
    print("  " + "-" * 84)
    sample = meta_test.copy()
    sample["y_pred"]   = y_pred
    sample["proba_p1"] = pipeline.predict_proba(X_test)[:, 1]
    sample["elo_diff"] = X_test[:, 0]

    # Show only the "canonical" direction (p1 is the better-tiered player, y=1)
    shown = sample[sample["y"] == 1].head(20)
    for _, row in shown.iterrows():
        correct = "OK" if row["y_pred"] == row["y"] else "WRONG"
        upset   = " <- UPSET" if row["elo_diff"] < 0 else ""
        print(f"  {row['p1']:<22} {row['p2']:<22} {int(row['y']):>5}  "
              f"{int(row['y_pred']):>5}  {row['proba_p1']:>11.2%}  "
              f"{row['elo_diff']:>+9.0f}  {correct}{upset}")

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    print("\nSaving plots...")
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(OUT_DIR, "lr_confusion_matrix.png"),
    )
    plot_feature_importance(
        coef,
        os.path.join(OUT_DIR, "lr_feature_importance.png"),
    )
    plot_season_coverage(
        meta_train, meta_test,
        os.path.join(OUT_DIR, "lr_season_coverage.png"),
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model     : Logistic Regression  (C=0.5, balanced, max_iter=2000)")
    print(f"  Train set : Seasons 1–8  ({len(X_train)} pairwise samples)")
    print(f"  Test set  : Season 9     ({len(X_test)} pairwise samples)")
    print(f"  Features  : {len(FEATURE_NAMES)} matchup diff features")
    print()
    print(f"  Accuracy              : {metrics['accuracy']:.4f}  ({metrics['accuracy']:.1%})")
    print(f"  Precision (p1 wins=1) : {metrics['precision']:.4f}")
    print(f"  Recall    (p1 wins=1) : {metrics['recall']:.4f}")
    print(f"  F1 Score              : {metrics['f1']:.4f}")
    print(f"  Upset Detection Rate  : {metrics['upset_detection_rate']:.4f}  "
          f"({metrics['n_upsets']} upsets in test set)")
    print("=" * 60)

    return pipeline, metrics


if __name__ == "__main__":
    main()
