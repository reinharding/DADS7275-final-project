"""
Season 10 Playoff Prediction
Models: Logistic Regression (H2H), LDA (outcome classification), PCA (feature exploration).
Evaluations: accuracy, precision/recall, Top-K accuracy, upset detection rate.
"""

import subprocess, sys, importlib
if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
for pkg in ["pandas", "numpy", "scikit-learn", "requests", "matplotlib"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        importlib.invalidate_caches()

import json, os, time, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

RAW     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
API     = "https://api.mcsrranked.com"
os.makedirs(OUT_DIR, exist_ok=True)

# -- Season 10 projected playoff pool --
# lowk3y_ included as a projected LCQ qualifier (currently inactive but expected to return)
S10_POOL = [
    "Infume", "edcr", "doogile", "Feinberg", "7rowl",
    "bing_pigs", "nahhann", "BlazeMind", "Aquacorde", "silverrruns",
    "BeefSalad", "meebie",
    "hackingnoises", "steez", "nhb_", "Ancoboyy",
    "lowk3y_",
]

# -- Confirmed overrides (used to patch/correct scraped data) --
# Only entries here will overwrite what the scraper found.
PLAYOFF_OVERRIDES = {
    9: {
        "champion": "hackingnoises",
        "finalist": "doogile",
        "top4":     ["Pinne", "Infume"],
        "qf_exit":  ["steez", "Aquacorde", "lowk3y_", "BlazeMind"],
        "r1_exit":  ["edcr", "Feinberg", "nhb_", "silverrruns",
                     "BeefSalad", "nahhann", "HDMICables", "bing_pigs"],
    },
}


def load_playoff_results() -> dict:
    """
    Load playoff results from scraped JSON files (data/raw/all_playoff_results.json).
    Falls back to PLAYOFF_OVERRIDES for any season not found or missing fields.
    Applies PLAYOFF_OVERRIDES as a patch on top of scraped data.
    """
    scraped_path = os.path.join(RAW, "all_playoff_results.json")
    results = {}

    if os.path.exists(scraped_path):
        with open(scraped_path) as f:
            raw = json.load(f)
        # JSON keys are strings; convert to int
        for k, v in raw.items():
            results[int(k)] = {
                "champion": v.get("champion"),
                "finalist": v.get("finalist"),
                "top4":     v.get("top4", []),
                "qf_exit":  v.get("qf_exit", []),
                "r1_exit":  v.get("r1_exit", []),
            }
        print(f"Loaded playoff results for seasons: {sorted(results.keys())}")
    else:
        print("WARNING: all_playoff_results.json not found. "
              "Run: python scraper.py playoffs")

    # Apply overrides (patches confirmed data on top of scraped data)
    for season, override in PLAYOFF_OVERRIDES.items():
        if season not in results:
            results[season] = {"champion": None, "finalist": None,
                               "top4": [], "qf_exit": [], "r1_exit": []}
        for field, val in override.items():
            if val is not None and val != [] and val != "":
                results[season][field] = val

    return results


# Populated at runtime by load_playoff_results()
PLAYOFF_RESULTS: dict = {}

# ---------------------------------------------------------------------------
# Recency weights — manually tune these to emphasise recent seasons.
# Higher number = that season's matchup pairs count more during training.
# S9 is the most recent completed playoff; S1 is the oldest.
# ---------------------------------------------------------------------------
SEASON_WEIGHTS = {
    1: 1,   # oldest — low weight
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 4,
    8: 6,
    9: 10,   # most recent completed playoff — highest weight
}

# LDA outcome labels
OUTCOME_LABEL = {"champion": 4, "finalist": 3, "top4": 2, "qf_exit": 1, "r1_exit": 0}
OUTCOME_NAME  = {4: "Champion", 3: "Finalist", 2: "Top 4", 1: "QF Exit", 0: "R1 Exit"}

# Matchup feature names (diffs: positive = p1 advantage)
MATCHUP_FEAT_NAMES = [
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
]

# Individual player feature columns used for LDA / PCA
PLAYER_FEAT_COLS = [
    "elo", "win_rate", "recent_wr", "consistency", "avg_time_ms",
    "deep_run_score", "champion_count", "finalist_count",
    "best_time_ms", "forfeit_rate", "elo_momentum",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_matches():
    frames = []
    for s in range(1, 10):
        fpath = os.path.join(RAW, f"season_{s}_matches.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            matches = json.load(f)
        for m in matches:
            players = m.get("players", [])
            result  = m.get("result") or {}
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
# FEATURE ENGINEERING (11 features per player)
# =============================================================================

def build_player_features(df: pd.DataFrame, players: list, season_filter=None) -> pd.DataFrame:
    if season_filter is not None:
        df = df[df["season"] <= season_filter]

    records = {}
    for nick in players:
        as_p1 = df[df["p1_nick"] == nick]
        as_p2 = df[df["p2_nick"] == nick]

        wins   = as_p1["p1_won"].sum() + (~as_p2["p1_won"]).sum()
        losses = (~as_p1["p1_won"]).sum() + as_p2["p1_won"].sum()
        total  = wins + losses
        win_rate = wins / total if total > 0 else 0.5

        # Completion times (non-forfeited wins only)
        times_p1 = as_p1[as_p1["p1_won"] & ~as_p1["forfeited"]]["win_time_ms"]
        times_p2 = as_p2[~as_p2["p1_won"] & ~as_p2["forfeited"]]["win_time_ms"]
        all_times = pd.concat([times_p1, times_p2]).dropna()

        avg_time  = all_times.mean() if len(all_times) > 0 else np.nan
        best_time = all_times.min()  if len(all_times) > 0 else np.nan
        std_time  = all_times.std()  if len(all_times) > 1 else np.nan
        consistency = 1 / (std_time / 1000 + 1) if not (std_time is np.nan or np.isnan(std_time)) else 0.5

        # Recent form (last 20 matches win rate)
        all_m = pd.concat([
            as_p1[["date","p1_won"]].rename(columns={"p1_won": "won"}),
            as_p2[["date","p1_won"]].rename(columns={"p1_won": "won"}).assign(won=lambda x: ~x["won"])
        ]).sort_values("date").tail(20)
        recent_wr = all_m["won"].mean() if len(all_m) > 0 else win_rate

        # Forfeit rate
        total_rows = len(as_p1) + len(as_p2)
        forfeit_rate = ((as_p1["forfeited"].sum() + as_p2["forfeited"].sum()) / total_rows
                        if total_rows > 0 else 0.0)

        # Elo momentum (average change over last 20 appearances)
        elo_ts = pd.concat([
            as_p1[["date","p1_elo"]].rename(columns={"p1_elo": "elo"}),
            as_p2[["date","p2_elo"]].rename(columns={"p2_elo": "elo"}),
        ]).sort_values("date")["elo"].dropna()
        recent_elo = elo_ts.tail(20)
        elo_momentum = float((recent_elo.iloc[-1] - recent_elo.iloc[0]) / len(recent_elo)) \
                       if len(recent_elo) > 1 else 0.0

        # Current Elo (latest appearance)
        elo_p1 = as_p1.sort_values("date").tail(1)["p1_elo"]
        elo_p2 = as_p2.sort_values("date").tail(1)["p2_elo"]
        last_elo = float(elo_p1.values[-1]) if len(elo_p1) else (
                   float(elo_p2.values[-1]) if len(elo_p2) else 1500)

        # Tournament pedigree (confirmed API data only)
        champion_count = sum(1 for s in PLAYOFF_RESULTS.values() if s.get("champion") == nick)
        finalist_count = sum(1 for s in PLAYOFF_RESULTS.values() if s.get("finalist") == nick)
        top4_count     = sum(1 for s in PLAYOFF_RESULTS.values() if nick in s.get("top4", []))
        qf_count       = sum(1 for s in PLAYOFF_RESULTS.values() if nick in s.get("qf_exit", []))
        deep_run_score = champion_count * 4 + finalist_count * 3 + top4_count * 2 + qf_count

        records[nick] = {
            "nickname":       nick,
            "elo":            last_elo,
            "win_rate":       round(win_rate, 4),
            "total_matches":  int(total),
            "avg_time_ms":    round(avg_time, 1) if not (avg_time is np.nan or np.isnan(avg_time)) else None,
            "best_time_ms":   round(best_time, 1) if not (best_time is np.nan or np.isnan(best_time)) else None,
            "consistency":    round(consistency, 4),
            "recent_wr":      round(recent_wr, 4),
            "forfeit_rate":   round(forfeit_rate, 4),
            "elo_momentum":   round(elo_momentum, 4),
            "champion_count": champion_count,
            "finalist_count": finalist_count,
            "top4_count":     top4_count,
            "deep_run_score": deep_run_score,
        }
    return pd.DataFrame(records.values())


# =============================================================================
# MATCHUP FEATURE VECTOR (11-dimensional diffs)
# =============================================================================

def build_matchup_features(feat: pd.DataFrame, p1: str, p2: str) -> np.ndarray:
    r1 = feat[feat["nickname"] == p1].iloc[0]
    r2 = feat[feat["nickname"] == p2].iloc[0]

    avg_time_diff  = 0.0
    best_time_diff = 0.0
    if r1["avg_time_ms"] and r2["avg_time_ms"]:
        avg_time_diff  = (r2["avg_time_ms"]  - r1["avg_time_ms"])  / 1000
    if r1["best_time_ms"] and r2["best_time_ms"]:
        best_time_diff = (r2["best_time_ms"] - r1["best_time_ms"]) / 1000

    return np.array([
        r1["elo"]           - r2["elo"],
        r1["win_rate"]      - r2["win_rate"],
        r1["recent_wr"]     - r2["recent_wr"],
        r1["consistency"]   - r2["consistency"],
        avg_time_diff,
        r1["deep_run_score"] - r2["deep_run_score"],
        r1["champion_count"] - r2["champion_count"],
        r1["finalist_count"] - r2["finalist_count"],
        best_time_diff,
        r1["forfeit_rate"]  - r2["forfeit_rate"],     # negative = p1 forfeits less
        r1["elo_momentum"]  - r2["elo_momentum"],
    ], dtype=float)


# =============================================================================
# TRAINING DATA (pairwise tier comparisons)
# =============================================================================

def build_training_data(df: pd.DataFrame, feat_by_season: dict):
    X, y, w = [], [], []
    tier_order = ["champion", "finalist", "top4", "qf_exit", "r1_exit"]

    for season, results in PLAYOFF_RESULTS.items():
        if season not in feat_by_season:
            continue
        feat   = feat_by_season[season]
        known  = set(feat["nickname"].values)
        weight = SEASON_WEIGHTS.get(season, 1)   # recency weight for this season

        tiers = []
        for t in tier_order:
            val = results.get(t)
            if val is None:
                tiers.append([])
            elif isinstance(val, str):
                tiers.append([val])
            else:
                tiers.append(list(val))

        for i, better in enumerate(tiers):
            for j in range(i + 1, len(tiers)):
                worse = tiers[j]
                for p_b in better:
                    for p_w in worse:
                        if p_b not in known or p_w not in known:
                            continue
                        fv = build_matchup_features(feat, p_b, p_w)
                        X.append(fv);  y.append(1);  w.append(weight)
                        X.append(-fv); y.append(0);  w.append(weight)

    return np.array(X), np.array(y), np.array(w, dtype=float)


# =============================================================================
# LDA: per-player outcome classification
# =============================================================================

def build_player_outcome_data(feat_by_season: dict):
    X, y, names = [], [], []
    tier_order  = ["champion", "finalist", "top4", "qf_exit", "r1_exit"]

    for season, results in PLAYOFF_RESULTS.items():
        if season not in feat_by_season:
            continue
        feat  = feat_by_season[season]
        known = set(feat["nickname"].values)

        for tier in tier_order:
            label = OUTCOME_LABEL[tier]
            val   = results.get(tier)
            if val is None:
                players_in_tier = []
            elif isinstance(val, str):
                players_in_tier = [val]
            else:
                players_in_tier = list(val)

            for p in players_in_tier:
                if p not in known:
                    continue
                row = feat[feat["nickname"] == p].iloc[0]
                fv  = [row[c] if row[c] is not None else np.nan for c in PLAYER_FEAT_COLS]
                X.append(fv)
                y.append(label)
                names.append(p)

    return np.array(X, dtype=float), np.array(y), names


def train_lda(feat_by_season: dict):
    X, y, names = build_player_outcome_data(feat_by_season)
    if len(X) == 0:
        return None, None, None, None

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    pipe.fit(X, y)

    acc = accuracy_score(y, pipe.predict(X))
    print(f"\n-- LDA (outcome classification) --")
    print(f"  Training samples : {len(X)}")
    print(f"  Classes          : {sorted(set(y))}")
    print(f"  Training accuracy: {acc:.1%}  (in-sample; limited data)")
    return pipe, X, y, names


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def topk_accuracy(champ_counts: dict, n_sims: int, actual_champion: str, k: int = 3) -> bool:
    ranked = sorted(champ_counts, key=champ_counts.get, reverse=True)
    in_topk = actual_champion in ranked[:k]
    pct     = champ_counts.get(actual_champion, 0) / n_sims
    print(f"\n-- Top-{k} Accuracy (S9 hold-out) --")
    print(f"  Actual S9 champion : hackingnoises")
    print(f"  Predicted rank     : #{ranked.index(actual_champion)+1 if actual_champion in ranked else 'N/A'}")
    print(f"  Champion%          : {pct:.1%}")
    print(f"  In Top-{k}?         : {'YES' if in_topk else 'NO'}")
    return in_topk


def upset_detection_rate(X_test: np.ndarray, y_test: np.ndarray,
                         y_pred: np.ndarray) -> float:
    """
    Upset = lower-Elo player wins (y=0 when elo_diff > 0, i.e. p1 had higher Elo but lost).
    Upset detection rate = fraction of true upsets correctly predicted.
    """
    upsets_true = (y_test == 0) & (X_test[:, 0] > 0)  # p1 higher Elo but p2 won
    if upsets_true.sum() == 0:
        print("\n-- Upset Detection: no upsets in test set --")
        return 0.0
    upset_correct = (y_pred == 0)[upsets_true]
    rate = upset_correct.mean()
    print(f"\n-- Upset Detection Rate (S9 hold-out) --")
    print(f"  Upsets in test set : {upsets_true.sum()}")
    print(f"  Correctly detected : {upset_correct.sum()}")
    print(f"  Detection rate     : {rate:.1%}")
    return rate


# =============================================================================
# PREDICTION
# =============================================================================

def predict_h2h(pipeline, feat, p1, p2):
    fv = build_matchup_features(feat, p1, p2).reshape(1, -1)
    return pipeline.predict_proba(fv)[0][1]


def simulate_bracket(pipeline, feat, players, n_sims=10000):
    seeds = (feat[feat["nickname"].isin(players)]
             .sort_values("elo", ascending=False)["nickname"].tolist())
    while len(seeds) < 16:
        seeds.append(seeds[-1])
    seeds = seeds[:16]

    champ_counts = defaultdict(int)
    top4_counts  = defaultdict(int)

    for _ in range(n_sims):
        survivors = []
        for i in range(8):
            p1, p2 = seeds[i], seeds[15 - i]
            prob   = predict_h2h(pipeline, feat, p1, p2)
            survivors.append(p1 if np.random.random() < prob else p2)

        qf_winners = []
        for i in range(0, 8, 2):
            p1, p2 = survivors[i], survivors[i + 1]
            prob   = predict_h2h(pipeline, feat, p1, p2)
            qf_winners.append(p1 if np.random.random() < prob else p2)

        sf_winners = []
        for i in range(0, 4, 2):
            p1, p2 = qf_winners[i], qf_winners[i + 1]
            prob   = predict_h2h(pipeline, feat, p1, p2)
            winner = p1 if np.random.random() < prob else p2
            sf_winners.append(winner)
            for p in [p1, p2]:
                top4_counts[p] += 1

        prob  = predict_h2h(pipeline, feat, sf_winners[0], sf_winners[1])
        champ = sf_winners[0] if np.random.random() < prob else sf_winners[1]
        champ_counts[champ] += 1

    return champ_counts, top4_counts, n_sims


# =============================================================================
# MAIN
# =============================================================================

def main():
    global PLAYOFF_RESULTS
    PLAYOFF_RESULTS = load_playoff_results()

    # Print what we loaded so you can verify
    print("\n-- Playoff Results Loaded --")
    for s in sorted(PLAYOFF_RESULTS):
        r = PLAYOFF_RESULTS[s]
        print(f"  S{s}: champion={r['champion']} | finalist={r['finalist']} | "
              f"top4={r['top4']} | qf={r['qf_exit']}")
    print()

    print("Loading match data...")
    df = load_all_matches()
    print(f"  {len(df)} total matches across seasons 1-9\n")

    all_players_hist = list({nick for col in ["p1_nick","p2_nick"] for nick in df[col].dropna()})
    feat_by_season   = {s: build_player_features(df, all_players_hist, season_filter=s)
                        for s in range(1, 10)}

    print("Building training data...")
    X, y, w = build_training_data(df, feat_by_season)
    print(f"  {len(X)} matchup samples | class balance: {y.mean():.2f}")
    print(f"  Season weights applied: {SEASON_WEIGHTS}\n")

    split = len(X) - 12
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train         = w[:split]

    # --- Logistic Regression (with recency sample weights) ---
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train, clf__sample_weight=w_train)
    y_pred = lr_pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression - S9 hold-out accuracy: {acc:.1%}")
    print(classification_report(y_test, y_pred,
                                 target_names=["p2 wins","p1 wins"], zero_division=0))

    upset_detection_rate(X_test, y_test, y_pred)

    # --- LDA ---
    lda_pipe, X_lda, y_lda, lda_names = train_lda(feat_by_season)

    # --- Season 10 player features ---
    print("\nBuilding Season 10 player features...")
    try:
        r  = requests.get(f"{API}/leaderboard", timeout=10)
        lb = {u["nickname"]: u["eloRate"] for u in r.json()["data"]["users"]}
    except Exception:
        lb = {}

    feat_s10 = build_player_features(df, S10_POOL)
    for i, row in feat_s10.iterrows():
        if row["nickname"] in lb:
            feat_s10.at[i, "elo"] = lb[row["nickname"]]

    print(f"\n{'Player':<22} {'Elo':>5}  {'WR':>6}  {'Consistency':>11}  "
          f"{'RecentWR':>8}  {'DeepRun':>7}  {'Champ':>5}  {'Finalist':>8}  {'EloMom':>6}")
    print("-" * 95)
    for _, row in feat_s10.sort_values("elo", ascending=False).iterrows():
        print(f"{row['nickname']:<22} {row['elo']:>5.0f}  {row['win_rate']:>6.1%}  "
              f"{row['consistency']:>11.4f}  {row['recent_wr']:>8.1%}  "
              f"{row['deep_run_score']:>7}  {row['champion_count']:>5}  "
              f"{row['finalist_count']:>8}  {row['elo_momentum']:>6.1f}")

    # --- LDA prediction for S10 players ---
    if lda_pipe is not None:
        print(f"\n-- LDA Outcome Prediction (S10 Players) --")
        print(f"  {'Player':<22} {'Pred. Outcome':<16}", end="")
        classes = lda_pipe.named_steps["lda"].classes_
        for c in classes:
            print(f"  {OUTCOME_NAME[c]:>9}", end="")
        print()
        print("-" * 90)
        lda_rows = []
        for _, row in feat_s10.sort_values("elo", ascending=False).iterrows():
            nick = row["nickname"]
            fv   = np.array([row[c] if row[c] is not None else np.nan
                             for c in PLAYER_FEAT_COLS], dtype=float).reshape(1, -1)
            pred_class  = lda_pipe.predict(fv)[0]
            pred_probas = lda_pipe.predict_proba(fv)[0]
            print(f"  {nick:<22} {OUTCOME_NAME[pred_class]:<16}", end="")
            for p in pred_probas:
                print(f"  {p:>9.1%}", end="")
            print()
            lda_rows.append({"player": nick, "lda_pred": OUTCOME_NAME[pred_class],
                             **{OUTCOME_NAME[c]: p for c, p in zip(classes, pred_probas)}})
        lda_df = pd.DataFrame(lda_rows)

    # --- H2H matrix ---
    top8 = feat_s10.sort_values("elo", ascending=False)["nickname"].tolist()[:8]
    print(f"\n-- Head-to-Head Win Probabilities (top 8 by Elo) --")
    print(f"{'':>18}" + "".join(f"{p:>14}" for p in top8))
    for p1 in top8:
        row_str = f"{p1:>18}"
        for p2 in top8:
            if p1 == p2:
                row_str += f"{'---':>14}"
            else:
                prob = predict_h2h(lr_pipe, feat_s10, p1, p2)
                row_str += f"{prob:>13.1%} "
        print(row_str)

    # --- Monte Carlo bracket ---
    print(f"\n-- Bracket Simulation (10,000 runs) --")
    champ_counts, top4_counts, n = simulate_bracket(lr_pipe, feat_s10, S10_POOL)

    results = pd.DataFrame([
        {"Player": p, "Champion%": champ_counts[p]/n, "Top4%": top4_counts[p]/n}
        for p in S10_POOL
    ]).sort_values("Champion%", ascending=False)

    print(f"\n{'Rank':<5} {'Player':<22} {'Champion%':>10}  {'Top4%':>8}")
    print("-" * 50)
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        bar = "#" * int(row["Champion%"] * 40)
        print(f"{rank:<5} {row['Player']:<22} {row['Champion%']:>9.1%}   {row['Top4%']:>8.1%}  {bar}")

    winner = results.iloc[0]["Player"]
    print(f"\n>> PREDICTED SEASON 10 CHAMPION: {winner} <<")
    print(f"   (wins in {results.iloc[0]['Champion%']:.1%} of simulations)")

    # Simulate S9 bracket for Top-K evaluation
    print("\n-- Simulating S9 bracket for Top-K evaluation --")
    s9_players = ["hackingnoises","doogile","Pinne","Infume","steez","Aquacorde",
                  "lowk3y_","BlazeMind","edcr","Feinberg","nhb_","silverrruns",
                  "BeefSalad","nahhann","HDMICables","bing_pigs"]
    feat_s9 = feat_by_season[9]
    champ_s9, _, n9 = simulate_bracket(lr_pipe, feat_s9, s9_players, n_sims=5000)
    topk_accuracy(champ_s9, n9, "hackingnoises", k=3)

    # Feature importance
    feat_names = MATCHUP_FEAT_NAMES
    coef = pd.Series(lr_pipe.named_steps["clf"].coef_[0],
                     index=feat_names).sort_values(key=abs, ascending=False)
    print("\n-- Logistic Regression Feature Importance --")
    for name, val in coef.items():
        direction = "favors higher" if val > 0 else "favors lower"
        print(f"  {name:<24}  coef={val:+.3f}  ({direction})")

    # Visualizations
    lda_df_arg = lda_df if lda_pipe is not None else None
    _make_charts(results, coef, feat_s10, lda_df_arg,
                 X_lda if lda_pipe else None,
                 y_lda if lda_pipe else None,
                 lda_names if lda_pipe else None)
    print(f"\nCharts saved to: {OUT_DIR}")


# =============================================================================
# CHARTS
# =============================================================================

def _make_charts(results, coef, feat, lda_df, X_lda, y_lda, lda_names):

    # 1. Champion probability bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if r > 0.1 else "#3498db" if r > 0.03 else "#95a5a6"
              for r in results["Champion%"]]
    ax.bar(results["Player"], results["Champion%"] * 100, color=colors, edgecolor="white")
    ax.bar(results["Player"], results["Top4%"] * 100,
           color=[c + "44" for c in colors], edgecolor="none")
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="Frontrunner (>10% to win)"),
        mpatches.Patch(color="#3498db", label="Contender (3-10%)"),
        mpatches.Patch(color="#95a5a6", label="Dark horse (<3%)"),
        mpatches.Patch(color="#cccccc", label="Faded = Top4%"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    ax.set_title("Season 10 Playoff Predictions - Champion & Top4 Probability", fontsize=13)
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Player (sorted by Champion%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "s10_champion_probs.png"), dpi=150)
    plt.close()

    # 2. Feature importance (Logistic Regression)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_feat = ["#27ae60" if v > 0 else "#e74c3c" for v in coef.values]
    ax.barh(coef.index[::-1], coef.values[::-1], color=colors_feat[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    pos_patch = mpatches.Patch(color="#27ae60", label="Positive: higher value = better odds")
    neg_patch = mpatches.Patch(color="#e74c3c", label="Negative: lower value = better odds")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", framealpha=0.9)
    ax.set_title("Logistic Regression Feature Importance\n(|coefficient| = influence strength)", fontsize=12)
    ax.set_xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "s10_feature_importance.png"), dpi=150)
    plt.close()

    # 3. Deep run score vs Elo scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    champ_prob = results.set_index("Player")["Champion%"]
    sizes  = [champ_prob.get(p, 0.001) * 8000 + 50 for p in feat["nickname"]]
    c_vals = ["#e74c3c" if champ_prob.get(p, 0) > 0.1 else
              "#3498db" if champ_prob.get(p, 0) > 0.03 else "#95a5a6"
              for p in feat["nickname"]]
    ax.scatter(feat["elo"], feat["deep_run_score"], s=sizes, c=c_vals,
               alpha=0.8, edgecolors="white", linewidths=0.8)
    for _, row in feat.iterrows():
        ax.annotate(row["nickname"], (row["elo"], row["deep_run_score"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="Frontrunner (>10%)"),
        mpatches.Patch(color="#3498db", label="Contender (3-10%)"),
        mpatches.Patch(color="#95a5a6", label="Dark horse (<3%)"),
        plt.scatter([], [], s=200, c="gray", alpha=0.5, label="Bubble = Champion%"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    ax.set_xlabel("Current Elo Rating")
    ax.set_ylabel("Deep Run Score (champion=4, finalist=3, top4=2, qf=1)")
    ax.set_title("Elo vs Playoff Pedigree - S10 Pool", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "s10_elo_vs_pedigree.png"), dpi=150)
    plt.close()

    # 4. LDA: PCA projection of training data colored by outcome
    if X_lda is not None and len(X_lda) > 0:
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_clean = scl.fit_transform(imp.fit_transform(X_lda))
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_clean)

        outcome_colors = {4: "#e74c3c", 3: "#f39c12", 2: "#27ae60", 1: "#3498db", 0: "#95a5a6"}
        fig, ax = plt.subplots(figsize=(10, 7))
        for label in sorted(set(y_lda), reverse=True):
            mask = y_lda == label
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=outcome_colors[label], s=70, alpha=0.8,
                       label=OUTCOME_NAME[label], edgecolors="white")
            for idx in np.where(mask)[0]:
                ax.annotate(lda_names[idx], (coords[idx, 0], coords[idx, 1]),
                            textcoords="offset points", xytext=(4, 3), fontsize=7)
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
        ax.set_title("LDA Training Data - PCA Projection\n"
                     "(color = actual playoff outcome)", fontsize=12)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "lda_pca_outcomes.png"), dpi=150)
        plt.close()

    # 5. LDA S10 outcome probabilities
    if lda_df is not None:
        outcome_cols = [c for c in lda_df.columns if c in OUTCOME_NAME.values()]
        if outcome_cols:
            fig, ax = plt.subplots(figsize=(13, 6))
            x = np.arange(len(lda_df))
            width = 0.15
            palette = {"Champion": "#e74c3c", "Finalist": "#f39c12",
                       "Top 4": "#27ae60", "QF Exit": "#3498db", "R1 Exit": "#95a5a6"}
            for i, col in enumerate(outcome_cols):
                color = palette.get(col, "gray")
                ax.bar(x + i * width, lda_df[col] * 100, width, label=col, color=color,
                       edgecolor="white", alpha=0.85)
            ax.set_xticks(x + width * (len(outcome_cols) - 1) / 2)
            ax.set_xticklabels(lda_df["player"], rotation=45, ha="right")
            ax.set_ylabel("Probability (%)")
            ax.set_title("LDA Predicted Outcome Probabilities - S10 Pool", fontsize=12)
            ax.legend(title="Outcome", loc="upper right", framealpha=0.9)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, "s10_lda_outcomes.png"), dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
