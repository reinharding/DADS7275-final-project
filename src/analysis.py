"""
MCSR Ranked - Exploratory Analysis
Loads raw scraped data, builds a flat match-level dataset,
computes per-player season features, and prints key insights.
Models: K-Means clustering, t-SNE visualization, PCA.
"""

import subprocess
import sys
import importlib

if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

for pkg in ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        importlib.invalidate_caches()

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

SEASONS = list(range(1, 10))

# Season 10 playoff pool (for highlighting in charts)
S10_POOL = {
    "Infume", "edcr", "doogile", "Feinberg", "7rowl", "bing_pigs",
    "nahhann", "BlazeMind", "Aquacorde", "silverrruns", "BeefSalad",
    "meebie", "hackingnoises", "steez", "nhb_", "Ancoboyy",
}

# Load playoff results scraped by scraper.py (used for champion highlighting in charts)
_playoff_path = os.path.join(RAW_DIR, "all_playoff_results.json")
if os.path.exists(_playoff_path):
    with open(_playoff_path) as _f:
        PLAYOFF_RESULTS = {int(k): v for k, v in json.load(_f).items()}
else:
    PLAYOFF_RESULTS = {}

PLAYOFF_PLAYERS = {
    1:  ["silverrruns","dandannyboy","Oxidiot","Reignex","priffie","orachi_","lowk3y_","7rowl","doogile","Ancoboyy","pulsar32","Ranik_","MoleyG","CroProYT","AutomattPL","Dylqn"],
    2:  ["lowk3y_","CroProYT","dandannyboy","doogile","7rowl","kW1st","priffie","dwoh","silverrruns","Emillk","bing_pigs","drx6","Ancoboyy","Ranik_","Oxidiot","AutomattPL"],
    3:  ["7rowl","Ancoboyy","dandannyboy","doogile","hackingnoises","lowk3y_","Oxidiot","priffie","ANJOUU","AutomattPL","BeefSalad","Bloonskiller","loodlow","paplerr","v_strid","autoqualler"],
    4:  ["7rowl","Ancoboyy","dandannyboy","doogile","AutomattPLUS","hackingnoises","Hinart","lowk3y_","Oxidot","paplerr","priffie","silverrruns","ANJOUU","bing_pigs","Cube1337x","v_strid"],
    5:  ["7rowl","Ancoboyy","BeefSalad","bing_pigs","doogile","AutomattPLUS","hackingnoises","lowk3y_","Oxidiot","silverrruns","TUDORULE","v_strid","Aquacorde","dandannyboy","KenanKardes","pulsar32"],
    6:  ["7rowl","Ayreliaa","BeefSalad","bing_pigs","doogile","AutomattPLUS","Feinberg","hackingnoises","lowk3y_","MrBudgiee","Oxidiot","silverrruns","dandannyboy","Erikfzf","ogurikappa","TUDORULE"],
    7:  ["7rowl","Ancoboyy","Aquacorde","BadGamer","BeefSalad","bing_pigs","doogile","Feinberg","Infume","lowk3y_","priffie","retropog","r7sD4fH6jK0wY5uB","hackingnoises","Oxidiot","silverrruns"],
    8:  ["7rowl","Aquacorde","BeefSalad","bing_pigs","DARVY__X1","doogile","edcr","Feinberg","Infume","lowk3y_","Ranik_","silverrruns","hackingnoises","KenanKardes","TUDORULE","v_strid"],
    9:  ["Feinberg","Infume","edcr","steez","hackingnoises","Aquacorde","nhb_","silverrruns","Pinne","BeefSalad","nahhann","lowk3y_","doogile","HDMICables","bing_pigs","BlazeMind"],
}

# Cluster archetype labels (assigned after fitting, ordered by win rate desc)
CLUSTER_NAMES = ["Elite Consistent", "Tournament Veteran", "Rising Contender", "Early Exits"]


# =============================================================================
# 1. LOAD & FLATTEN MATCH DATA
# =============================================================================

def load_matches() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        path = os.path.join(RAW_DIR, f"season_{s}_matches.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            matches = json.load(f)
        playoff_set = {p.lower() for p in PLAYOFF_PLAYERS.get(s, [])}
        for m in matches:
            players = m.get("players", [])
            if len(players) < 2:
                continue
            p1, p2 = players[0], players[1]
            result  = m.get("result") or {}
            changes = m.get("changes") or []
            winner_uuid = result.get("uuid")
            finish_time = result.get("time")
            forfeited   = m.get("forfeited", False)
            change_map  = {c["uuid"]: c.get("change", 0) for c in changes}
            for player, opponent in [(p1, p2), (p2, p1)]:
                uuid = player.get("uuid")
                won  = (uuid == winner_uuid) if winner_uuid else None
                rows.append({
                    "match_id":          m.get("id"),
                    "season":            m.get("season") or s,
                    "date":              m.get("date"),
                    "nickname":          player.get("nickname"),
                    "uuid":              uuid,
                    "opponent":          opponent.get("nickname"),
                    "elo_before":        player.get("eloRate"),
                    "elo_change":        change_map.get(uuid, 0),
                    "won":               won,
                    "finish_time_ms":    finish_time if won else None,
                    "forfeited":         forfeited,
                    "is_playoff_player": player.get("nickname", "").lower() in playoff_set,
                    "seed_type":         m.get("seedType"),
                    "bastion_type":      m.get("bastionType"),
                })
    return pd.DataFrame(rows)


# =============================================================================
# 2. PLAYER-SEASON FEATURES
# =============================================================================

def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    playoff_rows = []
    for season, players in PLAYOFF_PLAYERS.items():
        sdf = df[(df["season"] == season) & (df["is_playoff_player"])]
        for player in players:
            pdf = sdf[sdf["nickname"].str.lower() == player.lower()]
            if pdf.empty:
                continue
            wins  = pdf["won"].sum()
            total = pdf["won"].notna().sum()
            times = pdf[pdf["won"] == True]["finish_time_ms"].dropna()
            playoff_rows.append({
                "nickname":          player,
                "season":            season,
                "matches_played":    total,
                "wins":              wins,
                "losses":            total - wins,
                "winrate":           wins / total if total else np.nan,
                "avg_finish_ms":     times.mean()  if len(times) else np.nan,
                "best_finish_ms":    times.min()   if len(times) else np.nan,
                "consistency_score": 1 - (times.std() / times.mean()) if len(times) > 1 else np.nan,
                "forfeit_rate":      pdf["forfeited"].sum() / total if total else np.nan,
                "elo_at_season_end": pdf["elo_before"].iloc[-1] if not pdf.empty else np.nan,
            })
    return pd.DataFrame(playoff_rows)


# Career aggregates (one row per player across all seasons)
def build_career_features(features: pd.DataFrame) -> pd.DataFrame:
    agg = features.groupby("nickname").agg(
        seasons_played=("season", "nunique"),
        avg_winrate=("winrate", "mean"),
        avg_finish_ms=("avg_finish_ms", "mean"),
        best_finish_ms=("best_finish_ms", "min"),
        avg_consistency=("consistency_score", "mean"),
        avg_elo=("elo_at_season_end", "mean"),
        avg_matches=("matches_played", "mean"),
    ).reset_index()
    return agg


# =============================================================================
# 3. HEAD-TO-HEAD TABLE
# =============================================================================

def load_h2h() -> pd.DataFrame:
    stats_path = os.path.join(RAW_DIR, "all_player_stats.csv")
    if not os.path.exists(stats_path):
        return pd.DataFrame()
    stats = pd.read_csv(stats_path)
    uuid_to_nick = dict(zip(stats["uuid"], stats["nickname"]))

    rows = []
    for s in SEASONS:
        path = os.path.join(RAW_DIR, f"season_{s}_h2h.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            p1 = r.get("player1")
            p2 = r.get("player2")
            ranked = r.get("results_ranked") or {}
            total  = ranked.get("total", 0)
            if total == 0:
                continue
            uuid_wins = {k: v for k, v in ranked.items() if k != "total"}
            p1_uuid = next((k for k, nick in uuid_to_nick.items()
                            if isinstance(nick, str) and nick.lower() == (p1 or "").lower()), None)
            p1_wins = uuid_wins.get(p1_uuid, 0) if p1_uuid else 0
            rows.append({
                "season":     s,
                "player1":    p1,
                "player2":    p2,
                "total":      total,
                "p1_wins":    p1_wins,
                "p2_wins":    total - p1_wins,
                "p1_winrate": p1_wins / total if total else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 4. PLOTS (with legends + S10 highlighting)
# =============================================================================

def _s10_colors(names, s10_color="#e74c3c", other_color="#3498db"):
    return [s10_color if n in S10_POOL else other_color for n in names]


def plot_winrates(features: pd.DataFrame):
    avg = (features.groupby("nickname")["winrate"]
                   .mean()
                   .sort_values(ascending=False)
                   .head(20))
    colors = _s10_colors(avg.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(avg.index, avg.values * 100, color=colors, edgecolor="white")
    ax.set_title("Average Win Rate - Top 20 Playoff Players (all seasons)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="S10 Playoff Pool"),
        mpatches.Patch(color="#3498db", label="Historical only"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "winrates.png"), dpi=150)
    plt.close()
    print("Saved winrates.png")


def plot_elo_vs_winrate(features: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        features["elo_at_season_end"],
        features["winrate"],
        c=features["season"],
        s=features["matches_played"] * 2,
        alpha=0.6,
        cmap="plasma",
    )
    plt.colorbar(sc, ax=ax, label="Season")
    ax.set_xlabel("Elo at Season End")
    ax.set_ylabel("Win Rate")
    ax.set_title("Elo vs Win Rate (color = season, size = matches played)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Size legend
    for size_val, label in [(10, "5 matches"), (30, "15 matches"), (60, "30 matches")]:
        ax.scatter([], [], s=size_val, c="gray", alpha=0.5, label=label)
    ax.legend(title="Matches Played", loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "elo_vs_winrate.png"), dpi=150)
    plt.close()
    print("Saved elo_vs_winrate.png")


def plot_finish_times(features: pd.DataFrame):
    clean = features.dropna(subset=["avg_finish_ms"]).copy()
    clean["avg_finish_s"] = clean["avg_finish_ms"] / 1000

    # Color by S10 status
    palette = {s: "#e74c3c" if any(n in S10_POOL for n in
               features[features["season"] == s]["nickname"]) else "#3498db"
               for s in clean["season"].unique()}

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=clean, x="season", y="avg_finish_s", ax=ax, palette="Set2")
    ax.set_title("Avg Finish Time Distribution per Season (playoff players)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Avg Finish Time (s)")

    # Add legend note
    ax.text(0.01, 0.97, "Each box = all playoff players that season | whiskers = 1.5x IQR",
            transform=ax.transAxes, fontsize=8, va="top", color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "finish_times_by_season.png"), dpi=150)
    plt.close()
    print("Saved finish_times_by_season.png")


def plot_appearances(features: pd.DataFrame):
    counts = (features.groupby("nickname")["season"]
                      .nunique()
                      .sort_values(ascending=False)
                      .head(20))
    colors = _s10_colors(counts.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_title("Playoff Appearances - Top 20 Players")
    ax.set_ylabel("Number of Seasons")
    ax.set_xlabel("")

    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="S10 Playoff Pool"),
        mpatches.Patch(color="#3498db", label="Historical only"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "playoff_appearances.png"), dpi=150)
    plt.close()
    print("Saved playoff_appearances.png")


def plot_h2h_heatmap(h2h: pd.DataFrame, season: int = 9):
    sdf = h2h[h2h["season"] == season]
    if sdf.empty:
        print(f"No H2H data for season {season}, skipping heatmap.")
        return
    players = sorted(set(sdf["player1"].tolist() + sdf["player2"].tolist()))
    mat = pd.DataFrame(np.nan, index=players, columns=players)
    for _, row in sdf.iterrows():
        mat.loc[row["player1"], row["player2"]] = row["p1_winrate"]
        mat.loc[row["player2"], row["player1"]] = 1 - row["p1_winrate"]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(mat, annot=True, fmt=".0%", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title(f"Season {season} - Head-to-Head Win Rates\n"
                 "(green = row player wins more, red = row player loses more)")
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, f"h2h_heatmap_s{season}.png"), dpi=150)
    plt.close()
    print(f"Saved h2h_heatmap_s{season}.png")


# =============================================================================
# 5. K-MEANS CLUSTERING (player archetypes)
# =============================================================================

CLUSTER_FEATURE_COLS = ["avg_winrate", "avg_finish_ms", "avg_consistency",
                        "avg_elo", "seasons_played", "avg_matches"]

ARCHETYPE_LABELS = {
    # Assigned dynamically by cluster centroid win rate
    0: "Archetype A",
    1: "Archetype B",
    2: "Archetype C",
    3: "Archetype D",
}

ARCHETYPE_COLORS = ["#e74c3c", "#f39c12", "#27ae60", "#3498db"]


def _prep_cluster_matrix(career: pd.DataFrame):
    X_raw = career[CLUSTER_FEATURE_COLS].values.astype(float)
    imp   = SimpleImputer(strategy="median")
    scl   = StandardScaler()
    X     = scl.fit_transform(imp.fit_transform(X_raw))
    return X, imp, scl


def find_optimal_k(career: pd.DataFrame, k_range=range(2, 7)) -> int:
    X, _, _ = _prep_cluster_matrix(career)
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    print("\n-- K-Means Silhouette Scores --")
    for k, s in scores.items():
        bar = "#" * int(s * 30)
        print(f"  k={k}: {s:.4f}  {bar}")
    best_k = max(scores, key=scores.get)
    print(f"  Best k = {best_k} (silhouette = {scores[best_k]:.4f})")
    return best_k


def fit_kmeans(career: pd.DataFrame, k: int):
    X, imp, scl = _prep_cluster_matrix(career)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    career = career.copy()
    career["cluster"] = labels

    # Sort clusters so label 0 = most elite.
    # Composite prestige: high Elo (50%) + high win rate (30%) + fast finish (20%).
    g = career.groupby("cluster").agg(
        avg_elo     =("avg_elo",      "mean"),
        avg_winrate =("avg_winrate",  "mean"),
        avg_finish  =("avg_finish_ms","mean"),
    )
    def _norm(s): rng = s.max() - s.min(); return (s - s.min()) / rng if rng > 0 else s * 0
    prestige = (
        _norm(g["avg_elo"])     * 0.5 +
        _norm(g["avg_winrate"]) * 0.3 +
        _norm(-g["avg_finish"]) * 0.2
    ).sort_values(ascending=False)
    rank_map = {old: new for new, old in enumerate(prestige.index)}
    career["cluster"] = career["cluster"].map(rank_map)

    sil = silhouette_score(X, labels)

    # Embed PCA coords into career so linked plots share the same projection
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    career["pc1"] = coords[:, 0]
    career["pc2"] = coords[:, 1]
    career.attrs["pca_var"] = pca.explained_variance_ratio_   # store for annotation

    return career, km, sil


def compute_tsne(career: pd.DataFrame) -> pd.DataFrame:
    """Add t-SNE coordinates to career DataFrame (uses same feature matrix as clustering)."""
    X, _, _ = _prep_cluster_matrix(career)
    perplexity = min(30, len(career) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X)
    career = career.copy()
    career["t1"] = coords[:, 0]
    career["t2"] = coords[:, 1]
    return career


def label_archetypes(k: int) -> dict:
    names = [
        "Elite / Champion-tier",
        "Consistent Veteran",
        "Rising Contender",
        "Early Exit / Fringe",
        "Occasional Qualifier",
        "One-Season Wonder",
    ]
    return {i: names[i] if i < len(names) else f"Cluster {i+1}" for i in range(k)}


def _annotate_selective(ax, career: pd.DataFrame, x_col: str, y_col: str,
                         priority_set: set):
    """
    Label every player, but use two tiers:
      - priority players (S10 pool + known champions): bold, black, 8pt
      - everyone else: light grey, 6pt, no box
    This reduces visual clutter while keeping the chart fully readable.
    """
    for _, row in career.iterrows():
        nick = row["nickname"]
        x, y = row[x_col], row[y_col]
        if nick in priority_set:
            ax.annotate(nick, (x, y), textcoords="offset points", xytext=(5, 3),
                        fontsize=8, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))
        else:
            ax.annotate(nick, (x, y), textcoords="offset points", xytext=(4, 2),
                        fontsize=6, color="#888888", alpha=0.75)


def plot_kmeans_clusters(career: pd.DataFrame, k: int, sil_score: float):
    """PCA scatter — one dot per player (career average). Labels prioritise S10 pool."""
    archetype_map = label_archetypes(k)
    colors_k      = [plt.cm.tab10(i) for i in range(k)]
    var           = career.attrs.get("pca_var", [0, 0])
    total_var     = sum(var)

    # Players to highlight prominently
    known_champs = {r for res in PLAYOFF_RESULTS.values()
                    for r in ([res.get("champion")] if res.get("champion") else [])}
    priority = S10_POOL | known_champs

    fig, ax = plt.subplots(figsize=(12, 7))
    for cl in sorted(career["cluster"].unique()):
        mask = career["cluster"] == cl
        ax.scatter(career.loc[mask, "pc1"], career.loc[mask, "pc2"],
                   c=colors_k[cl], s=55, alpha=0.80,
                   label=f"Cluster {cl}: {archetype_map[cl]}", edgecolors="white", lw=0.4)

    _annotate_selective(ax, career, "pc1", "pc2", priority)

    # S10 pool: star marker on top of existing dots
    s10_mask = career["nickname"].isin(S10_POOL)
    ax.scatter(career.loc[s10_mask, "pc1"], career.loc[s10_mask, "pc2"],
               s=180, facecolors="none", edgecolors="#c0392b", linewidths=1.5,
               label="S10 Pool (circled)", zorder=6)

    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)")
    ax.set_title(
        f"Player Archetypes — K-Means (k={k}, silhouette={sil_score:.3f})\n"
        f"Each dot = one player (career average)  |  "
        f"PC1+PC2 explain {total_var:.0%} of variance  |  Red circle = S10 pool"
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "kmeans_clusters.png"), dpi=150)
    plt.close()
    print("Saved kmeans_clusters.png")

    # Print cluster summaries
    print(f"\n-- Cluster Archetypes (k={k}, silhouette={sil_score:.4f}) --")
    for cl in sorted(career["cluster"].unique()):
        members = career[career["cluster"] == cl]
        print(f"\n  [{cl}] {archetype_map[cl]} ({len(members)} players)")
        print(f"       Avg win rate : {members['avg_winrate'].mean():.1%}")
        print(f"       Avg Elo      : {members['avg_elo'].mean():.0f}")
        if not members["avg_finish_ms"].isna().all():
            print(f"       Avg finish   : {members['avg_finish_ms'].mean()/1000:.1f}s")
        print(f"       Seasons      : {members['seasons_played'].mean():.1f}")
        print(f"       Players      : {', '.join(sorted(members['nickname'].tolist()))}")


# =============================================================================
# 6. t-SNE VISUALIZATION
# =============================================================================

def plot_tsne(career: pd.DataFrame, k: int):
    """t-SNE scatter. Stats table sits below the plot, not over the data."""
    archetype_map = label_archetypes(k)
    colors_k      = [plt.cm.tab10(i) for i in range(k)]

    known_champs = {r for res in PLAYOFF_RESULTS.values()
                    for r in ([res.get("champion")] if res.get("champion") else [])}
    priority = S10_POOL | known_champs

    # Two-row layout: scatter on top, table panel below
    table_rows = k + 1   # header + one row per cluster
    fig_h      = 7 + 0.28 * table_rows
    fig = plt.figure(figsize=(12, fig_h))
    gs  = fig.add_gridspec(2, 1, height_ratios=[7, 0.28 * table_rows], hspace=0.08)
    ax       = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    for cl in sorted(career["cluster"].unique()):
        mask = career["cluster"] == cl
        ax.scatter(career.loc[mask, "t1"], career.loc[mask, "t2"],
                   c=colors_k[cl], s=55, alpha=0.80,
                   label=f"Cluster {cl}: {archetype_map[cl]}", edgecolors="white", lw=0.4)

    _annotate_selective(ax, career, "t1", "t2", priority)

    # S10 pool: circle outline (back to original style)
    s10_mask = career["nickname"].isin(S10_POOL)
    ax.scatter(career.loc[s10_mask, "t1"], career.loc[s10_mask, "t2"],
               s=180, facecolors="none", edgecolors="#c0392b", linewidths=1.5,
               label="S10 Pool (circled)", zorder=6)

    # Stats table placed in the dedicated axes below the scatter
    col_labels = ["Cluster", "n", "Avg Elo", "Win%", "Finish (s)"]
    table_data = []
    for cl in sorted(career["cluster"].unique()):
        m  = career[career["cluster"] == cl]
        elo = f"{m['avg_elo'].mean():.0f}"
        wr  = f"{m['avg_winrate'].mean():.0%}"
        ft  = f"{m['avg_finish_ms'].mean()/1000:.1f}" if not m["avg_finish_ms"].isna().all() else "N/A"
        table_data.append([f"C{cl}: {archetype_map[cl]}", str(len(m)), elo, wr, ft])

    tbl = ax_table.table(cellText=table_data, colLabels=col_labels,
                         loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f4f4")

    ax.set_xlabel("t-SNE Dim 1  (distances within plot not to scale)")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title("t-SNE Player Groupings — Color = K-Means Cluster\n"
                 "Red circle = S10 playoff pool  |  Table below shows why each cluster is named as it is")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.savefig(os.path.join(PROC_DIR, "tsne_players.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved tsne_players.png")


# =============================================================================
# 6b. LINKED PCA + t-SNE (side-by-side, same colors)
# =============================================================================

def plot_linked_pca_tsne(career: pd.DataFrame, k: int, sil_score: float):
    """
    Side-by-side PCA (left) and t-SNE (right) using the same cluster colors.
    A player in the Elite cluster appears the same color in both plots,
    making it easy to ask: 'Where does the Elite cluster land in each space?'
    """
    archetype_map = label_archetypes(k)
    colors_k      = [plt.cm.tab10(i) for i in range(k)]
    var           = career.attrs.get("pca_var", [0, 0])
    total_var     = sum(var)

    known_champs = {r for res in PLAYOFF_RESULTS.values()
                    for r in ([res.get("champion")] if res.get("champion") else [])}
    priority  = S10_POOL | known_champs
    s10_mask  = career["nickname"].isin(S10_POOL)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        f"Player Space: PCA (left) vs t-SNE (right)  —  K-Means k={k}\n"
        "Same color = same cluster in both plots. "
        "PCA preserves global distances; t-SNE preserves local neighborhoods.",
        fontsize=11
    )

    for ax, (xc, yc), xlabel, ylabel, subtitle in [
        (axes[0], ("pc1","pc2"),
         f"PC1 ({var[0]:.1%} var)", f"PC2 ({var[1]:.1%} var)",
         f"PCA  —  PC1+PC2 explain {total_var:.0%} of variance"),
        (axes[1], ("t1","t2"),
         "t-SNE Dim 1 (not to scale)", "t-SNE Dim 2 (not to scale)",
         "t-SNE  —  distances within plot not directly comparable to PCA"),
    ]:
        for cl in sorted(career["cluster"].unique()):
            mask = career["cluster"] == cl
            ax.scatter(career.loc[mask, xc], career.loc[mask, yc],
                       c=colors_k[cl], s=50, alpha=0.78,
                       label=f"C{cl}: {archetype_map[cl]}", edgecolors="white", lw=0.3)
        _annotate_selective(ax, career, xc, yc, priority)
        ax.scatter(career.loc[s10_mask, xc], career.loc[s10_mask, yc],
                   marker="*", s=200, color="none", edgecolors="#c0392b", linewidths=1.3,
                   label="S10 Pool", zorder=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "linked_pca_tsne.png"), dpi=150)
    plt.close()
    print("Saved linked_pca_tsne.png")


# =============================================================================
# 6c. PER-SEASON TRAJECTORY (shows player progression across seasons)
# =============================================================================

def plot_season_trajectories(features: pd.DataFrame, career: pd.DataFrame, k: int):
    """
    Projects each player-season through the career PCA space and draws
    connecting lines season-by-season to show progression trajectories.
    Only S10 pool players + known champions are labeled to avoid clutter.
    """
    if "pc1" not in career.columns:
        return

    archetype_map = label_archetypes(k)
    colors_k      = [plt.cm.tab10(i) for i in range(k)]
    known_champs  = {r for res in PLAYOFF_RESULTS.values()
                     for r in ([res.get("champion")] if res.get("champion") else [])}
    priority      = S10_POOL | known_champs

    # Re-fit scaler on career matrix to project season rows through same space
    X_career, imp, scl = _prep_cluster_matrix(career)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_career)

    feat_cols_map = {   # season-level column → career-level cluster feature
        "winrate":            "avg_winrate",
        "avg_finish_ms":      "avg_finish_ms",
        "consistency_score":  "avg_consistency",
        "elo_at_season_end":  "avg_elo",
        "matches_played":     "avg_matches",
    }
    season_feat_cols = list(feat_cols_map.keys())

    # Project each season row
    sub = features[["nickname","season"] + season_feat_cols].copy()
    sub = sub.rename(columns=feat_cols_map)
    # seasons_played is always 1 per row; needed to match the 6-feature imputer
    sub["seasons_played"] = 1.0

    # Use CLUSTER_FEATURE_COLS order so columns match what imp/scl were fit on
    X_s = sub[CLUSTER_FEATURE_COLS].values.astype(float)
    X_s = scl.transform(imp.transform(X_s))
    coords = pca.transform(X_s)
    sub["pc1"] = coords[:, 0]
    sub["pc2"] = coords[:, 1]
    sub = sub.merge(career[["nickname","cluster"]], on="nickname", how="left")

    var = career.attrs.get("pca_var", [0, 0])

    fig, ax = plt.subplots(figsize=(13, 8))
    for nick, grp in sub.groupby("nickname"):
        grp = grp.sort_values("season")
        cl  = int(grp["cluster"].iloc[0]) if not grp["cluster"].isna().any() else 0
        color = colors_k[cl % len(colors_k)]
        alpha = 0.85 if nick in priority else 0.30
        lw    = 1.4  if nick in priority else 0.6

        ax.plot(grp["pc1"], grp["pc2"], "-o", color=color,
                alpha=alpha, linewidth=lw, markersize=4, zorder=3 if nick in priority else 1)
        if nick in priority:
            # Label the latest season point
            last = grp.iloc[-1]
            ax.annotate(nick, (last["pc1"], last["pc2"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

    # Legend: one entry per cluster
    for cl in sorted(sub["cluster"].dropna().unique()):
        cl = int(cl)
        ax.plot([], [], "o-", color=colors_k[cl % len(colors_k)],
                label=f"C{cl}: {archetype_map[cl]}", linewidth=1.2, markersize=4)
    ax.plot([], [], "o-", color="grey", alpha=0.3, label="Background players", linewidth=0.6)

    ax.set_xlabel(f"PC1  ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.1%} variance)")
    ax.set_title(
        "Season-by-Season Trajectories in PCA Space\n"
        "Each dot = one season; lines connect a player's seasons in order  |  "
        f"PC1+PC2 = {sum(var):.0%} variance"
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(PROC_DIR, "pca_trajectories.png"), dpi=150)
    plt.close()
    print("Saved pca_trajectories.png")


# =============================================================================
# 7. INSIGHTS
# =============================================================================

def print_insights(features: pd.DataFrame, h2h: pd.DataFrame, matches: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    print(f"\nTotal playoff appearances tracked : {len(features)}")
    print(f"Unique players                    : {features['nickname'].nunique()}")
    print(f"Total ranked matches in dataset   : {matches['match_id'].nunique()}")

    print("\n--- Top 5 by avg win rate (min 2 seasons) ---")
    multi  = features.groupby("nickname").filter(lambda x: len(x) >= 2)
    top_wr = multi.groupby("nickname")["winrate"].mean().sort_values(ascending=False).head(5)
    for nick, wr in top_wr.items():
        print(f"  {nick:<20} {wr:.1%}")

    print("\n--- Top 5 fastest avg finish time ---")
    top_speed = (features.dropna(subset=["avg_finish_ms"])
                         .groupby("nickname")["avg_finish_ms"]
                         .mean()
                         .sort_values()
                         .head(5))
    for nick, ms in top_speed.items():
        print(f"  {nick:<20} {ms/1000:.1f}s")

    print("\n--- Most consistent players (highest consistency score) ---")
    top_cons = (features.dropna(subset=["consistency_score"])
                        .groupby("nickname")["consistency_score"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(5))
    for nick, sc in top_cons.items():
        print(f"  {nick:<20} {sc:.3f}")

    print("\n--- Elo correlation with win rate ---")
    valid = features.dropna(subset=["elo_at_season_end", "winrate"])
    corr  = valid["elo_at_season_end"].corr(valid["winrate"])
    print(f"  Pearson r = {corr:.3f}  (1 = perfect, 0 = no relationship)")

    print("\n--- Season 9 H2H dominance ---")
    s9 = h2h[h2h["season"] == 9]
    if not s9.empty:
        dom = (s9.groupby("player1")["p1_winrate"].mean()
                 .sort_values(ascending=False)
                 .head(5))
        for nick, wr in dom.items():
            print(f"  {nick:<20} avg H2H win rate: {wr:.1%}")

    print("\n--- Players with most playoff appearances ---")
    apps = features.groupby("nickname")["season"].nunique().sort_values(ascending=False).head(5)
    for nick, n in apps.items():
        print(f"  {nick:<20} {n} seasons")

    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Loading match data...")
    matches = load_matches()
    print(f"  {len(matches)} player-match rows ({matches['match_id'].nunique()} unique matches)")

    print("Building player-season features...")
    features = build_player_features(matches)
    print(f"  {len(features)} player-season records")

    print("Building career aggregates...")
    career = build_career_features(features)
    print(f"  {len(career)} unique players")

    print("Loading H2H data...")
    h2h = load_h2h()
    print(f"  {len(h2h)} H2H pairs loaded")

    # Save processed datasets
    matches.to_csv(os.path.join(PROC_DIR, "matches_flat.csv"), index=False)
    features.to_csv(os.path.join(PROC_DIR, "player_features.csv"), index=False)
    career.to_csv(os.path.join(PROC_DIR, "career_features.csv"), index=False)
    h2h.to_csv(os.path.join(PROC_DIR, "h2h_flat.csv"), index=False)
    print("Saved processed CSVs to data/processed/")

    print("\nGenerating standard plots...")
    plot_winrates(features)
    plot_elo_vs_winrate(features)
    plot_finish_times(features)
    plot_appearances(features)
    plot_h2h_heatmap(h2h, season=9)

    print("\nRunning K-Means clustering...")
    best_k   = find_optimal_k(career)
    career, km_model, sil = fit_kmeans(career, best_k)
    career.to_csv(os.path.join(PROC_DIR, "career_clustered.csv"), index=False)
    plot_kmeans_clusters(career, best_k, sil)

    print("\nComputing t-SNE coordinates...")
    career = compute_tsne(career)

    print("Generating t-SNE visualization...")
    plot_tsne(career, best_k)

    print("Generating linked PCA + t-SNE plot...")
    plot_linked_pca_tsne(career, best_k, sil)

    print("Generating per-season trajectory plot...")
    plot_season_trajectories(features, career, best_k)

    print_insights(features, h2h, matches)
