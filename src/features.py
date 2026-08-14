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


# Function to load all match records into a flat DataFrame
def load_all_matches():
    frames = []  # List to accumulate match rows across all seasons

    # Loop through each season and load the corresponding JSON file
    for s in SEASONS:
        fpath = os.path.join(RAW, f'season_{s}_matches.json')  # Path to season file
        if not os.path.exists(fpath):  # Skip if file doesn't exist
            continue
        with open(fpath) as f:
            matches = json.load(f)  # Load the list of match objects

        # Extract fields from each match record
        for m in matches:
            players     = m.get('players', [])           # List of two player objects
            result      = m.get('result') or {}          # Match result (winner UUID + time)
            winner_uuid = result.get('uuid')             # UUID of the winning player
            win_time    = result.get('time')             # Finish time in milliseconds
            forfeited   = m.get('forfeited', False)      # Whether the match was forfeited
            if len(players) < 2:  # Skip incomplete match records
                continue
            p1, p2 = players[0], players[1]  # Split into player 1 and player 2
            frames.append({
                'match_id':    m.get('id'),
                'season':      m.get('season', s),
                'date':        m.get('date'),
                'p1_nick':     p1.get('nickname'),    # Player 1 username
                'p1_uuid':     p1.get('uuid'),        # Player 1 unique ID
                'p1_elo':      p1.get('eloRate'),     # Player 1 Elo before match
                'p2_nick':     p2.get('nickname'),    # Player 2 username
                'p2_uuid':     p2.get('uuid'),        # Player 2 unique ID
                'p2_elo':      p2.get('eloRate'),     # Player 2 Elo before match
                'winner_uuid': winner_uuid,
                'win_time_ms': win_time,
                'forfeited':   forfeited,
            })

    df = pd.DataFrame(frames).drop_duplicates('match_id')  # Remove duplicate match IDs
    df['p1_won'] = df['winner_uuid'] == df['p1_uuid']       # True if player 1 won
    return df


# Function to load confirmed playoff bracket results for each season
def load_playoff_results():
    path = os.path.join(RAW, 'all_playoff_results.json')  # Path to results file
    with open(path) as f:
        results = {int(k): v for k, v in json.load(f).items()}  # Parse season keys as ints

    # Override with confirmed Season 9 ground truth
    results[9] = {
        'champion': 'hackingnoises',
        'finalist': 'doogile',
        'top4':     ['Pinne', 'Infume'],
        'qf_exit':  ['steez', 'Aquacorde', 'lowk3y_', 'BlazeMind'],
        'r1_exit':  ['edcr','Feinberg','nhb_','silverrruns','BeefSalad','nahhann','HDMICables','bing_pigs'],
    }
    return results


# Function to load head-to-head win rate records from the stats CSV
def load_h2h_csv():
    stats_path = os.path.join(RAW, 'all_player_stats.csv')  # Path to the stats CSV
    if not os.path.exists(stats_path):  # Return empty DataFrame if file not found
        return pd.DataFrame()

    stats        = pd.read_csv(stats_path)                       # Load the player stats
    uuid_to_nick = dict(zip(stats['uuid'], stats['nickname']))    # Build UUID → nickname lookup

    rows = []  # List to accumulate H2H records
    for s in SEASONS:
        path = os.path.join(RAW, f'season_{s}_h2h.json')  # Path to season H2H file
        if not os.path.exists(path):
            continue
        with open(path) as f:
            records = json.load(f)  # Load the H2H records for this season

        for r in records:
            p1, p2 = r.get('player1'), r.get('player2')   # Get player nicknames
            ranked  = r.get('results_ranked') or {}        # Ranked match results
            total   = ranked.get('total', 0)               # Total matches played
            if total == 0:  # Skip pairs with no ranked matches
                continue

            # Extract per-player win counts from the UUID-keyed dict
            uuid_wins = {k: v for k, v in ranked.items() if k != 'total'}
            p1_uuid   = next((k for k, n in uuid_to_nick.items()
                              if isinstance(n, str) and n.lower() == (p1 or '').lower()), None)
            p1_wins   = uuid_wins.get(p1_uuid, 0) if p1_uuid else 0  # Wins by player 1

            rows.append({
                'season':     s,
                'player1':    p1,
                'player2':    p2,
                'total':      total,
                'p1_wins':    p1_wins,
                'p2_wins':    total - p1_wins,
                'p1_winrate': p1_wins / total,  # Fraction of matches won by player 1
            })
    return pd.DataFrame(rows)


# Function to load per-season LCQ (Last Chance Qualifier) participant sets
def load_lcq_by_season():
    lcq = {}  # Dict mapping season number to set of LCQ players
    for s in SEASONS:
        path = os.path.join(RAW, f'season_{s}_playoffs.json')  # Path to playoff bracket file
        if not os.path.exists(path):
            lcq[s] = set()  # Empty set if file not found
            continue
        try:
            with open(path) as f:
                data = json.load(f)['data']['data']  # Navigate to the bracket data
            # LCQ players have a seed number >= 12 (lower seeds = LCQ qualifiers)
            lcq[s] = {p['nickname'] for p in data.get('players', [])
                      if p.get('seedNumber', 0) >= 12}
        except (KeyError, TypeError, json.JSONDecodeError):
            lcq[s] = set()  # Default to empty on any parsing error
    return lcq


# Function to load each player's seeding vs actual placement delta per season
def load_delta_by_season():
    deltas = {}  # Dict mapping season to player → delta dict
    for s in SEASONS:
        path = os.path.join(RAW, f'season_{s}_playoffs.json')
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)['data']['data']
            idx_to_nick = {p['seedNumber']: p['nickname'] for p in data.get('players', [])}
            season_d = {}
            for r in data.get('results', []):
                idx, place = r.get('player'), r.get('place')  # Seed index and final place
                nick = idx_to_nick.get(idx)
                if nick and place is not None:
                    season_d[nick] = (idx + 1) - place   # Positive = outperformed seed
            deltas[s] = season_d
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return deltas


# Function to build a head-to-head win rate lookup dict from match records
def build_h2h_lookup(df, season_filter):
    sub = df[df['season'] <= season_filter][['p1_nick', 'p2_nick', 'p1_won']].dropna()
    wins  = {}   # Dict mapping (p1, p2) → wins by p1
    total = {}   # Dict mapping (p1, p2) → total matches between p1 and p2

    # Iterate through every match and update both directions of the H2H record
    for row in sub.itertuples(index=False):
        p1, p2, p1_won = row.p1_nick, row.p2_nick, row.p1_won
        total[(p1, p2)] = total.get((p1, p2), 0) + 1                # Increment total for p1 vs p2
        wins[(p1, p2)]  = wins.get((p1, p2),  0) + int(p1_won)      # Increment p1's wins
        total[(p2, p1)] = total.get((p2, p1), 0) + 1                # Symmetric: total for p2 vs p1
        wins[(p2, p1)]  = wins.get((p2, p1),  0) + int(not p1_won)  # Symmetric: p2's wins

    return {k: wins[k] / total[k] for k in total}  # Compute win rates


# Helper: compute overall win rate from a player's match rows
def compute_win_rate(as_p1, as_p2):
    wins  = int(as_p1['p1_won'].sum() + (~as_p2['p1_won']).sum())  # Wins as p1 + wins as p2
    total = len(as_p1) + len(as_p2)                                  # Total matches played
    return wins / total if total > 0 else np.nan, wins, total


# Helper: compute finish time statistics from won non-forfeit matches
def compute_finish_stats(as_p1, as_p2):
    times_p1    = as_p1[as_p1['p1_won'] & ~as_p1['forfeited']]['win_time_ms']   # Win times as p1
    times_p2    = as_p2[~as_p2['p1_won'] & ~as_p2['forfeited']]['win_time_ms']  # Win times as p2
    all_times   = pd.concat([times_p1, times_p2]).dropna()                        # Combine and clean
    avg_time    = float(all_times.mean()) if len(all_times) > 0 else np.nan       # Mean finish time
    best_time   = float(all_times.min())  if len(all_times) > 0 else np.nan       # Fastest win
    std_time    = float(all_times.std())  if len(all_times) > 1 else np.nan       # Variability
    consistency = 1.0 / (std_time / 1000 + 1) if std_time and not np.isnan(std_time) else np.nan
    return avg_time, best_time, consistency


# Helper: compute win rate over the most recent 20 matches
def compute_recent_form(as_p1, as_p2, fallback_wr):
    recent = pd.concat([
        as_p1[['date', 'p1_won']].rename(columns={'p1_won': 'won'}),
        as_p2[['date', 'p1_won']].rename(columns={'p1_won': 'won'}).assign(won=lambda x: ~x['won'])
    ]).sort_values('date').tail(20)                                   # Last 20 matches by date
    return float(recent['won'].mean()) if len(recent) > 0 else fallback_wr


# Helper: get the player's most recently recorded Elo rating
def compute_current_elo(as_p1, as_p2, default=1500.0):
    elo_p1 = as_p1.sort_values('date').tail(1)['p1_elo']   # Most recent Elo as p1
    elo_p2 = as_p2.sort_values('date').tail(1)['p2_elo']   # Most recent Elo as p2
    if len(elo_p1) > 0:
        return float(elo_p1.values[-1])   # Use p1 Elo if available
    if len(elo_p2) > 0:
        return float(elo_p2.values[-1])   # Fall back to p2 Elo
    return default                         # Default to 1500 if no data


# Helper: compute the slope of the player's recent Elo changes (momentum)
def compute_elo_momentum(as_p1, as_p2):
    elo_ts = pd.concat([
        as_p1[['date', 'p1_elo']].rename(columns={'p1_elo': 'elo'}),
        as_p2[['date', 'p2_elo']].rename(columns={'p2_elo': 'elo'}),
    ]).sort_values('date')['elo'].dropna()
    recent = elo_ts.tail(20)   # Look at the last 20 recorded Elo values
    if len(recent) > 1:
        return float((recent.iloc[-1] - recent.iloc[0]) / len(recent))  # Average Elo change per match
    return 0.0  # No momentum if fewer than 2 data points


# Helper: compute tournament pedigree score and counts from historical results
def compute_pedigree(nick, ped):
    champ_count = sum(1 for v in ped.values() if v.get('champion') == nick)       # Times won
    fin_count   = sum(1 for v in ped.values() if v.get('finalist') == nick)       # Times finalist
    top4_count  = sum(1 for v in ped.values() if nick in v.get('top4', []))       # Times top 4
    qf_count    = sum(1 for v in ped.values() if nick in v.get('qf_exit', []))    # Times QF exit
    deep_run    = champ_count * 4 + fin_count * 3 + top4_count * 2 + qf_count    # Weighted score
    return deep_run, champ_count, fin_count


# Function to build a 13-feature player snapshot for a given season (leakage-safe)
def build_player_features(df, players, playoff_results, season_filter,
                           pedigree_cutoff=None, lcq_by_season=None,
                           delta_by_season=None, lcq_season=None):
    sub = df[df['season'] <= season_filter]   # Only use data up through the given season
    if pedigree_cutoff is None:
        pedigree_cutoff = season_filter - 1   # Default: exclude results from the target season

    # Filter playoff results to only seasons before the pedigree cutoff (no future leakage)
    ped     = {k: v for k, v in playoff_results.items() if k <= pedigree_cutoff}
    records = []   # List to accumulate one feature row per player

    for nick in players:
        as_p1 = sub[sub['p1_nick'] == nick]   # All matches this player played as p1
        as_p2 = sub[sub['p2_nick'] == nick]   # All matches this player played as p2

        # Compute each feature group using the helper functions defined above
        win_rate, _, _               = compute_win_rate(as_p1, as_p2)
        avg_time, best_time, consist = compute_finish_stats(as_p1, as_p2)
        recent_wr                    = compute_recent_form(as_p1, as_p2, win_rate)
        last_elo                     = compute_current_elo(as_p1, as_p2)
        momentum                     = compute_elo_momentum(as_p1, as_p2)
        deep_run, champ_count, fin_count = compute_pedigree(nick, ped)

        # Forfeit rate: fraction of all matches (including both sides) that were forfeited
        total_rows   = len(as_p1) + len(as_p2)
        forfeit_rate = float((as_p1['forfeited'].sum() + as_p2['forfeited'].sum()) / total_rows) if total_rows > 0 else 0.0

        # LCQ flag: did this player qualify via last-chance bracket for the target season?
        _lcq_s   = lcq_season if lcq_season is not None else season_filter
        lcq_flag = int(nick in lcq_by_season.get(_lcq_s, set())) if lcq_by_season else 0

        # Average tournament outperformance across all past seasons
        avg_delta = np.nan
        if delta_by_season:
            past = [delta_by_season[s][nick] for s in range(1, pedigree_cutoff + 1)
                    if s in delta_by_season and nick in delta_by_season[s]]
            if past:
                avg_delta = float(np.mean(past))  # Mean of seed-adjusted placement deltas

        records.append({
            'nickname':            nick,
            'elo':                 last_elo,
            'win_rate':            win_rate,
            'recent_wr':           recent_wr,
            'consistency':         consist,
            'avg_time_ms':         avg_time,
            'best_time_ms':        best_time,
            'forfeit_rate':        forfeit_rate,
            'elo_momentum':        momentum,
            'champion_count':      champ_count,
            'finalist_count':      fin_count,
            'deep_run_score':      deep_run,
            'lcq_flag':            lcq_flag,
            'avg_tournament_delta': avg_delta,
        })

    return pd.DataFrame(records)
