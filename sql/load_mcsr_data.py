import ast
import json
import os
import random
import sys
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta

random.seed(42)

# =====================
# DB CONNECTION
# Set MYSQL_PASSWORD in your environment before running.
# =====================
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
if not MYSQL_PASSWORD:
    sys.exit("Set the MYSQL_PASSWORD environment variable before running this script.")

conn = mysql.connector.connect(
    host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
    port=int(os.environ.get("MYSQL_PORT", "3306")),
    user=os.environ.get("MYSQL_USER", "root"),
    password=MYSQL_PASSWORD,
    database=os.environ.get("MYSQL_DATABASE", "mcsr_ranked_playoffs"),
)
cursor = conn.cursor()

# =====================
# FILE PATHS
# Resolve raw data relative to this file: ../data/raw  (sibling ML project's data)
# =====================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

players_df = pd.read_csv(f"{DATA_DIR}/all_player_stats.csv")
matches_df = pd.read_csv(f"{DATA_DIR}/all_matches.csv")
with open(f"{DATA_DIR}/all_playoff_results.json") as f:
    playoffs = json.load(f)

# =====================
# HELPERS
# =====================
def fmt_uuid(raw):
    """Normalize a 32-char hex UUID to the standard 36-char dashed format."""
    if pd.isna(raw):
        return None
    raw = str(raw).strip().replace("-", "")
    if len(raw) != 32:
        return None
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

# =====================
# 1. INSERT Players
#    Deduplicate by uuid — keep the latest season row per player.
# =====================
print("Inserting players...")

# Sort so latest season is last, then drop_duplicates keeps last occurrence
players_df = players_df.sort_values("season")
latest_players = players_df.drop_duplicates(subset="uuid", keep="last").copy()
latest_players["uuid_fmt"] = latest_players["uuid"].apply(fmt_uuid)
latest_players = latest_players[latest_players["uuid_fmt"].notna()]

# Build uuid lookup keyed by formatted uuid
uuid_set = set(latest_players["uuid_fmt"])

for _, row in latest_players.iterrows():
    elo = int(row["elo_rate"]) if pd.notna(row["elo_rate"]) else 0
    cursor.execute("""
        INSERT IGNORE INTO Player (user_id, username, elo, personal_best)
        VALUES (%s, %s, %s, NULL)
    """, (row["uuid_fmt"], str(row["nickname"]).strip(), elo))

conn.commit()
print(f"  {len(latest_players)} players inserted")

# nick -> formatted uuid lookup (most recent entry wins on duplicates)
nick_to_uuid = dict(zip(
    latest_players["nickname"].str.strip(),
    latest_players["uuid_fmt"]
))

# =====================
# 2a. INSERT SeasonRanking  (one row per player per season)
# =====================
print("Inserting season rankings...")
players_df["uuid_fmt"] = players_df["uuid"].apply(fmt_uuid)
season_rows = players_df[players_df["uuid_fmt"].isin(uuid_set)]
ranking_count = 0
for _, row in season_rows.iterrows():
    if pd.isna(row["season"]) or pd.isna(row["elo_rate"]):
        continue
    cursor.execute("""
        INSERT IGNORE INTO SeasonRanking (user_id, season, elo)
        VALUES (%s, %s, %s)
    """, (row["uuid_fmt"], int(row["season"]), int(row["elo_rate"])))
    ranking_count += 1
conn.commit()
print(f"  {ranking_count} season ranking rows inserted")

# =====================
# 2. INSERT PlayerStats  (one row per player)
# =====================
print("Inserting player stats...")
for uuid_fmt in uuid_set:
    cursor.execute("""
        INSERT IGNORE INTO PlayerStats (user_id, win_count, loss_count, draw_count, game_count)
        VALUES (%s, 0, 0, 0, 0)
    """, (uuid_fmt,))
    # INSERT IGNORE won't help here since there's no UNIQUE key on user_id —
    # but we only loop once per uuid so duplicates won't be created.

conn.commit()
print("  PlayerStats inserted")

# =====================
# 3. INSERT Tournaments & Rounds
#    Playoff JSON structure per season:
#      champion   : str  (Final winner)
#      finalist   : str  (Final loser)
#      top4       : [str, str]  (Semifinal losers)
#      qf_exit    : [str, str, str, str]  (Quarterfinal losers)
#      r1_exit    : [str*8]  (Round-of-16 losers)
# =====================
print("Inserting tournaments and rounds...")
tournament_ids = {}
round_ids = {}

# Map playoff JSON keys to round names and winner resolution
ROUND_MAP = [
    # (round_name,  who_won_this_round)
    ("Round of 16",  None),   # 8 winners — no single winner_id
    ("Quarterfinal", None),   # 4 winners — no single winner_id
    ("Semifinal",    None),   # 2 winners — no single winner_id
    ("Final",        "champion"),  # champion field holds the winner
]

for season_str, data in playoffs.items():
    season = int(season_str)

    # Skip if this tournament already exists
    cursor.execute(
        "SELECT tournament_id FROM Tournament WHERE tournament_name = %s",
        (f"MCSR Playoffs Season {season}",)
    )
    existing = cursor.fetchone()
    if existing:
        tournament_ids[season] = existing[0]
        cursor.execute(
            "SELECT round_id, round_name FROM Round WHERE tournament_id = %s",
            (existing[0],)
        )
        for rid, rname in cursor.fetchall():
            round_ids[(season, rname)] = rid
        continue

    # Use approximate year: season 1 started ~2021
    approx_year = 2021 + (season - 1)
    cursor.execute("""
        INSERT INTO Tournament (tournament_name, start_date, prize_money)
        VALUES (%s, %s, NULL)
    """, (f"MCSR Playoffs Season {season}", f"{approx_year}-01-01"))
    t_id = cursor.lastrowid
    tournament_ids[season] = t_id

    champ_name = data.get("champion")
    champ_uuid = nick_to_uuid.get(champ_name)

    for rname, winner_key in ROUND_MAP:
        winner_uuid = champ_uuid if winner_key == "champion" else None
        cursor.execute("""
            INSERT INTO Round (tournament_id, round_name, winner_id)
            VALUES (%s, %s, %s)
        """, (t_id, rname, winner_uuid))
        round_ids[(season, rname)] = cursor.lastrowid

conn.commit()
print(f"  {len(tournament_ids)} tournaments and rounds inserted")

# =====================
# 4. INSERT Matches
#    Only process type-2 matches (standard 1v1, no spectator entries).
#    Filter players to roleType != 3 (exclude spectators in type-3 rows).
# =====================
print("Inserting matches...")

matches_df = matches_df[matches_df["type"] == 2].copy()

inserted_match_ids = set()
ladder_count = 0
skipped = 0

for _, row in matches_df.iterrows():
    match_id = int(row["id"])
    if match_id in inserted_match_ids:
        continue

    # Parse players list; exclude spectators (roleType == 3)
    try:
        players_raw = ast.literal_eval(str(row["players"]))
        players_raw = [p for p in players_raw if p.get("roleType") != 3]
    except Exception:
        skipped += 1
        continue

    if len(players_raw) != 2:
        skipped += 1
        continue

    # Parse winner uuid
    try:
        result_raw = ast.literal_eval(str(row["result"]))
        winner_uuid_raw = str(result_raw.get("uuid", "")).strip()
        winner_uuid = fmt_uuid(winner_uuid_raw)
    except Exception:
        winner_uuid = None

    if not winner_uuid or winner_uuid not in uuid_set:
        skipped += 1
        continue

    # Parse match date (epoch ms)
    try:
        match_date = datetime.fromtimestamp(int(row["date"]) / 1000)
    except Exception:
        skipped += 1
        continue

    # Seed: the CSV stores a dict; extract integer id (usually None → store NULL)
    world_seed = None
    try:
        seed_dict = ast.literal_eval(str(row["seed"]))
        seed_id = seed_dict.get("id")
        world_seed = int(seed_id) if seed_id is not None else None
    except Exception:
        world_seed = None

    season = int(row["season"]) if pd.notna(row["season"]) else 1

    cursor.execute("""
        INSERT IGNORE INTO `Match` (match_id, world_seed, match_date, winner_id)
        VALUES (%s, %s, %s, %s)
    """, (match_id, world_seed, match_date, winner_uuid))

    cursor.execute("""
        INSERT IGNORE INTO LadderMatch (match_id, season)
        VALUES (%s, %s)
    """, (match_id, season))

    inserted_match_ids.add(match_id)
    ladder_count += 1

    # Participation (exactly 2 players)
    for p in players_raw:
        try:
            p_uuid = fmt_uuid(str(p.get("uuid", "")))
            if p_uuid and p_uuid in uuid_set:
                cursor.execute("""
                    INSERT IGNORE INTO Participation (user_id, match_id)
                    VALUES (%s, %s)
                """, (p_uuid, match_id))
        except Exception:
            continue

conn.commit()
print(f"  {ladder_count} matches inserted, {skipped} skipped")

# =====================
# 5. INSERT TournamentMatch
#    Reconstruct playoff brackets from JSON and fabricate pairings.
#    Bracket sizes: R16 (8 matches), QF (4), SF (2), Final (1) = 15 per tournament.
# =====================
print("Inserting tournament matches...")
tm_count = 0
tm_skipped = 0

for season_str, data in playoffs.items():
    season = int(season_str)
    t_id = tournament_ids.get(season)
    if t_id is None:
        continue

    champion = nick_to_uuid.get(data.get("champion"))
    finalist = nick_to_uuid.get(data.get("finalist"))
    top4     = [nick_to_uuid.get(n) for n in data.get("top4", [])]
    qf_exit  = [nick_to_uuid.get(n) for n in data.get("qf_exit", [])]
    r1_exit  = [nick_to_uuid.get(n) for n in data.get("r1_exit", [])]

    if not champion or not finalist or len(top4) < 2 or len(qf_exit) < 4 or len(r1_exit) < 8:
        tm_skipped += 1
        continue

    # Synthesize a date: tournaments started around year 2021+season-1
    base_date = datetime(2021 + (season - 1), 6, 1)

    # winners of each round (advance to next)
    qf_winners = [champion, top4[0], finalist, top4[1]]  # advance to SF
    sf_winners = [champion, finalist]
    final_winner = champion

    pairings = []  # (round_name, winner_uuid, loser_uuid, day_offset)
    # Round of 16: 8 matches
    r16_winners = qf_winners + qf_exit  # 8 R16-winners (they all made QF)
    for i, loser in enumerate(r1_exit):
        pairings.append(("Round of 16", r16_winners[i], loser, 0))
    # Quarterfinal: 4 matches
    for w, l in zip(qf_winners, qf_exit):
        pairings.append(("Quarterfinal", w, l, 7))
    # Semifinal: 2 matches
    pairings.append(("Semifinal", champion, top4[0], 14))
    pairings.append(("Semifinal", finalist, top4[1], 14))
    # Final
    pairings.append(("Final", final_winner, finalist, 21))

    for rname, winner, loser, day_offset in pairings:
        rid = round_ids.get((season, rname))
        if not winner or not loser or rid is None:
            tm_skipped += 1
            continue
        match_date = base_date + timedelta(days=day_offset)
        cursor.execute("""
            INSERT INTO `Match` (world_seed, match_date, winner_id)
            VALUES (NULL, %s, %s)
        """, (match_date, winner))
        new_match_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO TournamentMatch (match_id, round_id)
            VALUES (%s, %s)
        """, (new_match_id, rid))

        for p_uuid in (winner, loser):
            cursor.execute("""
                INSERT IGNORE INTO Participation (user_id, match_id)
                VALUES (%s, %s)
            """, (p_uuid, new_match_id))
        tm_count += 1

conn.commit()
print(f"  {tm_count} tournament matches inserted, {tm_skipped} skipped")

# =====================
# 6. INSERT Splits (synthetic)
#    Generate 4 milestones per match for both participants.
# =====================
print("Inserting splits...")
SPLIT_TEMPLATES = [
    ("nether_entry",  (60000, 120000)),
    ("fortress",      (180000, 300000)),
    ("end_entry",     (420000, 540000)),
    ("dragon_kill",   (540000, 720000)),
]

cursor.execute("""
    SELECT p.match_id, p.user_id, m.winner_id
    FROM Participation p
    JOIN `Match` m ON m.match_id = p.match_id
""")
part_rows = cursor.fetchall()

# Group by match: { match_id: [(user_id, is_winner), ...] }
match_players = {}
for match_id, user_id, winner_id in part_rows:
    match_players.setdefault(match_id, []).append((user_id, user_id == winner_id))

split_count = 0
for match_id, players in match_players.items():
    if len(players) != 2:
        continue
    # Assign each split time per player; winner generally faster
    winner_times = {}
    loser_times = {}
    for name, (lo, hi) in SPLIT_TEMPLATES:
        w_time = random.randint(lo, int(lo + (hi - lo) * 0.6))
        l_time = w_time + random.randint(2000, 30000)
        winner_times[name] = w_time
        loser_times[name] = l_time

    for user_id, is_winner in players:
        times = winner_times if is_winner else loser_times
        other = loser_times if is_winner else winner_times
        for name, _ in SPLIT_TEMPLATES:
            t = times[name]
            diff = t - other[name]
            cursor.execute("""
                INSERT INTO Split (user_id, match_id, split_name, split_time, time_difference)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, match_id, name, t, diff))
            split_count += 1

conn.commit()
print(f"  {split_count} splits inserted")

# =====================
# 7. BACKFILL PlayerStats counters and personal_best
# =====================
print("Backfilling player stats and personal bests...")

cursor.execute("""
    UPDATE PlayerStats ps
    JOIN (
        SELECT p.user_id,
               SUM(CASE WHEN m.winner_id = p.user_id THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN m.winner_id IS NOT NULL
                         AND m.winner_id <> p.user_id THEN 1 ELSE 0 END) AS losses,
               COUNT(*) AS games
        FROM Participation p
        JOIN `Match` m ON m.match_id = p.match_id
        GROUP BY p.user_id
    ) agg ON agg.user_id = ps.user_id
    SET ps.win_count  = agg.wins,
        ps.loss_count = agg.losses,
        ps.game_count = agg.games
""")

cursor.execute("""
    UPDATE PlayerStats ps
    JOIN (
        SELECT s.user_id, AVG(s.split_time) AS avg_t
        FROM Split s
        JOIN `Match` m ON m.match_id = s.match_id
        WHERE s.split_name = 'dragon_kill' AND m.winner_id = s.user_id
        GROUP BY s.user_id
    ) agg ON agg.user_id = ps.user_id
    SET ps.average_time = agg.avg_t
""")

cursor.execute("""
    UPDATE Player pl
    JOIN (
        SELECT s.user_id, MIN(s.split_time) AS pb
        FROM Split s
        JOIN `Match` m ON m.match_id = s.match_id
        WHERE s.split_name = 'dragon_kill' AND m.winner_id = s.user_id
        GROUP BY s.user_id
    ) agg ON agg.user_id = pl.user_id
    SET pl.personal_best = agg.pb
""")

conn.commit()
print("  PlayerStats + personal_best updated")

# =====================
# 8. SUMMARY
# =====================
print("\n--- Summary ---")
for table in ["Player", "SeasonRanking", "PlayerStats", "Tournament", "Round", "`Match`",
              "LadderMatch", "TournamentMatch", "Participation", "Split"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  {table}: {count} rows")

cursor.close()
conn.close()
print("\nDone!")
