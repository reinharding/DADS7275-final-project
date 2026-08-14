"""
MCSR Ranked API Scraper
Collects playoff participant data, match history, and Elo stats
for Seasons 1-9.
"""

import json
import os
import sys
import time

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://api.mcsrranked.com"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Known playoff participants per season (UUIDs or nicknames to be filled in)
# Source: MCSR Ranked official playoff announcements
# Format: {season: [player_nickname, ...]}
PLAYOFF_PLAYERS = {
    1: [
        "silverrruns", "dandannyboy", "Oxidiot", "Reignex", "priffie", "orachi_",
        "lowk3y_", "7rowl", "doogile", "Ancoboyy", "pulsar32", "Ranik_",
        "MoleyG", "CroProYT", "AutomattPL", "Dylqn",
    ],
    2: [
        "lowk3y_", "CroProYT", "dandannyboy", "doogile", "7rowl", "kW1st",
        "priffie", "dwoh", "silverrruns", "Emillk", "bing_pigs", "drx6",
        "Ancoboyy", "Ranik_", "Oxidiot", "AutomattPL",
    ],
    3: [
        "7rowl", "Ancoboyy", "dandannyboy", "doogile", "hackingnoises", "lowk3y_",
        "Oxidiot", "priffie", "ANJOUU", "AutomattPL", "BeefSalad", "Bloonskiller",
        "loodlow", "paplerr", "v_strid", "autoqualler",
    ],
    4: [
        "7rowl", "Ancoboyy", "dandannyboy", "doogile", "AutomattPLUS", "hackingnoises",
        "Hinart", "lowk3y_", "Oxidiot", "paplerr", "priffie", "silverrruns",
        "ANJOUU", "bing_pigs", "Cube1337x", "v_strid",
    ],
    5: [
        "7rowl", "Ancoboyy", "BeefSalad", "bing_pigs", "doogile", "AutomattPLUS",
        "hackingnoises", "lowk3y_", "Oxidiot", "silverrruns", "TUDORULE", "v_strid",
        "Aquacorde", "dandannyboy", "KenanKardes", "pulsar32",
    ],
    6: [
        "7rowl", "Ayreliaa", "BeefSalad", "bing_pigs", "doogile", "AutomattPLUS",
        "Feinberg", "hackingnoises", "lowk3y_", "MrBudgiee", "Oxidiot", "silverrruns",
        "dandannyboy", "Erikfzf", "ogurikappa", "TUDORULE",
    ],
    7: [
        "7rowl", "Ancoboyy", "Aquacorde", "BadGamer", "BeefSalad", "bing_pigs",
        "doogile", "Feinberg", "Infume", "lowk3y_", "priffie", "retropog",
        "r7sD4fH6jK0wY5uB", "hackingnoises", "Oxidiot", "silverrruns",
    ],
    8: [
        "7rowl", "Aquacorde", "BeefSalad", "bing_pigs", "DARVY__X1", "doogile",
        "edcr", "Feinberg", "Infume", "lowk3y_", "Ranik_", "silverrruns",
        "hackingnoises", "KenanKardes", "TUDORULE", "v_strid",
    ],
    9: [
        "Feinberg", "Infume", "edcr", "steez", "hackingnoises", "Aquacorde",
        "nhb_", "silverrruns", "Pinne", "BeefSalad", "nahhann", "lowk3y_",
        "doogile", "HDMICables", "bing_pigs", "BlazeMind",
    ],
}


def get(endpoint: str, params: dict = None, retries: int = 5) -> dict | None:
    """Make a GET request with retry on timeout and rate-limit handling."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt * 15
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"HTTP {resp.status_code} for {url}")
                return None
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            wait = 2 ** attempt * 5
            print(f"Timeout/connection error (attempt {attempt + 1}/{retries}), retrying in {wait}s...")
            time.sleep(wait)
    return None


def get_player_profile(nickname: str) -> dict | None:
    """Fetch a player's profile by nickname."""
    data = get(f"/users/{nickname}")
    if data and data.get("status") == "success":
        return data["data"]
    return None


def get_player_matches(nickname: str, season: int = None) -> list:
    """Fetch up to 100 most recent matches for a player (API max per request)."""
    params = {"count": 100}
    if season is not None:
        params["season"] = season
    data = get(f"/users/{nickname}/matches", params=params)
    if data and data.get("status") == "success":
        return data["data"] or []
    return []


def get_player_stats(nickname: str, season: int = None) -> dict | None:
    """Fetch aggregated stats for a player."""
    params = {}
    if season is not None:
        params["season"] = season
    data = get(f"/users/{nickname}/statistics", params=params)
    if data and data.get("status") == "success":
        return data["data"]
    return None


def get_versus(player1: str, player2: str) -> dict | None:
    """Fetch head-to-head stats between two players."""
    data = get(f"/users/{player1}/versus/{player2}")
    if data and data.get("status") == "success":
        return data["data"]
    # fallback to alternate endpoint
    data = get(f"/versus/{player1}/{player2}")
    if data and data.get("status") == "success":
        return data["data"]
    return None


def get_leaderboard(season: int = None) -> list:
    """Fetch the Elo leaderboard, optionally for a specific season."""
    params = {}
    if season is not None:
        params["season"] = season
    data = get("/leaderboard", params=params)
    if data and data.get("status") == "success":
        return data["data"] or []
    return []


def collect_player_data(players: list[str], season: int) -> list[dict]:
    """
    For each player collect profile + stats for the given season.
    Returns a list of flat dicts ready for DataFrame construction.
    """
    records = []
    for nickname in tqdm(players, desc=f"Season {season} players"):
        profile = get_player_profile(nickname)
        stats = get_player_stats(nickname, season=season)
        time.sleep(0.15)  # stay under 500 req/10 min

        record = {"nickname": nickname, "season": season}
        if profile:
            record["uuid"] = profile.get("uuid")
            record["elo_rate"] = profile.get("eloRate")
            record["elo_rank"] = profile.get("eloRank")
            record["country"] = profile.get("country")

        if stats:
            record["season_wins"] = stats.get("wins")
            record["season_losses"] = stats.get("losses")
            record["season_matches"] = stats.get("playedMatches")
            record["winrate"] = stats.get("winRate")
            record["best_time_ms"] = stats.get("bestTime")
            record["avg_time_ms"] = stats.get("avgTime")
            record["forfeits"] = stats.get("forfeits")
            record["completion_rate"] = stats.get("completionRate")

        records.append(record)

    return records


def collect_match_history(players: list[str], season: int) -> list[dict]:
    """Collect all ranked match records for the given players/season."""
    all_matches = {}
    for nickname in tqdm(players, desc=f"Season {season} matches"):
        matches = get_player_matches(nickname, season=season)
        for m in matches:
            mid = m.get("id")
            if mid and mid not in all_matches:
                all_matches[mid] = m
        time.sleep(0.15)
    return list(all_matches.values())


def flatten_versus(player1: str, player2: str, versus: dict) -> dict:
    """Flatten a nested versus response into a single-level dict."""
    record = {"player1": player1, "player2": player2}
    for key, val in versus.items():
        if isinstance(val, dict):
            # prefix nested keys, e.g. {"player1": {"wins": 3}} -> player1_wins
            for subkey, subval in val.items():
                record[f"{key}_{subkey}"] = subval
        elif isinstance(val, list):
            record[f"{key}_count"] = len(val)
        else:
            record[key] = val
    return record


def collect_head_to_head(players: list[str]) -> list[dict]:
    """Collect head-to-head records for all player pairs."""
    records = []
    pairs = [(players[i], players[j]) for i in range(len(players)) for j in range(i + 1, len(players))]
    for p1, p2 in tqdm(pairs, desc="H2H records"):
        versus = get_versus(p1, p2)
        if versus:
            records.append(flatten_versus(p1, p2, versus))
        time.sleep(0.15)
    return records


def save_json(data, filename: str):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} records -> {path}")


# ---------------------------------------------------------------------------
# PLAYOFF BRACKET SCRAPING
# ---------------------------------------------------------------------------

def fetch_playoff_page(playoff_id: int = None) -> dict | None:
    """
    Fetch a raw playoff page from the API.
    Uses path param /playoffs/{id} for historical seasons (prev/next are IDs, not season numbers).
    Falls back to /playoffs for the latest.
    """
    url = f"{BASE_URL}/playoffs/{playoff_id}" if playoff_id is not None else f"{BASE_URL}/playoffs"
    params = {}
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()   # full envelope: {data, prev, next}
            elif resp.status_code == 429:
                wait = 2 ** attempt * 15
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code} for {url}")
                return None
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            wait = 2 ** attempt * 5
            print(f"  Timeout (attempt {attempt+1}/5), retrying in {wait}s...")
            time.sleep(wait)
    return None


def _parse_inner(inner: dict) -> dict:
    """
    Parse the innermost bracket dict (envelope["data"]["data"]) into tier categories.

    Structure:
      inner["players"]  -> list of {uuid, nickname, seedNumber, ...}  (index == seedNumber)
      inner["results"]  -> list of {player: seedNumber, place: int}
                             place 1   = champion
                             place 2   = finalist
                             place 3-4 = top4
                             place 5-8 = qf_exit
                             place 9+  = r1_exit
    """
    players = inner.get("players", [])
    results = inner.get("results", [])

    seed_to_nick = {p.get("seedNumber", i): p.get("nickname", "")
                    for i, p in enumerate(players)}

    tier = {"champion": None, "finalist": None,
            "top4": [], "qf_exit": [], "r1_exit": []}

    for r in results:
        seed  = r.get("player")
        place = r.get("place")
        nick  = seed_to_nick.get(seed, "")
        if not nick:
            continue
        if place == 1:
            tier["champion"] = nick
        elif place == 2:
            tier["finalist"] = nick
        elif place in (3, 4):
            tier["top4"].append(nick)
        elif 5 <= place <= 8:
            tier["qf_exit"].append(nick)
        else:
            tier["r1_exit"].append(nick)

    return tier


def parse_playoff_bracket(envelope: dict) -> tuple[int, dict]:
    """Parse full API envelope -> (season_number, tier_dict). Kept for compatibility."""
    outer  = envelope.get("data") or {}
    inner  = outer.get("data") or {}
    season = inner.get("season")
    return season, _parse_inner(inner)


def collect_playoff_brackets():
    """
    Navigate the API backwards from the latest playoff using the 'prev' field,
    collecting all available seasons. Saves each raw page and a combined summary.
    """
    print("\n=== Scraping Playoff Brackets (navigating backwards via prev IDs) ===")
    all_parsed  = {}
    seen_ids    = set()
    playoff_id  = None   # start from /playoffs (latest)

    while True:
        print(f"\n  Fetching /playoffs/{playoff_id if playoff_id is not None else '(latest)'}...")
        envelope = fetch_playoff_page(playoff_id)
        if envelope is None:
            print("  No response, stopping.")
            break

        # Structure: envelope["data"] = outer (prev/next)
        #            envelope["data"]["data"] = inner (players/results/matches/season)
        outer = envelope.get("data") or {}
        inner = outer.get("data") or {}

        season_num = inner.get("season")
        prev       = outer.get("prev")
        print(f"  inner.season={season_num}  outer.prev={prev}")

        if season_num is None or season_num in seen_ids:
            print(f"  Skipping (None or duplicate), stopping.")
            break
        seen_ids.add(season_num)

        # Save raw envelope
        raw_path = os.path.join(DATA_DIR, f"season_{season_num}_playoffs.json")
        with open(raw_path, "w") as f:
            json.dump(envelope, f, indent=2)
        print(f"  season={season_num} -> saved {raw_path}")

        # Parse using inner dict directly
        parsed = _parse_inner(inner)
        all_parsed[season_num] = parsed

        print(f"  Champion : {parsed['champion'] or '???'}")
        print(f"  Finalist : {parsed['finalist'] or '???'}")
        print(f"  Top 4    : {parsed['top4'] or '???'}")
        print(f"  QF exit  : {parsed['qf_exit'] or '???'}")
        print(f"  R1 exit  : {parsed['r1_exit'] or '???'}")

        if prev is None:
            print("  No 'prev' link, reached oldest season.")
            break
        playoff_id = prev
        time.sleep(0.5)

    # Save combined parsed summary
    summary_path = os.path.join(DATA_DIR, "all_playoff_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_parsed, f, indent=2)
    print(f"\nAll parsed results -> {summary_path}")
    print(f"Seasons collected  : {sorted(all_parsed.keys())}")
    return all_parsed


def run_collection():
    """Main collection routine: gather all data for all seasons."""
    all_player_stats = []
    all_matches = []
    all_h2h = []

    for season, players in PLAYOFF_PLAYERS.items():
        if not players:
            print(f"Season {season}: no players listed, skipping.")
            continue

        print(f"\n=== Season {season} ({len(players)} players) ===")

        # Player profiles + season stats
        player_records = collect_player_data(players, season)
        all_player_stats.extend(player_records)
        save_json(player_records, f"season_{season}_players.json")

        # Match history
        matches = collect_match_history(players, season)
        all_matches.extend(matches)
        save_json(matches, f"season_{season}_matches.json")

        # Head-to-head (all pairs)
        h2h = collect_head_to_head(players)
        all_h2h.extend(h2h)
        save_json(h2h, f"season_{season}_h2h.json")

    # Save combined CSVs
    pd.DataFrame(all_player_stats).to_csv(os.path.join(DATA_DIR, "all_player_stats.csv"), index=False)
    pd.DataFrame(all_matches).to_csv(os.path.join(DATA_DIR, "all_matches.csv"), index=False)
    pd.DataFrame(all_h2h).to_csv(os.path.join(DATA_DIR, "all_h2h.csv"), index=False)
    print("\nCollection complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "playoffs":
        # Run only playoff bracket scraping: python scraper.py playoffs
        collect_playoff_brackets()
    else:
        run_collection()
