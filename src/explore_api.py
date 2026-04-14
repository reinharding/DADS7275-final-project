"""
Quick API explorer — run this to test endpoints and see raw response shapes
before committing to the full collection run.
"""

import requests
import json

BASE_URL = "https://api.mcsrranked.com"


def show(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=15)
    print(f"\n--- GET {url} (params={params}) ---")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2)[:2000])  # truncate large responses
    except Exception:
        print(resp.text[:500])


if __name__ == "__main__":
    for test_player in ["Feinberg", "Infume"]:
        print(f"\n{'='*50}\nTesting player: {test_player}\n{'='*50}")
        show(f"/users/{test_player}")
        show(f"/users/{test_player}/statistics", {"season": 9})
        show(f"/users/{test_player}/matches", {"count": 3, "season": 9})

    # Head-to-head between the two
    show("/versus/Feinberg/Infume")
    show("/leaderboard", {"season": 9})
