import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return features.load_all_matches()


@pytest.fixture(scope="session")
def results():
    return features.load_playoff_results()


@pytest.fixture(scope="session")
def lcq():
    return features.load_lcq_by_season()


@pytest.fixture(scope="session")
def deltas():
    return features.load_delta_by_season()


@pytest.fixture(scope="session")
def players(results):
    """Every player appearing in any bracket tier, across all seasons.

    Derived from bracket data rather than a hardcoded roster: the hardcoded
    lists in scraper.py hold nicknames as announced at the time, which the API
    no longer agrees with for at least Season 6.
    """
    names = set()
    for season in results.values():
        for value in season.values():
            if isinstance(value, str):
                names.add(value)
            elif isinstance(value, list):
                names.update(value)
    return sorted(names)
