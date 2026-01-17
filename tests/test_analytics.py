"""Tests for analytics engine."""
import sys
import pytest
import tempfile
import pandas as pd
from pathlib import Path

# Ensure src root is on sys.path so we can import modules as if running from src/
# This allows 'from database.models import Database' to work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from database.models import Database
from analysis.analytics_engine import AnalyticsEngine

@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    db = Database(db_path=db_path)
    yield db

    # Cleanup
    db.close()
    Path(db_path).unlink(missing_ok=True)

@pytest.fixture
def analytics_engine(test_db):
    """Create an AnalyticsEngine instance using the test database."""
    engine = AnalyticsEngine()
    # Replace the db instance with our test db
    engine.db = test_db
    return engine

def test_get_user_professions_empty(analytics_engine):
    assert analytics_engine.get_user_professions() == []

def test_get_matchup_stats_empty(analytics_engine):
    stats = analytics_engine.get_matchup_stats()
    assert stats.empty
    # Check that required columns are present even when empty
    expected_cols = ['enemy_profession', 'wins', 'total', 'win_rate']
    for col in expected_cols:
        assert col in stats.columns

def test_analytics_flow(test_db, analytics_engine):
    # Log a match where User (Warrior, Red) wins against Blue (Mesmer, Ranger)
    players_1 = [
        {'name': 'UserChar', 'profession': 'Warrior', 'team': 'red', 'is_user': True},
        {'name': 'Enemy1', 'profession': 'Mesmer', 'team': 'blue', 'is_user': False},
        {'name': 'Enemy2', 'profession': 'Ranger', 'team': 'blue', 'is_user': False}
    ]
    # log_match signature: red_score, blue_score, user_team, user_char_name, players
    test_db.log_match(500, 400, 'red', 'UserChar', players_1)

    # Log a match where User (Warrior, Red) loses against Blue (Mesmer, Thief)
    players_2 = [
        {'name': 'UserChar', 'profession': 'Warrior', 'team': 'red', 'is_user': True},
        {'name': 'Enemy1', 'profession': 'Mesmer', 'team': 'blue', 'is_user': False},
        {'name': 'Enemy3', 'profession': 'Thief', 'team': 'blue', 'is_user': False}
    ]
    test_db.log_match(400, 500, 'red', 'UserChar', players_2)

    # Check User Professions
    profs = analytics_engine.get_user_professions()
    assert 'Warrior' in profs
    assert len(profs) == 1

    # Check Stats for ALL
    stats = analytics_engine.get_matchup_stats()

    # We expect:
    # Vs Mesmer: 1 Win, 1 Loss -> 50% WR, 2 Games
    # Vs Ranger: 1 Win -> 100% WR, 1 Game
    # Vs Thief: 1 Loss -> 0% WR, 1 Game

    assert not stats.empty

    mesmer = stats[stats['enemy_profession'] == 'Mesmer'].iloc[0]
    assert mesmer['wins'] == 1
    assert mesmer['total'] == 2
    assert mesmer['win_rate'] == 50.0

    ranger = stats[stats['enemy_profession'] == 'Ranger'].iloc[0]
    assert ranger['wins'] == 1
    assert ranger['total'] == 1
    assert ranger['win_rate'] == 100.0

    thief = stats[stats['enemy_profession'] == 'Thief'].iloc[0]
    assert thief['wins'] == 0
    assert thief['total'] == 1
    assert thief['win_rate'] == 0.0

def test_filtering_by_profession(test_db, analytics_engine):
    # Match 1: User plays Warrior
    players_1 = [
        {'name': 'UserChar', 'profession': 'Warrior', 'team': 'red', 'is_user': True},
        {'name': 'Enemy1', 'profession': 'Mesmer', 'team': 'blue', 'is_user': False}
    ]
    test_db.log_match(500, 0, 'red', 'UserChar', players_1)

    # Match 2: User plays Guardian
    players_2 = [
        {'name': 'UserChar', 'profession': 'Guardian', 'team': 'red', 'is_user': True},
        {'name': 'Enemy1', 'profession': 'Mesmer', 'team': 'blue', 'is_user': False}
    ]
    test_db.log_match(0, 500, 'red', 'UserChar', players_2)

    # Filter for Warrior (Win vs Mesmer)
    stats_warrior = analytics_engine.get_matchup_stats('Warrior')
    mesmer_war = stats_warrior[stats_warrior['enemy_profession'] == 'Mesmer'].iloc[0]
    assert mesmer_war['win_rate'] == 100.0
    assert mesmer_war['total'] == 1

    # Filter for Guardian (Loss vs Mesmer)
    stats_guard = analytics_engine.get_matchup_stats('Guardian')
    mesmer_guard = stats_guard[stats_guard['enemy_profession'] == 'Mesmer'].iloc[0]
    assert mesmer_guard['win_rate'] == 0.0
    assert mesmer_guard['total'] == 1

def test_filtering_by_match_type(test_db, analytics_engine):
    # Match 1: Ranked Win vs Mesmer
    players_1 = [
        {'name': 'UserChar', 'profession': 'Warrior', 'team': 'red', 'is_user': True},
        {'name': 'Enemy1', 'profession': 'Mesmer', 'team': 'blue', 'is_user': False}
    ]
    test_db.log_match(500, 0, 'red', 'UserChar', players_1, arena_type='ranked')

    # Match 2: Unranked Loss vs Mesmer
    players_2 = [
        {'name': 'UserChar', 'profession': 'Warrior', 'team': 'red', 'is_user': True},
        {'name': 'Enemy1', 'profession': 'Mesmer', 'team': 'blue', 'is_user': False}
    ]
    test_db.log_match(0, 500, 'red', 'UserChar', players_2, arena_type='unranked')

    # Filter for Ranked (Win vs Mesmer)
    stats_ranked = analytics_engine.get_matchup_stats(match_type='Ranked')
    assert not stats_ranked.empty
    ranked_row = stats_ranked[stats_ranked['enemy_profession'] == 'Mesmer'].iloc[0]
    assert ranked_row['win_rate'] == 100.0
    assert ranked_row['total'] == 1

    # Filter for Unranked (Loss vs Mesmer)
    stats_unranked = analytics_engine.get_matchup_stats(match_type='Unranked')
    assert not stats_unranked.empty
    unranked_row = stats_unranked[stats_unranked['enemy_profession'] == 'Mesmer'].iloc[0]
    assert unranked_row['win_rate'] == 0.0
    assert unranked_row['total'] == 1

    # Filter for All (1 Win 1 Loss)
    stats_all = analytics_engine.get_matchup_stats(match_type='All')
    all_row = stats_all[stats_all['enemy_profession'] == 'Mesmer'].iloc[0]
    assert all_row['win_rate'] == 50.0
    assert all_row['total'] == 2
