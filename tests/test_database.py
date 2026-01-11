"""Tests for database operations."""
import sys
import pytest
import tempfile
from pathlib import Path

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import Database


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
def sample_players():
    """Sample player data for a match."""
    return [
        {'name': 'RedPlayer1', 'profession': 'Warrior', 'team': 'red'},
        {'name': 'RedPlayer2', 'profession': 'Necromancer', 'team': 'red'},
        {'name': 'RedPlayer3', 'profession': 'Guardian', 'team': 'red'},
        {'name': 'RedPlayer4', 'profession': 'Thief', 'team': 'red'},
        {'name': 'RedPlayer5', 'profession': 'Mesmer', 'team': 'red'},
        {'name': 'BluePlayer1', 'profession': 'Elementalist', 'team': 'blue'},
        {'name': 'BluePlayer2', 'profession': 'Ranger', 'team': 'blue'},
        {'name': 'BluePlayer3', 'profession': 'Engineer', 'team': 'blue'},
        {'name': 'BluePlayer4', 'profession': 'Revenant', 'team': 'blue'},
        {'name': 'BluePlayer5', 'profession': 'Warrior', 'team': 'blue'},
    ]


class TestLogMatchCreatesMatchRecord:
    """Test that log_match creates a match record correctly."""

    def test_match_record_inserted(self, test_db, sample_players):
        """Verify that calling log_match inserts a match record."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players,
            screenshot_start_path='/path/to/start.png',
            screenshot_end_path='/path/to/end.png'
        )

        assert match_id is not None
        assert match_id > 0

    def test_match_record_has_correct_scores(self, test_db, sample_players):
        """Verify match record stores correct scores."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # Query the match directly
        cursor = test_db.connection.cursor()
        cursor.execute("SELECT red_score, blue_score FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row['red_score'] == 347
        assert row['blue_score'] == 500

    def test_match_record_determines_winning_team(self, test_db, sample_players):
        """Verify winning_team is correctly determined from scores."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("SELECT winning_team FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()

        assert row['winning_team'] == 'blue'  # Blue had higher score

    def test_match_record_stores_user_info(self, test_db, sample_players):
        """Verify match record stores user team and character name."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("SELECT user_team, user_char_name FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()

        assert row['user_team'] == 'red'
        assert row['user_char_name'] == 'RedPlayer1'

    def test_match_record_stores_arena_type(self, test_db, sample_players):
        """Verify match record stores arena type when provided."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players,
            arena_type='ranked'
        )

        cursor = test_db.connection.cursor()
        cursor.execute("SELECT arena_type FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()

        assert row['arena_type'] == 'ranked'


class TestLogMatchCreatesParticipants:
    """Test that log_match creates participant records correctly."""

    def test_creates_10_participants(self, test_db, sample_players):
        """Verify that exactly 10 participant records are created."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM match_participants WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()

        assert row['count'] == 10

    def test_participants_have_correct_teams(self, test_db, sample_players):
        """Verify participants are assigned to correct teams."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("""
            SELECT team_color, COUNT(*) as count
            FROM match_participants
            WHERE match_id = ?
            GROUP BY team_color
        """, (match_id,))
        rows = cursor.fetchall()

        team_counts = {row['team_color']: row['count'] for row in rows}
        assert team_counts.get('red') == 5
        assert team_counts.get('blue') == 5

    def test_participants_have_correct_professions(self, test_db, sample_players):
        """Verify participants have correct professions stored."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("""
            SELECT char_name, profession
            FROM match_participants
            WHERE match_id = ?
        """, (match_id,))
        rows = cursor.fetchall()

        participant_professions = {row['char_name']: row['profession'] for row in rows}

        assert participant_professions['RedPlayer1'] == 'Warrior'
        assert participant_professions['BluePlayer1'] == 'Elementalist'

    def test_user_participant_is_flagged(self, test_db, sample_players):
        """Verify user's participant record has is_user=1."""
        match_id = test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("""
            SELECT char_name, is_user
            FROM match_participants
            WHERE match_id = ?
        """, (match_id,))
        rows = cursor.fetchall()

        user_flags = {row['char_name']: row['is_user'] for row in rows}

        assert user_flags['RedPlayer1'] == 1
        # Other players should not be flagged as user
        assert user_flags['RedPlayer2'] == 0
        assert user_flags['BluePlayer1'] == 0


class TestLogMatchUpdatesPlayerStats:
    """Test that log_match updates player statistics correctly."""

    def test_creates_player_records(self, test_db, sample_players):
        """Verify player records are created for all participants."""
        test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        cursor = test_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM players")
        row = cursor.fetchone()

        assert row['count'] == 10

    def test_winning_team_gets_wins(self, test_db, sample_players):
        """Verify players on winning team get wins incremented."""
        test_db.log_match(
            red_score=347,
            blue_score=500,  # Blue wins
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # Blue team should have wins
        blue_player = test_db.get_player('BluePlayer1')
        assert blue_player is not None
        assert blue_player['global_wins'] == 1
        assert blue_player['global_losses'] == 0
        assert blue_player['total_matches'] == 1

    def test_losing_team_gets_losses(self, test_db, sample_players):
        """Verify players on losing team get losses incremented."""
        test_db.log_match(
            red_score=347,
            blue_score=500,  # Blue wins, red loses
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # Red team should have losses
        red_player = test_db.get_player('RedPlayer1')
        assert red_player is not None
        assert red_player['global_wins'] == 0
        assert red_player['global_losses'] == 1
        assert red_player['total_matches'] == 1

    def test_multiple_matches_accumulate_stats(self, test_db, sample_players):
        """Verify stats accumulate across multiple matches."""
        # Match 1: Blue wins
        test_db.log_match(
            red_score=347,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # Match 2: Red wins
        test_db.log_match(
            red_score=500,
            blue_score=300,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # RedPlayer1 should have 1 win and 1 loss
        red_player = test_db.get_player('RedPlayer1')
        assert red_player['global_wins'] == 1
        assert red_player['global_losses'] == 1
        assert red_player['total_matches'] == 2


class TestGetPlayerWinrate:
    """Test that get_player_winrate returns correct values."""

    def test_returns_zero_for_unknown_player(self, test_db):
        """Verify unknown player returns 0% win rate and 0 matches."""
        win_rate, total_matches = test_db.get_player_winrate('NonExistentPlayer')

        assert win_rate == 0.0
        assert total_matches == 0

    def test_returns_100_percent_for_all_wins(self, test_db, sample_players):
        """Verify player with only wins has 100% win rate."""
        # Log a match where blue wins
        test_db.log_match(
            red_score=300,
            blue_score=500,
            user_team='blue',
            user_char_name='BluePlayer1',
            players=sample_players
        )

        win_rate, total_matches = test_db.get_player_winrate('BluePlayer1')

        assert win_rate == 100.0
        assert total_matches == 1

    def test_returns_0_percent_for_all_losses(self, test_db, sample_players):
        """Verify player with only losses has 0% win rate."""
        # Log a match where blue wins (red loses)
        test_db.log_match(
            red_score=300,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        win_rate, total_matches = test_db.get_player_winrate('RedPlayer1')

        assert win_rate == 0.0
        assert total_matches == 1

    def test_returns_correct_percentage(self, test_db, sample_players):
        """Verify correct win rate percentage calculation."""
        # Match 1: Blue wins
        test_db.log_match(
            red_score=300,
            blue_score=500,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # Match 2: Red wins
        test_db.log_match(
            red_score=500,
            blue_score=300,
            user_team='red',
            user_char_name='RedPlayer1',
            players=sample_players
        )

        # RedPlayer1: 1 win, 1 loss = 50%
        win_rate, total_matches = test_db.get_player_winrate('RedPlayer1')

        assert win_rate == 50.0
        assert total_matches == 2

    def test_returns_correct_total_matches(self, test_db, sample_players):
        """Verify total_matches count is accurate."""
        # Log 3 matches
        for _ in range(3):
            test_db.log_match(
                red_score=500,
                blue_score=300,
                user_team='red',
                user_char_name='RedPlayer1',
                players=sample_players
            )

        win_rate, total_matches = test_db.get_player_winrate('RedPlayer1')

        assert total_matches == 3
        assert win_rate == 100.0  # Won all 3


class TestGetRecentMatches:
    """Test that get_recent_matches returns correct data."""

    def test_returns_empty_list_with_no_matches(self, test_db):
        """Verify empty list returned when no matches exist."""
        matches = test_db.get_recent_matches()
        assert matches == []

    def test_returns_matches_in_reverse_chronological_order(self, test_db, sample_players):
        """Verify matches are returned newest first."""
        # Log 3 matches
        match_ids = []
        for i in range(3):
            match_id = test_db.log_match(
                red_score=300 + i * 50,
                blue_score=500,
                user_team='red',
                user_char_name='RedPlayer1',
                players=sample_players
            )
            match_ids.append(match_id)

        matches = test_db.get_recent_matches(limit=3)

        # Newest match (highest ID) should be first
        assert len(matches) == 3
        assert matches[0]['match_id'] == match_ids[2]
        assert matches[2]['match_id'] == match_ids[0]

    def test_respects_limit_parameter(self, test_db, sample_players):
        """Verify limit parameter restricts results."""
        # Log 5 matches
        for _ in range(5):
            test_db.log_match(
                red_score=300,
                blue_score=500,
                user_team='red',
                user_char_name='RedPlayer1',
                players=sample_players
            )

        matches = test_db.get_recent_matches(limit=2)

        assert len(matches) == 2


class TestAddPlayer:
    """Test add_player functionality."""

    def test_adds_new_player(self, test_db):
        """Verify new player can be added."""
        result = test_db.add_player('TestPlayer')

        assert result is True
        player = test_db.get_player('TestPlayer')
        assert player is not None
        assert player['char_name'] == 'TestPlayer'

    def test_returns_false_for_duplicate(self, test_db):
        """Verify adding duplicate player returns False."""
        test_db.add_player('TestPlayer')
        result = test_db.add_player('TestPlayer')

        assert result is False

    def test_initializes_stats_to_zero(self, test_db):
        """Verify new player has zeroed stats."""
        test_db.add_player('TestPlayer')
        player = test_db.get_player('TestPlayer')

        assert player['global_wins'] == 0
        assert player['global_losses'] == 0
        assert player['total_matches'] == 0


class TestGetAllPlayerNames:
    """Test get_all_player_names functionality."""

    def test_returns_empty_list_when_no_players(self, test_db):
        """Verify empty list returned when no players exist."""
        names = test_db.get_all_player_names()
        assert names == []

    def test_returns_all_player_names(self, test_db):
        """Verify all player names are returned."""
        test_db.add_player('Player1')
        test_db.add_player('Player2')
        test_db.add_player('Player3')

        names = test_db.get_all_player_names()

        assert len(names) == 3
        assert 'Player1' in names
        assert 'Player2' in names
        assert 'Player3' in names
