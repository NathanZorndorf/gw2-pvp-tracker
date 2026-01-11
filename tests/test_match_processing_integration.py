"""End-to-end integration tests for match processing pipeline."""
import sys
import pytest
import tempfile
from pathlib import Path
import yaml

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.database.models import Database
from src.automation.match_processor import MatchProcessor
from src.config import Config


SAMPLES_DIR = Path("data/samples/ranked-1")
CONFIG_PATH = Path("config.yaml")


@pytest.fixture
def config():
    """Load the application config."""
    if not CONFIG_PATH.exists():
        pytest.skip("Config file not found")
    return Config(str(CONFIG_PATH))


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
def ground_truth():
    """Load ground truth data for the ranked samples."""
    gt_path = SAMPLES_DIR / "ground_truth.yaml"
    if not gt_path.exists():
        pytest.skip("Ground truth file not found")

    with open(gt_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def match_processor(config, test_db):
    """Create a MatchProcessor instance with test database."""
    try:
        import easyocr  # noqa: F401
    except ImportError:
        pytest.skip("EasyOCR not installed in test environment")

    return MatchProcessor(config, test_db)


@pytest.fixture
def sample_images():
    """Get paths to sample start and end images."""
    start_path = SAMPLES_DIR / "match_start_20260110_102550_full.png"
    end_path = SAMPLES_DIR / "match_end_20260110_103751_full.png"

    if not start_path.exists() or not end_path.exists():
        pytest.skip("Sample images not found")

    return str(start_path), str(end_path)


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Samples directory missing")
class TestMatchProcessingPipeline:
    """Test the full match processing pipeline."""

    def test_process_match_extracts_scores(
        self, match_processor, sample_images, ground_truth
    ):
        """Verify scores are extracted correctly from end frame."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)

        assert result['success'] is True

        # Get expected scores from ground truth
        end_sample = next(
            s for s in ground_truth['samples']
            if 'end' in s['filename']
        )
        expected_red = end_sample['scores']['red']
        expected_blue = end_sample['scores']['blue']

        assert result['red_score'] == expected_red, (
            f"Red score mismatch: expected {expected_red}, got {result['red_score']}"
        )
        assert result['blue_score'] == expected_blue, (
            f"Blue score mismatch: expected {expected_blue}, got {result['blue_score']}"
        )

    def test_process_match_extracts_10_players(
        self, match_processor, sample_images
    ):
        """Verify 10 players are extracted from start frame."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)

        assert result['success'] is True
        assert len(result['players']) == 10

    def test_process_match_extracts_player_names(
        self, match_processor, sample_images, ground_truth
    ):
        """Verify player names match ground truth (fuzzy match allowed)."""
        from thefuzz import fuzz

        start_path, end_path = sample_images
        result = match_processor.process_match(start_path, end_path)

        assert result['success'] is True

        # Get expected names from ground truth
        start_sample = next(
            s for s in ground_truth['samples']
            if 'start' in s['filename']
        )
        expected_red = start_sample['red_team']
        expected_blue = start_sample['blue_team']

        # Extract names by team from result
        extracted_red = [p['name'] for p in result['players'] if p['team'] == 'red']
        extracted_blue = [p['name'] for p in result['players'] if p['team'] == 'blue']

        # Allow fuzzy matching (80% threshold) since OCR may have minor variations
        fuzzy_threshold = 80

        def token_sort_ratio(s1, s2):
            """Compare strings by sorting tokens (handles word order differences)."""
            return fuzz.token_sort_ratio(s1.lower(), s2.lower())

        # Check red team
        for expected in expected_red:
            best_match = max(
                extracted_red,
                key=lambda x: token_sort_ratio(x, expected)
            )
            similarity = token_sort_ratio(best_match, expected)
            assert similarity >= fuzzy_threshold, (
                f"Red player '{expected}' not found. Best match: '{best_match}' ({similarity}%)"
            )

        # Check blue team
        for expected in expected_blue:
            best_match = max(
                extracted_blue,
                key=lambda x: token_sort_ratio(x, expected)
            )
            similarity = token_sort_ratio(best_match, expected)
            assert similarity >= fuzzy_threshold, (
                f"Blue player '{expected}' not found. Best match: '{best_match}' ({similarity}%)"
            )

    def test_process_match_detects_arena_type(
        self, match_processor, sample_images
    ):
        """Verify arena type is detected as ranked."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)

        assert result['success'] is True
        assert result['arena_type'] == 'ranked'

    def test_process_match_assigns_teams_correctly(
        self, match_processor, sample_images
    ):
        """Verify players are assigned to correct teams (5 each)."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)

        assert result['success'] is True

        red_count = sum(1 for p in result['players'] if p['team'] == 'red')
        blue_count = sum(1 for p in result['players'] if p['team'] == 'blue')

        assert red_count == 5, f"Expected 5 red players, got {red_count}"
        assert blue_count == 5, f"Expected 5 blue players, got {blue_count}"


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Samples directory missing")
class TestDatabaseIntegration:
    """Test that processed matches are correctly logged to database."""

    def test_log_match_creates_database_records(
        self, match_processor, sample_images, test_db
    ):
        """Verify log_match creates all required database records."""
        start_path, end_path = sample_images

        # Process the match
        result = match_processor.process_match(start_path, end_path)
        assert result['success'] is True

        # Log the match
        match_id = match_processor.log_match(result, start_path, end_path)

        # Verify match record exists
        cursor = test_db.connection.cursor()
        cursor.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
        match_record = cursor.fetchone()

        assert match_record is not None
        assert match_record['red_score'] == result['red_score']
        assert match_record['blue_score'] == result['blue_score']

    def test_log_match_creates_10_participants(
        self, match_processor, sample_images, test_db
    ):
        """Verify log_match creates 10 participant records."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)
        match_id = match_processor.log_match(result, start_path, end_path)

        # Check participant count
        cursor = test_db.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM match_participants WHERE match_id = ?",
            (match_id,)
        )
        row = cursor.fetchone()

        assert row['count'] == 10

    def test_log_match_creates_player_records(
        self, match_processor, sample_images, test_db
    ):
        """Verify log_match creates player records for all participants."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)
        match_processor.log_match(result, start_path, end_path)

        # Check player count
        cursor = test_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM players")
        row = cursor.fetchone()

        assert row['count'] == 10

    def test_log_match_updates_win_loss_stats(
        self, match_processor, sample_images, test_db, ground_truth
    ):
        """Verify log_match updates player win/loss statistics correctly."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)
        match_processor.log_match(result, start_path, end_path)

        # Determine winning team from ground truth scores
        end_sample = next(
            s for s in ground_truth['samples']
            if 'end' in s['filename']
        )
        red_score = end_sample['scores']['red']
        blue_score = end_sample['scores']['blue']
        winning_team = 'blue' if blue_score > red_score else 'red'

        # Check that players on winning team have wins
        cursor = test_db.connection.cursor()
        cursor.execute("""
            SELECT p.char_name, p.global_wins, p.global_losses, mp.team_color
            FROM players p
            JOIN match_participants mp ON p.char_name = mp.char_name
        """)

        for row in cursor.fetchall():
            if row['team_color'] == winning_team:
                assert row['global_wins'] == 1, (
                    f"{row['char_name']} on winning team should have 1 win"
                )
                assert row['global_losses'] == 0
            else:
                assert row['global_wins'] == 0
                assert row['global_losses'] == 1, (
                    f"{row['char_name']} on losing team should have 1 loss"
                )

    def test_winrate_query_after_match_log(
        self, match_processor, sample_images, test_db
    ):
        """Verify get_player_winrate works after logging a match."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)
        match_processor.log_match(result, start_path, end_path)

        # Get a player name from the result
        first_player = result['players'][0]['name']

        # Query winrate
        win_rate, total_matches = test_db.get_player_winrate(first_player)

        assert total_matches == 1
        # Win rate should be either 0 or 100 after one match
        assert win_rate in [0.0, 100.0]

    def test_multiple_match_logs_accumulate_stats(
        self, match_processor, sample_images, test_db
    ):
        """Verify logging the same match twice accumulates stats."""
        start_path, end_path = sample_images

        result = match_processor.process_match(start_path, end_path)

        # Log the match twice
        match_processor.log_match(result, start_path, end_path)
        match_processor.log_match(result, start_path, end_path)

        # Get a player name
        first_player = result['players'][0]['name']

        # Check total matches is 2
        win_rate, total_matches = test_db.get_player_winrate(first_player)

        assert total_matches == 2


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Samples directory missing")
class TestUserDetection:
    """Test user character detection from screenshots."""

    def test_detect_user_from_start_image(
        self, match_processor, sample_images
    ):
        """Verify user detection returns a valid result."""
        start_path, _ = sample_images

        user_name, user_team, confidence = match_processor.detect_user_from_image(
            start_path
        )

        # We may not know who the user is in the sample, but detection should work
        assert user_name is not None, "User name should be detected"
        assert user_team in ['red', 'blue'], "User team should be red or blue"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"

    def test_user_detection_respects_f8_detection(
        self, match_processor, sample_images, ground_truth
    ):
        """Verify process_match uses pre-detected user from F8."""
        start_path, end_path = sample_images

        # Get a known player name from ground truth
        start_sample = next(
            s for s in ground_truth['samples']
            if 'start' in s['filename']
        )
        known_player = start_sample['red_team'][0]  # First red team player

        # Process with pre-detected user
        result = match_processor.process_match(
            start_path,
            end_path,
            detected_user=(known_player, 'red')
        )

        assert result['success'] is True
        assert result['user_character'] == known_player
        assert result['user_team'] == 'red'
        assert result['user_confidence'] == 1.0  # Pre-detected = 100% confidence
