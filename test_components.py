"""
Test script to validate core components:
- Database operations
- Screenshot capture
- OCR engine

Run this script to verify the setup is working correctly.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.models import Database
from vision.capture import ScreenCapture
from vision.ocr_engine import OCREngine
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_config():
    """Test configuration loading."""
    print_section("Testing Configuration")

    config = Config()
    print(f"OK Config loaded")
    print(f"  Database path: {config.get('database.path')}")
    print(f"  Screenshots dir: {config.get('paths.screenshots')}")
    print(f"  Tesseract path: {config.get('ocr.tesseract_path')}")


def test_database():
    """Test database operations."""
    print_section("Testing Database")

    # Use test database
    db = Database("data/test_pvp_tracker.db")
    print("OK Database initialized")

    # Test adding players
    db.add_player("TestPlayer.1234")
    db.add_player("Opponent.5678")
    print("OK Added test players")

    # Test retrieving player
    player = db.get_player("TestPlayer.1234")
    print(f"OK Retrieved player: {player['char_name']}")
    print(f"  Stats: {player['global_wins']}W - {player['global_losses']}L")

    # Test logging a match
    test_players = [
        {"name": "Blue1", "profession": "Guardian", "team": "blue"},
        {"name": "Blue2", "profession": "Thief", "team": "blue"},
        {"name": "Blue3", "profession": "Warrior", "team": "blue"},
        {"name": "Blue4", "profession": "Ranger", "team": "blue"},
        {"name": "TestPlayer.1234", "profession": "Necromancer", "team": "blue"},
        {"name": "Red1", "profession": "Revenant", "team": "red"},
        {"name": "Red2", "profession": "Elementalist", "team": "red"},
        {"name": "Red3", "profession": "Mesmer", "team": "red"},
        {"name": "Red4", "profession": "Engineer", "team": "red"},
        {"name": "Opponent.5678", "profession": "Guardian", "team": "red"},
    ]

    match_id = db.log_match(
        red_score=487,
        blue_score=500,
        user_team="blue",
        user_char_name="TestPlayer.1234",
        players=test_players
    )
    print(f"OK Logged match #{match_id}: Blue 500 - Red 487")

    # Test player stats update
    updated_player = db.get_player("TestPlayer.1234")
    print(f"OK Player stats updated:")
    print(f"  Record: {updated_player['global_wins']}W - {updated_player['global_losses']}L")
    print(f"  Total matches: {updated_player['total_matches']}")

    # Test win rate query
    win_rate, total = db.get_player_winrate("TestPlayer.1234")
    print(f"OK Win rate: {win_rate}% ({total} matches)")

    # Test recent matches
    recent = db.get_recent_matches(limit=5)
    print(f"OK Retrieved {len(recent)} recent matches")

    db.close()
    print("OK Database connection closed")


def test_screen_capture():
    """Test screenshot capture."""
    print_section("Testing Screen Capture")

    capture = ScreenCapture()
    print("OK Screen capture initialized")

    # Get monitor info
    monitors = capture.get_monitor_info()
    print(f"OK Detected {len(monitors)} monitor(s):")
    for mon in monitors:
        print(f"  Monitor {mon['index']}: {mon['width']}x{mon['height']}")

    # Capture full screen
    print("\nCapturing full screen in 3 seconds...")
    print("(Make sure GW2 or any window is visible)")
    import time
    time.sleep(3)

    try:
        full_path = capture.capture_and_save_full("test")
        print(f"OK Full screen captured: {full_path}")

        # Capture a small region (center of screen)
        mon = monitors[0]
        center_x = mon['width'] // 2 - 200
        center_y = mon['height'] // 2 - 100
        region_path = capture.capture_and_save_region(
            center_x, center_y, 400, 200, "test"
        )
        print(f"OK Region captured: {region_path}")

    except Exception as e:
        print(f"ERROR Capture failed: {e}")


def test_ocr_engine():
    """Test OCR engine."""
    print_section("Testing OCR Engine")

    config = Config()
    tesseract_path = config.get('ocr.tesseract_path')

    try:
        ocr = OCREngine(tesseract_path=tesseract_path)
        print("OK OCR engine initialized")

        # Create test image with text
        import numpy as np
        import cv2

        # Create white image with black text
        img = np.ones((60, 300, 3), dtype=np.uint8) * 255
        cv2.putText(
            img, "TestPlayer.1234", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )

        # Test text extraction
        text = ocr.extract_text(img, preprocess=False)
        print(f"OK OCR extracted: '{text}'")

        # Create test image with number
        score_img = np.ones((60, 100, 3), dtype=np.uint8) * 255
        cv2.putText(
            score_img, "500", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2
        )

        # Test score extraction
        score = ocr.extract_score(score_img)
        print(f"OK Score extracted: {score}")

        # Test fuzzy matching
        known_names = ["TestPlayer.1234", "SomeOtherPlayer.9999"]
        matched = ocr.extract_player_name(img, known_names=known_names)
        print(f"OK Fuzzy matched name: '{matched}'")

    except Exception as e:
        print(f"ERROR OCR test failed: {e}")
        print(f"\nNote: Make sure Tesseract is installed:")
        print("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  2. Install to: C:\\Program Files\\Tesseract-OCR\\")
        print("  3. Or update config.yaml with correct path")


def run_all_tests():
    """Run all component tests."""
    print("\n" + "=" * 60)
    print("  GW2 PvP Tracker - Component Tests")
    print("=" * 60)

    try:
        test_config()
        test_database()
        test_screen_capture()
        test_ocr_engine()

        print("\n" + "=" * 60)
        print("  All tests completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review captured screenshots in: screenshots/")
        print("  2. Check test database in: data/test_pvp_tracker.db")
        print("  3. Ready to implement template matching and user detection")

    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
