"""Profile MatchProcessor arena detection and player extraction.

Run: python scripts/diagnostics/profile_match_processor.py
"""
import cProfile
import pstats
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(project_root / "src"))

from config import Config
from database.models import Database
from automation.match_processor import MatchProcessor
import cv2


def main():
    config = Config()
    db = Database("data/pvp_tracker.db")
    processor = MatchProcessor(config, db)

    image_path = project_root / "data" / "samples" / "ranked-1" / "match_start_20260110_102550_full.png"
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    profiler = cProfile.Profile()
    profiler.enable()

    # Profile arena detection and full player extraction
    processor.detect_arena_type(image)
    processor._extract_all_players(image, "start")

    profiler.disable()

    stats_path = project_root / "logs" / "match_processor_profile.prof"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(stats_path))

    ps = pstats.Stats(profiler).sort_stats("cumulative")
    print("Top 30 cumulative time functions:")
    ps.print_stats(30)

    print(f"Profile saved to: {stats_path}")


if __name__ == "__main__":
    main()
