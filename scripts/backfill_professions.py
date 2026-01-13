"""
Backfill profession data for existing matches by re-processing saved screenshots.

This script:
1. Queries all matches that have saved screenshots
2. Re-runs profession detection on match_start screenshots
3. Updates match_participants with detected professions
4. Recalculates most_played_profession for all affected players
"""

import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.profession_detector import ProfessionDetector
from src.database.models import Database
from config import Config


def main():
    """Main execution."""
    print("=" * 60)
    print("Backfilling Profession Data from Screenshots")
    print("=" * 60)

    # Initialize components
    config_path = project_root / 'config.yaml'
    db_path = project_root / 'data' / 'pvp_tracker.db'

    print(f"\nLoading config from {config_path}")
    config = Config(str(config_path))

    print(f"Opening database: {db_path}")
    db = Database(str(db_path))

    print("Initializing profession detector...")
    detector = ProfessionDetector(config)

    if not detector.enabled:
        print("ERROR: Profession detection is disabled in config")
        return

    # Get all matches with screenshots
    matches = db.get_matches_with_screenshots()
    print(f"\nFound {len(matches)} matches with screenshots")

    if not matches:
        print("No matches to process")
        return

    # Process each match
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    updated_players = set()

    for i, match in enumerate(matches, start=1):
        match_id = match['match_id']
        screenshot_path = match['screenshot_start_path']
        arena_type = match.get('arena_type', 'ranked')

        print(f"\n[{i}/{len(matches)}] Match #{match_id}")
        print(f"  Screenshot: {screenshot_path}")
        print(f"  Arena: {arena_type}")

        # Check if screenshot exists
        if not screenshot_path or not Path(screenshot_path).exists():
            print(f"  SKIP: Screenshot not found")
            skipped_count += 1
            continue

        # Load screenshot
        image = cv2.imread(screenshot_path)
        if image is None:
            print(f"  FAIL: Could not load screenshot")
            failed_count += 1
            continue

        # Detect professions
        try:
            results = detector.detect_professions(image, arena_type=arena_type)

            if not results or len(results) != 10:
                print(f"  FAIL: Expected 10 results, got {len(results)}")
                failed_count += 1
                continue

            # Get participants for this match
            participants = db.get_match_participants(match_id)

            if len(participants) != 10:
                print(f"  FAIL: Expected 10 participants, got {len(participants)}")
                failed_count += 1
                continue

            # Sort results by team and index to match participant order
            results_sorted = sorted(results, key=lambda x: (0 if x['team'] == 'red' else 1, x['player_index']))

            # Update each participant
            update_count = 0
            for participant, result in zip(participants, results_sorted):
                participant_id = participant['participant_id']
                char_name = participant['char_name']
                old_profession = participant['profession']
                new_profession = result['profession']
                confidence = result['confidence']

                # Skip if profession didn't change
                if old_profession == new_profession:
                    continue

                # Update profession
                success = db.update_participant_profession(participant_id, new_profession)

                if success:
                    update_count += 1
                    updated_players.add(char_name)
                    print(f"    {char_name}: {old_profession} -> {new_profession} (conf={confidence:.3f})")

            if update_count > 0:
                print(f"  OK: Updated {update_count}/10 professions")
                processed_count += 1
            else:
                print(f"  SKIP: No changes needed")
                skipped_count += 1

        except Exception as e:
            print(f"  FAIL: {e}")
            failed_count += 1

    # Recalculate most_played_profession for all affected players
    if updated_players:
        print(f"\n\nRecalculating most_played_profession for {len(updated_players)} players...")
        for char_name in sorted(updated_players):
            db.recalculate_most_played_profession(char_name)
            print(f"  Recalculated: {char_name}")

    # Summary
    print("\n" + "=" * 60)
    print("Backfill Summary")
    print("=" * 60)
    print(f"Total matches: {len(matches)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"Players updated: {len(updated_players)}")
    print("=" * 60)

    # Close database
    db.close()


if __name__ == '__main__':
    main()
