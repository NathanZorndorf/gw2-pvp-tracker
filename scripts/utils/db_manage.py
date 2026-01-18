"""
Database management utilities for GW2 PvP Tracker.

Commands:
    python scripts/utils/db_manage.py clear          - Clear all data
    python scripts/utils/db_manage.py import <dir>   - Import match from screenshot folder
    python scripts/utils/db_manage.py import-all     - Import all matches from samples and screenshots
    python scripts/utils/db_manage.py status         - Show database statistics
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from database.models import Database
from automation.match_processor import MatchProcessor
from config import Config


def clear_database(db_path: str = "data/pvp_tracker.db"):
    """Clear all data from the database."""
    print(f"Clearing database: {db_path}")

    db = Database(db_path)
    cursor = db.connection.cursor()

    # Get counts before clearing
    cursor.execute("SELECT COUNT(*) FROM matches")
    match_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]

    # Clear tables in order (respecting foreign keys)
    cursor.execute("DELETE FROM match_participants")
    cursor.execute("DELETE FROM matches")
    cursor.execute("DELETE FROM players")
    
    # Reset auto-increment sequences so match IDs start back at #1
    cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('matches', 'match_participants')")
    
    db.connection.commit()

    print(f"  Deleted {match_count} matches")
    print(f"  Deleted {player_count} players")
    print("Database cleared, and Match IDs reset!")

    db.close()


def import_match(folder_path: str, db_path: str = "data/pvp_tracker.db"):
    """Import a match from a screenshot folder."""
    folder = Path(folder_path)

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return False

    # Find start and end screenshots
    start_files = list(folder.glob("match_start_*.png"))
    end_files = list(folder.glob("match_end_*.png"))

    if not start_files:
        print(f"ERROR: No match_start_*.png found in {folder}")
        return False
    if not end_files:
        print(f"ERROR: No match_end_*.png found in {folder}")
        return False

    start_path = str(start_files[0])
    end_path = str(end_files[0])

    print(f"Importing match from: {folder.name}")
    print(f"  Start: {Path(start_path).name}")
    print(f"  End: {Path(end_path).name}")

    # Initialize processor
    config = Config()
    db = Database(db_path)
    processor = MatchProcessor(config, db)

    try:
        # First detect user from start screenshot (cleaner, has cyan highlight)
        user_char, user_team, confidence = processor.detect_user_from_image(start_path)
        detected_user = (user_char, user_team) if user_char else None

        # Process the match with pre-detected user
        match_data = processor.process_match(start_path, end_path, detected_user=detected_user)

        if not match_data['success']:
            print(f"  ERROR: {match_data.get('error', 'Unknown error')}")
            db.close()
            return False

        # Display what we found
        print(f"  Scores: Red {match_data['red_score']} - Blue {match_data['blue_score']}")
        print(f"  Arena: {match_data.get('arena_type', 'unknown')}")
        print(f"  User: {match_data['user_character']} ({match_data['user_team']})")

        if match_data.get('validation_errors'):
            print(f"  Warnings: {', '.join(match_data['validation_errors'])}")

        # Log to database
        match_id = processor.log_match(match_data, start_path, end_path)
        print(f"  -> Logged as Match #{match_id}")

        db.close()
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        db.close()
        return False


def import_all_available(db_path: str = "data/pvp_tracker.db"):
    """Import all matches from both data/samples/ and screenshots/ folders."""
    locations = ["data/samples", "screenshots"]
    
    found_any = False
    for loc in locations:
        path = Path(loc)
        if path.exists():
            print(f"--- Processing {loc} ---")
            bulk_import(str(path), db_path)
            found_any = True
            print()
            
    if not found_any:
        print("Neither data/samples/ nor screenshots/ directories found.")


def bulk_import(root_path: str, db_path: str = "data/pvp_tracker.db"):
    """Import all matches from any subdirectories in root_path."""
    root = Path(root_path)

    if not root.is_dir():
        print(f"ERROR: Not a directory: {root}")
        return

    # Find all directories that contain both a match_start and match_end
    match_folders = []
    for folder in root.iterdir():
        if folder.is_dir():
            start_files = list(folder.glob("match_start_*.png"))
            end_files = list(folder.glob("match_end_*.png"))
            if start_files and end_files:
                match_folders.append(folder)

    if not match_folders:
        print(f"No match folders (containing start/end screenshots) found in {root}")
        return

    print(f"Found {len(match_folders)} folders to import in {root}\n")

    success_count = 0
    for folder in match_folders:
        if import_match(str(folder), db_path):
            success_count += 1
        print()

    print(f"Imported {success_count}/{len(match_folders)} matches from {root}")


def show_status(db_path: str = "data/pvp_tracker.db"):
    """Show database statistics."""
    db = Database(db_path)
    cursor = db.connection.cursor()

    print(f"Database: {db_path}\n")

    # Match count
    cursor.execute("SELECT COUNT(*) FROM matches")
    match_count = cursor.fetchone()[0]
    print(f"Total matches: {match_count}")

    # Player count
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    print(f"Total players: {player_count}")

    # Top players by matches
    if player_count > 0:
        print("\nTop 5 players by matches:")
        cursor.execute("""
            SELECT char_name, total_matches,
                   ROUND(CAST(global_wins AS FLOAT) / NULLIF(total_matches, 0) * 100, 1) as win_rate
            FROM players
            WHERE total_matches > 0
            ORDER BY total_matches DESC
            LIMIT 5
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} matches ({row[2]}% win rate)")

    db.close()


def main():
    parser = argparse.ArgumentParser(description="Database management for GW2 PvP Tracker")
    parser.add_argument("command", choices=["clear", "import", "import-all", "bulk-import", "status", "delete", "fix-score"],
                        help="Command to run")
    parser.add_argument("path", nargs="?", help="Path for import or bulk-import command")
    parser.add_argument("--id", type=int, help="Match ID for delete or fix-score commands")
    parser.add_argument("--red", type=int, help="Red score for fix-score")
    parser.add_argument("--blue", type=int, help="Blue score for fix-score")
    parser.add_argument("--db", default="data/pvp_tracker.db", help="Database path")

    args = parser.parse_args()

    if args.command == "clear":
        confirm = input("Are you sure you want to clear all data? (yes/no): ")
        if confirm.lower() == "yes":
            clear_database(args.db)
        else:
            print("Cancelled")

    elif args.command == "import":
        if not args.path:
            print("ERROR: Please specify a folder path")
            print("Usage: python db_manage.py import <folder_path>")
            sys.exit(1)
        import_match(args.path, args.db)

    elif args.command == "delete":
        if not args.id:
            print("ERROR: Please specify --id <match_id> to delete")
            print("Usage: python db_manage.py delete --id 29")
            sys.exit(1)
        db = Database(args.db)
        try:
            ok = db.delete_match(args.id)
            if ok:
                print(f"Deleted match #{args.id}")
            else:
                print(f"Match #{args.id} not found")
        finally:
            db.close()

    elif args.command == "fix-score":
        if not args.id or args.red is None or args.blue is None:
            print("ERROR: Please specify --id, --red and --blue for fix-score")
            print("Usage: python db_manage.py fix-score --id 29 --red 544 --blue 173")
            sys.exit(1)
        db = Database(args.db)
        try:
            ok = db.update_match_scores(args.id, args.red, args.blue)
            if ok:
                print(f"Updated scores for match #{args.id} -> Red {args.red} - Blue {args.blue}")
            else:
                print(f"Match #{args.id} not found")
        finally:
            db.close()

    elif args.command == "import-all":
        import_all_available(args.db)

    elif args.command == "bulk-import":
        if not args.path:
            print("ERROR: Please specify a root folder path")
            print("Usage: python db_manage.py bulk-import <root_path>")
            sys.exit(1)
        bulk_import(args.path, args.db)

    elif args.command == "status":
        show_status(args.db)


if __name__ == "__main__":
    main()
