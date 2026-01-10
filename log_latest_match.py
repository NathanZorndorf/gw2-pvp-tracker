"""
Log the most recent captured match to the database.

This script will:
1. Find the most recent match_start and match_end screenshots
2. Let you manually enter the match data
3. Log it to the database
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.models import Database
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def display_section(title):
    """Display formatted section."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def find_latest_match_screenshots():
    """Find the most recent match_start and match_end screenshots."""
    screenshots_dir = Path("screenshots")

    # Find all match screenshots
    start_files = sorted(screenshots_dir.glob("match_start_*.png"), reverse=True)
    end_files = sorted(screenshots_dir.glob("match_end_*.png"), reverse=True)

    if not start_files or not end_files:
        return None, None

    return start_files[0], end_files[0]


def get_player_input(prompt, default=""):
    """Get player name input with default."""
    value = input(prompt).strip()
    return value if value else default


def log_match_interactive():
    """Interactive match logging."""
    display_section("GW2 PvP Tracker - Log Latest Match")

    # Find latest screenshots
    start_path, end_path = find_latest_match_screenshots()

    if not start_path or not end_path:
        print("\nNo match screenshots found!")
        print("Make sure you've captured screenshots with:")
        print("  - F8 at match start")
        print("  - F9 at match end")
        return

    print(f"\nFound screenshots:")
    print(f"  Start: {start_path.name}")
    print(f"  End:   {end_path.name}")

    # Get match scores
    display_section("Match Scores")
    print("\nEnter the final scores from your match:")

    try:
        red_score = int(input("Red team score (0-500): ").strip())
        blue_score = int(input("Blue team score (0-500): ").strip())

        if not (0 <= red_score <= 500 and 0 <= blue_score <= 500):
            print("Invalid scores! Must be between 0 and 500.")
            return

    except ValueError:
        print("Invalid input! Scores must be numbers.")
        return

    winner = "blue" if blue_score > red_score else "red"
    print(f"\nWinner: {winner.upper()} team ({blue_score if winner == 'blue' else red_score} points)")

    # Get user team
    display_section("Your Team")
    print("\nWhich team were you on?")
    print(f"  1. Red Team (Score: {red_score}) {'<-- LOST' if winner == 'blue' else '<-- WON'}")
    print(f"  2. Blue Team (Score: {blue_score}) {'<-- LOST' if winner == 'red' else '<-- WON'}")

    user_team_choice = input("\nEnter 1 or 2: ").strip()
    user_team = "red" if user_team_choice == "1" else "blue"

    # Get user character name
    user_char = input(f"\nYour character name (on {user_team.upper()} team): ").strip()

    if not user_char:
        print("Character name is required!")
        return

    # Get all 10 player names and professions
    display_section("Player Names")
    print("\nEnter the player names and professions for all 10 players.")
    print("Format: Just the name (we'll ask for profession next)")
    print("\nRed Team (5 players):")

    players = []

    # Red team
    for i in range(1, 6):
        name = input(f"  Red {i}: ").strip()
        if name:
            profession = input(f"    Profession: ").strip() or "Unknown"
            players.append({"name": name, "profession": profession, "team": "red"})

    print("\nBlue Team (5 players):")

    # Blue team
    for i in range(1, 6):
        name = input(f"  Blue {i}: ").strip()
        if name:
            profession = input(f"    Profession: ").strip() or "Unknown"
            players.append({"name": name, "profession": profession, "team": "blue"})

    if len(players) != 10:
        print(f"\nWarning: Only {len(players)} players entered (expected 10)")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return

    # Confirm data
    display_section("Confirm Match Data")
    print(f"\nScore: Red {red_score} - Blue {blue_score}")
    print(f"Winner: {winner.upper()} team")
    print(f"You: {user_char} ({user_team.upper()} team)")
    print(f"Your Result: {'WIN' if user_team == winner else 'LOSS'}")
    print(f"\nPlayers: {len(players)}")

    confirm = input("\nLog this match? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Match not logged.")
        return

    # Log to database
    display_section("Logging Match")

    db = Database("data/pvp_tracker.db")

    match_id = db.log_match(
        red_score=red_score,
        blue_score=blue_score,
        user_team=user_team,
        user_char_name=user_char,
        players=players,
        screenshot_start_path=str(start_path),
        screenshot_end_path=str(end_path)
    )

    print(f"\nMatch #{match_id} logged successfully!")
    print(f"  Your Result: {'WIN' if user_team == winner else 'LOSS'}")

    # Show updated stats
    display_section("Your Stats")

    player_stats = db.get_player(user_char)
    win_rate, total = db.get_player_winrate(user_char)

    print(f"\nCharacter: {user_char}")
    print(f"Record: {player_stats['global_wins']}W - {player_stats['global_losses']}L")
    print(f"Total Matches: {total}")
    print(f"Win Rate: {win_rate:.1f}%")

    db.close()

    print("\nMatch logged! Run 'python view_stats.py' to see detailed statistics.")


if __name__ == "__main__":
    try:
        log_match_interactive()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
