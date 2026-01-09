"""
Manual match entry tool - Enter match data from screenshots to test database.

Looking at your screenshots, I can see:

BEGINNING OF MATCH (roster - beginning.PNG):
Red Team:
- Dragon's Despair (Necromancer)
- Legendary L O K I (Guardian)
- Bound By Familiars (Necromancer)
- Optimum Panda (Necromancer)
- Emperor Leaf Face (Necromancer)
- Qish Baern (Necromancer)

Blue Team:
- Terminal Gearning (Ranger)
- Bombombaumn (Elementalist)
- Iro (Warrior)
- Doneae (Ranger)
- Nemosys (Engineer)
- HongKongHero (Thief)

END OF MATCH (roster - end.PNG):
Final Score: Red 308 - Blue 500 (Blue Wins!)
"""

import sys
from pathlib import Path

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


def enter_match_data():
    """
    Enter match data manually based on what we can see in the screenshots.
    """
    display_section("GW2 PvP Tracker - Manual Match Entry")

    print("\nBased on your screenshots, I can see:")
    print("\nMatch Start:")
    print("  Red Team: Dragon's Despair, Legendary L O K I, Bound By Familiars,")
    print("            Optimum Panda, Emperor Leaf Face, Qish Baern")
    print("  Blue Team: Terminal Gearning, Bombombaumn, Iro, Doneae,")
    print("             Nemosys, HongKongHero")
    print("\nMatch End:")
    print("  Final Score: Red 308 - Blue 500")
    print("  Winner: Blue Team")

    # Define match data from screenshots
    match_data = {
        "red_score": 308,
        "blue_score": 500,
        "user_team": "red",  # Assuming you're on Red team (we can change this)
        "user_char_name": "Dragon's Despair",  # Assuming you're the first player
        "players": [
            # Red Team (rows 1-6)
            {"name": "Dragon's Despair", "profession": "Necromancer", "team": "red"},
            {"name": "Legendary L O K I", "profession": "Guardian", "team": "red"},
            {"name": "Bound By Familiars", "profession": "Necromancer", "team": "red"},
            {"name": "Optimum Panda", "profession": "Necromancer", "team": "red"},
            {"name": "Emperor Leaf Face", "profession": "Necromancer", "team": "red"},
            {"name": "Qish Baern", "profession": "Necromancer", "team": "red"},
            # Blue Team (rows 7-12)
            {"name": "Terminal Gearning", "profession": "Ranger", "team": "blue"},
            {"name": "Bombombaumn", "profession": "Elementalist", "team": "blue"},
            {"name": "Iro", "profession": "Warrior", "team": "blue"},
            {"name": "Doneae", "profession": "Ranger", "team": "blue"},
            {"name": "Nemosys", "profession": "Engineer", "team": "blue"},
            {"name": "HongKongHero", "profession": "Thief", "team": "blue"},
        ]
    }

    print("\n\nWhich team were you on?")
    print("  1. Red Team (lost 308-500)")
    print("  2. Blue Team (won 500-308)")
    user_team_choice = input("\nEnter 1 or 2 (or press Enter for Red): ").strip()

    if user_team_choice == "2":
        match_data["user_team"] = "blue"
        match_data["user_char_name"] = "Terminal Gearning"  # First blue player
        print(f"Setting user to: {match_data['user_char_name']} (Blue Team)")
    else:
        print(f"Setting user to: {match_data['user_char_name']} (Red Team)")

    display_section("Logging Match to Database")

    # Initialize database
    db = Database("data/pvp_tracker.db")  # Use real database, not test
    print("Database initialized")

    # Log the match
    match_id = db.log_match(
        red_score=match_data["red_score"],
        blue_score=match_data["blue_score"],
        user_team=match_data["user_team"],
        user_char_name=match_data["user_char_name"],
        players=match_data["players"],
        screenshot_start_path="roster - beginning.PNG",
        screenshot_end_path="roster - end.PNG"
    )

    print(f"\nMatch #{match_id} logged successfully!")
    print(f"  Score: Red {match_data['red_score']} - Blue {match_data['blue_score']}")
    print(f"  Winner: {'Blue' if match_data['blue_score'] > match_data['red_score'] else 'Red'} Team")
    print(f"  Your Result: {'WIN' if (match_data['user_team'] == 'blue') else 'LOSS'}")

    display_section("Player Statistics")

    # Display stats for all players
    print("\nUpdated player records:\n")
    print(f"{'Player Name':<25} {'W-L':<10} {'Matches':<10} {'Win Rate':<10} {'Main Prof'}")
    print("-" * 70)

    for player in match_data["players"]:
        player_name = player["name"]
        stats = db.get_player(player_name)
        win_rate, total = db.get_player_winrate(player_name)

        w = stats['global_wins']
        l = stats['global_losses']
        prof = stats['most_played_profession'] or player['profession']

        print(f"{player_name:<25} {w}-{l:<8} {total:<10} {win_rate:>5.1f}%    {prof}")

    display_section("Your Match History")

    # Show user's recent matches
    recent = db.get_recent_matches(limit=5)
    print(f"\nRecent matches (showing last {len(recent)}):\n")

    for match in recent:
        timestamp = match['timestamp']
        red = match['red_score']
        blue = match['blue_score']
        winner = match['winning_team']
        user_team = match['user_team']
        result = "WIN" if winner == user_team else "LOSS"

        print(f"  [{timestamp}] Red {red} - Blue {blue} -> {result}")

    db.close()
    print("\nDatabase connection closed")

    display_section("Summary")
    print("\nYour match data has been successfully logged to the database!")
    print(f"\nDatabase location: data/pvp_tracker.db")
    print("\nYou can now:")
    print("  1. Run this script again to add more matches")
    print("  2. Use any SQLite browser to explore the database")
    print("  3. Query player statistics programmatically")
    print("\nNext steps:")
    print("  - Install Tesseract OCR to enable automatic text extraction")
    print("  - Implement computer vision for automatic scoreboard detection")


if __name__ == "__main__":
    enter_match_data()
