"""
View player statistics and match history from the database.
"""

import sys
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from database.models import Database


def display_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def view_all_players():
    """Display all players sorted by win rate."""
    display_section("All Players (Sorted by Win Rate)")

    db = Database("data/pvp_tracker.db")

    cursor = db.connection.cursor()
    cursor.execute("""
        SELECT char_name, global_wins, global_losses, total_matches,
               win_rate_pct, most_played_profession
        FROM player_winrates
        ORDER BY win_rate_pct DESC, total_matches DESC
    """)

    players = cursor.fetchall()

    if not players:
        print("\nNo players found in database")
        return

    print(f"\nTotal players tracked: {len(players)}\n")
    print(f"{'Player Name':<30} {'Record':<12} {'Win Rate':<12} {'Profession'}")
    print("-" * 70)

    for player in players:
        name = player['char_name']
        wins = player['global_wins']
        losses = player['global_losses']
        winrate = player['win_rate_pct'] or 0.0
        prof = player['most_played_profession'] or "Unknown"

        print(f"{name:<30} {wins}-{losses:<10} {winrate:>5.1f}%       {prof}")

    db.close()


def view_match_history():
    """Display match history."""
    display_section("Match History")

    db = Database("data/pvp_tracker.db")

    matches = db.get_recent_matches(limit=50)

    if not matches:
        print("\nNo matches found in database")
        return

    print(f"\nShowing last {len(matches)} matches:\n")

    for match in matches:
        match_id = match['match_id']
        timestamp = match['timestamp']
        red = match['red_score']
        blue = match['blue_score']
        winner = match['winning_team'].upper()
        user_team = match['user_team']
        user_name = match['user_char_name']
        result = "WIN" if winner.lower() == user_team else "LOSS"

        print(f"Match #{match_id} [{timestamp}]")
        print(f"  Score: Red {red} - Blue {blue} ({winner} wins)")
        print(f"  You: {user_name} ({user_team.upper()} team) -> {result}\n")

    db.close()


def search_player(player_name):
    """Search for a specific player and show detailed stats."""
    display_section(f"Player Details: {player_name}")

    db = Database("data/pvp_tracker.db")

    player = db.get_player(player_name)

    if not player:
        print(f"\nPlayer '{player_name}' not found in database")
        print("\nAvailable players:")
        all_names = db.get_all_player_names()
        for name in sorted(all_names):
            print(f"  - {name}")
        db.close()
        return

    # Display basic stats
    print(f"\nPlayer: {player['char_name']}")
    print(f"Record: {player['global_wins']}W - {player['global_losses']}L")
    print(f"Total Matches: {player['total_matches']}")

    win_rate, _ = db.get_player_winrate(player_name)
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Most Played: {player['most_played_profession'] or 'Unknown'}")
    print(f"First Seen: {player['first_seen']}")
    print(f"Last Seen: {player['last_seen']}")

    # Get match history for this player
    cursor = db.connection.cursor()
    cursor.execute("""
        SELECT m.match_id, m.timestamp, m.red_score, m.blue_score,
               m.winning_team, mp.team_color, mp.profession
        FROM matches m
        JOIN match_participants mp ON m.match_id = mp.match_id
        WHERE mp.char_name = ?
        ORDER BY m.timestamp DESC
        LIMIT 10
    """, (player_name,))

    matches = cursor.fetchall()

    if matches:
        print(f"\nRecent Matches ({len(matches)}):")
        for match in matches:
            match_id = match['match_id']
            timestamp = match['timestamp']
            red = match['red_score']
            blue = match['blue_score']
            winner = match['winning_team']
            team = match['team_color']
            prof = match['profession']
            result = "WIN" if winner == team else "LOSS"

            print(f"  #{match_id} [{timestamp}] - {prof}")
            print(f"    Red {red} - Blue {blue} ({team.upper()} team) -> {result}")

    db.close()


def main():
    """Main menu."""
    print("\n" + "=" * 70)
    print("  GW2 PvP Tracker - Stats Viewer")
    print("=" * 70)

    print("\nWhat would you like to view?")
    print("  1. All players (sorted by win rate)")
    print("  2. Match history")
    print("  3. Search for specific player")
    print("  4. All of the above")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        view_all_players()
    elif choice == "2":
        view_match_history()
    elif choice == "3":
        player_name = input("\nEnter player name: ").strip()
        if player_name:
            search_player(player_name)
    elif choice == "4":
        view_all_players()
        view_match_history()
        print("\n" + "=" * 70)
        print("  To search for a player, run: python view_stats.py")
        print("=" * 70)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
