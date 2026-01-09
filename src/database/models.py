"""
Database models and schema for GW2 PvP Tracker.
Handles SQLite database creation and CRUD operations.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager for PvP tracker."""

    def __init__(self, db_path: str = "data/pvp_tracker.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self._connect()
        self._initialize_schema()

    def _connect(self):
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _initialize_schema(self):
        """Create tables if they don't exist."""
        cursor = self.connection.cursor()

        # Create players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                char_name TEXT PRIMARY KEY,
                global_wins INTEGER DEFAULT 0,
                global_losses INTEGER DEFAULT 0,
                total_matches INTEGER DEFAULT 0,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                most_played_profession TEXT,
                CHECK (total_matches = global_wins + global_losses)
            )
        """)

        # Create matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                red_score INTEGER NOT NULL,
                blue_score INTEGER NOT NULL,
                winning_team TEXT NOT NULL CHECK(winning_team IN ('red', 'blue')),
                user_team TEXT NOT NULL CHECK(user_team IN ('red', 'blue')),
                user_char_name TEXT NOT NULL,
                screenshot_start_path TEXT,
                screenshot_end_path TEXT,
                FOREIGN KEY (user_char_name) REFERENCES players(char_name)
            )
        """)

        # Create match_participants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_participants (
                participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                char_name TEXT NOT NULL,
                profession TEXT NOT NULL,
                team_color TEXT NOT NULL CHECK(team_color IN ('red', 'blue')),
                is_user BOOLEAN DEFAULT 0,
                FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
                FOREIGN KEY (char_name) REFERENCES players(char_name),
                UNIQUE(match_id, char_name)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_players_winrate
            ON players(global_wins, total_matches)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_players_last_seen
            ON players(last_seen)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_matches_timestamp
            ON matches(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_matches_user
            ON matches(user_char_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_participants_match
            ON match_participants(match_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_participants_char
            ON match_participants(char_name)
        """)

        # Create views
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS player_winrates AS
            SELECT
                char_name,
                global_wins,
                global_losses,
                total_matches,
                ROUND(CAST(global_wins AS FLOAT) / NULLIF(total_matches, 0) * 100, 1) as win_rate_pct,
                most_played_profession
            FROM players
            WHERE total_matches > 0
        """)

        self.connection.commit()
        logger.info("Database schema initialized")

    def add_player(self, char_name: str) -> bool:
        """
        Add a new player to the database.

        Args:
            char_name: Character name to add

        Returns:
            True if added successfully, False if already exists
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "INSERT INTO players (char_name) VALUES (?)",
                (char_name,)
            )
            self.connection.commit()
            logger.info(f"Added new player: {char_name}")
            return True
        except sqlite3.IntegrityError:
            logger.debug(f"Player already exists: {char_name}")
            return False

    def get_player(self, char_name: str) -> Optional[Dict]:
        """
        Get player statistics.

        Args:
            char_name: Character name to lookup

        Returns:
            Dictionary with player stats or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM players WHERE char_name = ?",
            (char_name,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_player_names(self) -> List[str]:
        """Get all known player names for fuzzy matching."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT char_name FROM players")
        return [row[0] for row in cursor.fetchall()]

    def update_player_stats(
        self,
        char_name: str,
        won: bool,
        profession: Optional[str] = None
    ):
        """
        Update player statistics after a match.

        Args:
            char_name: Character name
            won: True if player won, False if lost
            profession: Profession played (optional)
        """
        cursor = self.connection.cursor()

        # Ensure player exists
        self.add_player(char_name)

        # Update stats
        if won:
            cursor.execute("""
                UPDATE players
                SET global_wins = global_wins + 1,
                    total_matches = total_matches + 1,
                    last_seen = CURRENT_TIMESTAMP
                WHERE char_name = ?
            """, (char_name,))
        else:
            cursor.execute("""
                UPDATE players
                SET global_losses = global_losses + 1,
                    total_matches = total_matches + 1,
                    last_seen = CURRENT_TIMESTAMP
                WHERE char_name = ?
            """, (char_name,))

        # Update most played profession if provided
        if profession:
            self._update_most_played_profession(char_name)

        self.connection.commit()
        logger.debug(f"Updated stats for {char_name}: won={won}")

    def _update_most_played_profession(self, char_name: str):
        """Update the most played profession for a player."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT profession, COUNT(*) as count
            FROM match_participants
            WHERE char_name = ?
            GROUP BY profession
            ORDER BY count DESC
            LIMIT 1
        """, (char_name,))

        row = cursor.fetchone()
        if row:
            most_played = row[0]
            cursor.execute("""
                UPDATE players
                SET most_played_profession = ?
                WHERE char_name = ?
            """, (most_played, char_name))

    def log_match(
        self,
        red_score: int,
        blue_score: int,
        user_team: str,
        user_char_name: str,
        players: List[Dict[str, str]],
        screenshot_start_path: Optional[str] = None,
        screenshot_end_path: Optional[str] = None
    ) -> int:
        """
        Log a complete match to the database.

        Args:
            red_score: Red team score
            blue_score: Blue team score
            user_team: 'red' or 'blue'
            user_char_name: User's character name
            players: List of dicts with keys: name, profession, team
            screenshot_start_path: Path to start screenshot
            screenshot_end_path: Path to end screenshot

        Returns:
            Match ID of the inserted match
        """
        cursor = self.connection.cursor()

        # Determine winner
        winning_team = 'blue' if blue_score > red_score else 'red'

        # Insert match
        cursor.execute("""
            INSERT INTO matches (
                red_score, blue_score, winning_team, user_team, user_char_name,
                screenshot_start_path, screenshot_end_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            red_score, blue_score, winning_team, user_team, user_char_name,
            screenshot_start_path, screenshot_end_path
        ))

        match_id = cursor.lastrowid

        # Insert participants
        for player in players:
            cursor.execute("""
                INSERT INTO match_participants (
                    match_id, char_name, profession, team_color, is_user
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                match_id,
                player['name'],
                player['profession'],
                player['team'],
                1 if player['name'] == user_char_name else 0
            ))

        # Update all player stats
        for player in players:
            player_won = (player['team'] == winning_team)
            self.update_player_stats(
                player['name'],
                player_won,
                player['profession']
            )

        self.connection.commit()
        logger.info(f"Logged match {match_id}: {blue_score}-{red_score}")
        return match_id

    def get_player_winrate(self, char_name: str) -> Tuple[float, int]:
        """
        Get player win rate and total matches.

        Returns:
            Tuple of (win_rate_percentage, total_matches)
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT win_rate_pct, total_matches
            FROM player_winrates
            WHERE char_name = ?
        """, (char_name,))

        row = cursor.fetchone()
        if row:
            return (row[0] or 0.0, row[1])
        return (0.0, 0)

    def get_recent_matches(self, limit: int = 10) -> List[Dict]:
        """Get recent match history."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                match_id,
                timestamp,
                red_score,
                blue_score,
                winning_team,
                user_team,
                user_char_name
            FROM matches
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
