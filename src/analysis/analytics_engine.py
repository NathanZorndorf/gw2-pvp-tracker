import pandas as pd
import sqlite3
import logging
from database.models import Database
from .win_rate_utils import get_display_win_rate
from config import Config

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    def __init__(self):
        self.db = Database()
        self.config = Config()

    def get_user_professions(self):
        """Get unique list of professions played by the user."""
        query = """
            SELECT DISTINCT profession 
            FROM match_participants 
            WHERE is_user = 1 AND profession IS NOT NULL
            ORDER BY profession
        """
        try:
            # Check if we can execute this query (table exists, etc)
            # Use pandas read_sql_query with the sqlite3 connection
            with self.db.connection:
                 df = pd.read_sql_query(query, self.db.connection)
            
            if not df.empty and 'profession' in df.columns:
                return [p for p in df['profession'].tolist() if p]
            return []
        except Exception as e:
            logger.error(f"Error fetching user professions: {e}")
            return []

    def get_matchup_stats(self, player_profession=None, match_type="All"):
        """
        Calculate win rates against enemy professions.
        
        Args:
            player_profession (str, optional): Filter by user's profession. If None or "All", uses all.
            match_type (str, optional): "Unranked", "Ranked", or "All".
            
        Returns:
            pd.DataFrame: DataFrame with columns ['enemy_profession', 'wins', 'total', 'win_rate']
        """
        
        # 1. Get all matches where user played
        # We need: match_id, user_team, user_profession
        
        match_type_filter = ""
        params = {}
        
        if match_type == "Ranked":
            match_type_filter = "AND m.arena_type = 'ranked'"
        elif match_type == "Unranked":
            match_type_filter = "AND m.arena_type = 'unranked'"
            
        user_matches_query = f"""
            SELECT m.match_id, m.winning_team, mp.team_color as user_team, mp.profession as user_prof
            FROM matches m
            JOIN match_participants mp ON m.match_id = mp.match_id
            WHERE mp.is_user = 1 {match_type_filter}
        """
        
        # 2. Get all enemy participants for those matches
        # We need: match_id, enemy_profession
        
        enemy_participants_query = """
            SELECT mp.match_id, mp.profession as enemy_prof, mp.team_color as enemy_team
            FROM match_participants mp
            WHERE mp.is_user = 0
        """

        try:
            with self.db.connection:
                user_matches_df = pd.read_sql_query(user_matches_query, self.db.connection)
                enemy_participants_df = pd.read_sql_query(enemy_participants_query, self.db.connection)
        except Exception as e:
            logger.error(f"Error fetching data for analytics: {e}")
            return pd.DataFrame()
            
        if user_matches_df.empty or enemy_participants_df.empty:
             return pd.DataFrame(columns=['enemy_profession', 'wins', 'total', 'win_rate'])

        # Filter by player profession if requested
        if player_profession and player_profession != "All":
            user_matches_df = user_matches_df[user_matches_df['user_prof'] == player_profession]
            
        if user_matches_df.empty:
            return pd.DataFrame(columns=['enemy_profession', 'wins', 'total', 'win_rate'])

        # Determine win/loss per match for the user
        # winning_team is 'red' or 'blue'. user_team is 'red' or 'blue'.
        # If winning_team matches user_team, it's a win.
        user_matches_df['is_win'] = user_matches_df['winning_team'] == user_matches_df['user_team']
        
        # Merge enemies with user matches on match_id
        # match_id is the key
        merged_df = pd.merge(user_matches_df, enemy_participants_df, on='match_id')
        
        # Filter for actual enemies (different team color)
        # Just in case user is defined as is_user=0 for some reason, or to filter out teammates
        # Note: teammates also have is_user=0. So we MUST filter by team_color.
        merged_df = merged_df[merged_df['user_team'] != merged_df['enemy_team']]
        
        # Now we have a row for every user match vs every enemy in that match.
        # Filter out empty professions if any
        merged_df = merged_df[merged_df['enemy_prof'].notna() & (merged_df['enemy_prof'] != '')]

        # Group by enemy profession
        stats = merged_df.groupby('enemy_prof').agg(
            wins=('is_win', 'sum'),
            total=('is_win', 'count')
        ).reset_index()
        
        stats.rename(columns={'enemy_prof': 'enemy_profession'}, inplace=True)
        
        # Apply win rate calculation based on config
        self.config = Config() # Reload config in case it changed
        stats['win_rate'] = stats.apply(
            lambda row: get_display_win_rate(row['wins'], row['total'], self.config.data),
            axis=1
        )
        
        return stats
