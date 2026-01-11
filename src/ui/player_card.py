"""
Player card widget for displaying individual player stats.
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Optional

from .styles import (
    COLORS, FONTS, ROW_HEIGHT, PADDING_INNER,
    get_winrate_color, get_star_count, SYMBOLS
)
from .confidence import (
    get_stars_display, format_winrate, format_matches
)


@dataclass
class PlayerStats:
    """Data class for player statistics."""
    name: str
    team: str               # "red" or "blue"
    win_rate: float         # 0.0 - 100.0
    total_matches: int
    is_user: bool = False   # Highlight user's row


class PlayerCard(tk.Frame):
    """Widget displaying a single player's stats row."""

    def __init__(self, parent, player: PlayerStats, **kwargs):
        """
        Initialize player card.

        Args:
            parent: Parent tkinter widget
            player: PlayerStats dataclass with player info
        """
        # Determine background color
        if player.is_user:
            bg_color = COLORS['bg_row_user']
        elif player.team == 'red':
            bg_color = COLORS['bg_row_red']
        else:
            bg_color = COLORS['bg_row_blue']

        super().__init__(parent, bg=bg_color, height=ROW_HEIGHT, **kwargs)
        self.pack_propagate(False)

        self.player = player
        self._create_widgets(bg_color)

    def _create_widgets(self, bg_color: str):
        """Create the player card widgets."""
        player = self.player

        # Team indicator
        team_color = COLORS['team_red'] if player.team == 'red' else COLORS['team_blue']
        team_label = tk.Label(
            self,
            text=f"[{player.team[0].upper()}]",
            font=FONTS['small'],
            fg=team_color,
            bg=bg_color,
            width=3
        )
        team_label.pack(side=tk.LEFT, padx=(PADDING_INNER, 2))

        # Stars display
        stars = get_stars_display(player.win_rate, player.total_matches)
        stars_label = tk.Label(
            self,
            text=stars,
            font=FONTS['stars'],
            fg=COLORS['star_filled'] if player.total_matches > 0 else COLORS['text_secondary'],
            bg=bg_color,
            width=6
        )
        stars_label.pack(side=tk.LEFT, padx=2)

        # Win rate percentage
        winrate_text = format_winrate(player.win_rate, player.total_matches)
        winrate_color = (
            get_winrate_color(player.win_rate)
            if player.total_matches > 0
            else COLORS['winrate_new']
        )
        winrate_label = tk.Label(
            self,
            text=winrate_text,
            font=FONTS['stats'],
            fg=winrate_color,
            bg=bg_color,
            width=6,
            anchor='e'
        )
        winrate_label.pack(side=tk.LEFT, padx=4)

        # Match count (confidence)
        matches_text = format_matches(player.total_matches)
        matches_label = tk.Label(
            self,
            text=f"({matches_text})",
            font=FONTS['small'],
            fg=COLORS['text_secondary'],
            bg=bg_color,
            width=5,
            anchor='w'
        )
        matches_label.pack(side=tk.LEFT, padx=2)

        # Player name (takes remaining space)
        name_font = FONTS['player_name_user'] if player.is_user else FONTS['player_name']
        name_color = COLORS['text_header'] if player.is_user else COLORS['text_primary']

        # Truncate long names
        display_name = player.name
        if len(display_name) > 20:
            display_name = display_name[:18] + "..."

        name_label = tk.Label(
            self,
            text=display_name,
            font=name_font,
            fg=name_color,
            bg=bg_color,
            anchor='w'
        )
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=PADDING_INNER)
