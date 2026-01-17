"""
Player card widget for displaying individual player stats.
"""

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Optional, Dict, Callable
from PIL import ImageTk

from .styles import (
    COLORS, FONTS, ROW_HEIGHT, PADDING_INNER,
    get_winrate_color, get_star_count, SYMBOLS
)
from .confidence import (
    get_stars_display, format_winrate, format_matches
)
from .profession_selector import ProfessionSelectorPopup


@dataclass
class PlayerStats:
    """Data class for player statistics."""
    name: str
    profession: str         # Profession name
    team: str               # "red" or "blue"
    win_rate: float         # 0.0 - 100.0
    total_matches: int
    is_user: bool = False   # Highlight user's row
    index: int = -1         # Index in the players list


class PlayerCard(tk.Frame):
    """Widget displaying a single player's stats row."""

    def __init__(self, parent, player: PlayerStats, 
                 profession_icons: Optional[Dict[str, ImageTk.PhotoImage]] = None,
                 on_profession_change: Optional[Callable[[int, str], None]] = None,
                 **kwargs):
        """
        Initialize player card.

        Args:
            parent: Parent tkinter widget
            player: PlayerStats dataclass with player info
            profession_icons: Dictionary of profession icons
            on_profession_change: Callback for profession change
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
        self.profession_icons = profession_icons
        self.on_profession_change = on_profession_change

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

        # Profession Icon
        if self.profession_icons:
            icon_img = self.profession_icons.get(player.profession)
            if not icon_img:
                icon_img = self.profession_icons.get('Unknown')
            
            if icon_img:
                prof_btn = tk.Button(
                    self, 
                    image=icon_img, 
                    bg=bg_color, 
                    activebackground=bg_color, 
                    bd=0,
                    command=self._on_icon_click
                )
                prof_btn.image = icon_img # Keep ref
                # Pack near name
                prof_btn.pack(side=tk.LEFT, padx=(2, 2))

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

    def _on_icon_click(self):
        """Handle profession icon click."""
        if self.on_profession_change and self.profession_icons:
            ProfessionSelectorPopup(
                self.winfo_toplevel(),
                self.profession_icons,
                lambda p: self.on_profession_change(self.player.index, p)
            )
