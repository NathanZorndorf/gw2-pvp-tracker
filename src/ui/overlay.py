"""
Win rate overlay window for displaying player statistics.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional
import logging

from .styles import (
    COLORS, FONTS, OVERLAY_WIDTH, OVERLAY_MIN_HEIGHT,
    PADDING_OUTER, PADDING_INNER, SECTION_SPACING
)
from .player_card import PlayerCard, PlayerStats

logger = logging.getLogger(__name__)


class WinRateOverlay:
    """
    Popup overlay window showing player win rates.

    This window appears on top of the game when F8 is pressed,
    displaying all 10 players with their statistics.
    """

    def __init__(self, root: Optional[tk.Tk] = None):
        """
        Initialize the overlay.

        Args:
            root: Optional existing Tk root. If None, creates hidden root.
        """
        self._owns_root = root is None
        self.root = root or tk.Tk()

        if self._owns_root:
            self.root.withdraw()  # Hide the root window

        self.window: Optional[tk.Toplevel] = None
        self._player_cards: List[PlayerCard] = []

    def show(self, players: List[PlayerStats]):
        """
        Show the overlay with player statistics.

        Args:
            players: List of PlayerStats for all 10 players
        """
        try:
            # Close existing window if open
            if self.window is not None:
                self.hide()

            # Create new top-level window
            self.window = tk.Toplevel(self.root)
            self._configure_window()
            self._create_content(players)

            # Center on screen
            self._center_window()

            # Bring to front
            self.window.lift()
            self.window.attributes('-topmost', True)

            logger.info("Win rate overlay displayed")

        except Exception as e:
            logger.error(f"Failed to show overlay: {e}")

    def hide(self):
        """Hide and destroy the overlay window."""
        if self.window is not None:
            try:
                self.window.destroy()
            except tk.TclError:
                pass  # Window already destroyed
            self.window = None
            self._player_cards = []
            logger.info("Win rate overlay hidden")

    def is_visible(self) -> bool:
        """Check if overlay is currently visible."""
        return self.window is not None and self.window.winfo_exists()

    def _configure_window(self):
        """Configure the overlay window properties."""
        self.window.title("Match Analysis")
        self.window.configure(bg=COLORS['bg_main'])
        self.window.resizable(False, False)

        # Remove window decorations on Windows for cleaner look
        # Comment out if you want standard title bar
        # self.window.overrideredirect(True)

        # Set window size
        self.window.geometry(f"{OVERLAY_WIDTH}x{OVERLAY_MIN_HEIGHT}")

        # Bind close button
        self.window.protocol("WM_DELETE_WINDOW", self.hide)

        # Bind Escape key to close
        self.window.bind('<Escape>', lambda e: self.hide())

    def _center_window(self):
        """Center the window on the screen."""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def _create_content(self, players: List[PlayerStats]):
        """Create the overlay content with side-by-side team layout."""
        # Main container
        main_frame = tk.Frame(self.window, bg=COLORS['bg_main'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=PADDING_OUTER, pady=PADDING_OUTER)

        # Title
        title_label = tk.Label(
            main_frame,
            text="Match Analysis",
            font=FONTS['title'],
            fg=COLORS['text_header'],
            bg=COLORS['bg_main']
        )
        title_label.pack(pady=(0, SECTION_SPACING))

        # Separate players by team
        red_players = [p for p in players if p.team == 'red']
        blue_players = [p for p in players if p.team == 'blue']

        # Teams container (side by side)
        teams_frame = tk.Frame(main_frame, bg=COLORS['bg_main'])
        teams_frame.pack(fill=tk.BOTH, expand=True)

        # Red team on left
        red_frame = tk.Frame(teams_frame, bg=COLORS['bg_main'])
        red_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, PADDING_INNER))
        self._create_team_section(red_frame, "RED TEAM", red_players, 'red')

        # Vertical divider
        divider = tk.Frame(teams_frame, bg=COLORS['divider'], width=2)
        divider.pack(side=tk.LEFT, fill=tk.Y, padx=PADDING_INNER)

        # Blue team on right
        blue_frame = tk.Frame(teams_frame, bg=COLORS['bg_main'])
        blue_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PADDING_INNER, 0))
        self._create_team_section(blue_frame, "BLUE TEAM", blue_players, 'blue')

        # Legend at bottom
        self._create_legend(main_frame)

    def _create_team_section(
        self,
        parent: tk.Frame,
        title: str,
        players: List[PlayerStats],
        team: str
    ):
        """Create a team section with header and player cards."""
        # Team header
        team_color = COLORS['team_red'] if team == 'red' else COLORS['team_blue']
        header = tk.Label(
            parent,
            text=title,
            font=FONTS['header'],
            fg=team_color,
            bg=COLORS['bg_main'],
            anchor='w'
        )
        header.pack(fill=tk.X, pady=(0, PADDING_INNER))

        # Player cards
        for player in players:
            card = PlayerCard(parent, player)
            card.pack(fill=tk.X, pady=1)
            self._player_cards.append(card)

    def _create_legend(self, parent: tk.Frame):
        """Create legend explaining the display."""
        legend_frame = tk.Frame(parent, bg=COLORS['bg_main'])
        legend_frame.pack(fill=tk.X, pady=(SECTION_SPACING, 0))

        legend_text = "Stars = Win Rate | (n) = Games Seen"
        legend_label = tk.Label(
            legend_frame,
            text=legend_text,
            font=FONTS['small'],
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_main']
        )
        legend_label.pack()

    def update(self):
        """Process pending tkinter events."""
        if self.root:
            try:
                self.root.update()
            except tk.TclError:
                pass

    def destroy(self):
        """Clean up the overlay and root window if owned."""
        self.hide()
        if self._owns_root and self.root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self.root = None
