"""
UI styling constants for the win rate overlay.
"""

# Window dimensions
OVERLAY_WIDTH = 750  # Wider for side-by-side layout
OVERLAY_MIN_HEIGHT = 320  # Shorter since teams are side by side

# Padding and spacing
PADDING_OUTER = 15
PADDING_INNER = 8
ROW_HEIGHT = 32
SECTION_SPACING = 10

# Colors - Dark theme matching GW2 aesthetic
COLORS = {
    # Background colors
    'bg_main': '#1a1a2e',           # Dark blue-black
    'bg_header': '#16213e',          # Slightly lighter header
    'bg_row_red': '#2d1f1f',         # Dark red tint for red team rows
    'bg_row_blue': '#1f1f2d',        # Dark blue tint for blue team rows
    'bg_row_user': '#2a2a4a',        # Highlighted row for user

    # Team colors
    'team_red': '#e74c3c',           # Red team accent
    'team_blue': '#3498db',          # Blue team accent
    'team_red_light': '#ff6b5b',     # Lighter red for text
    'team_blue_light': '#5dade2',    # Lighter blue for text

    # Text colors
    'text_primary': '#ecf0f1',       # Main text (light gray)
    'text_secondary': '#95a5a6',     # Secondary text (muted)
    'text_header': '#ffffff',        # Header text (white)

    # Win rate indicator colors
    'winrate_high': '#2ecc71',       # Green for >55%
    'winrate_mid': '#f39c12',        # Orange for 45-55%
    'winrate_low': '#e74c3c',        # Red for <45%
    'winrate_new': '#95a5a6',        # Gray for new players

    # Star colors
    'star_filled': '#f1c40f',        # Gold for filled stars
    'star_empty': '#34495e',         # Dark gray for empty stars

    # Border and accents
    'border': '#34495e',             # Border color
    'divider': '#2c3e50',            # Section divider
}

# Fonts
FONTS = {
    'title': ('Segoe UI', 14, 'bold'),
    'header': ('Segoe UI', 11, 'bold'),
    'player_name': ('Segoe UI', 10),
    'player_name_user': ('Segoe UI', 10, 'bold'),
    'stats': ('Consolas', 10),
    'stars': ('Segoe UI Symbol', 10),
    'small': ('Segoe UI', 9),
}

# Star rating thresholds (win rate -> stars)
STAR_THRESHOLDS = [
    (0, 1),      # 0-20% = 1 star
    (20, 2),     # 20-40% = 2 stars
    (40, 3),     # 40-55% = 3 stars
    (55, 4),     # 55-70% = 4 stars
    (70, 5),     # 70%+ = 5 stars
]

# Symbols
SYMBOLS = {
    'star_filled': '\u2605',    # Black star
    'star_empty': '\u2606',     # White star
    'check': '\u2714',          # Check mark
    'cross': '\u2716',          # X mark
    'circle_check': '\u2713',   # Check
    'circle_cross': '\u2717',   # Ballot X
}


def get_winrate_color(win_rate: float) -> str:
    """Get color based on win rate percentage."""
    if win_rate >= 55:
        return COLORS['winrate_high']
    elif win_rate >= 45:
        return COLORS['winrate_mid']
    else:
        return COLORS['winrate_low']


def get_stylesheet() -> str:
        """Return a Qt stylesheet string using the COLORS constants.

        This centralizes the app styling so all windows/tabs use a consistent
        dark theme with good contrast for text, headers, tables and buttons.
        """
        c = COLORS
        # Base font family
        font_family = 'Segoe UI, Helvetica, Arial'

        qss = f"""
/* Base widget and window */
QWidget {{
    background-color: {c['bg_main']};
    color: {c['text_primary']};
    font-family: {font_family};
}}

/* Tab widget */
QTabWidget::pane {{
    border: 1px solid {c['border']};
    background: {c['bg_main']};
}}
QTabBar::tab {{
    background: {c['bg_header']};
    color: {c['text_primary']};
    padding: 8px 14px;
    margin: 1px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {c['bg_header']};
    color: {c['text_header']};
}}

/* Buttons */
QPushButton {{
    background-color: transparent;
    border: 1px solid {c['border']};
    color: {c['text_primary']};
    padding: 6px 10px;
    border-radius: 4px;
}}
QPushButton:hover {{
    background-color: rgba(255,255,255,0.02);
}}
QPushButton:pressed {{
    background-color: rgba(0,0,0,0.12);
}}
QPushButton:disabled {{
    color: {c['text_secondary']};
    border-color: rgba(255,255,255,0.04);
}}
QPushButton:checked {{
    background-color: rgba(255,255,255,0.03);
    border: 1px solid {c['divider']};
}}

/* Tables */
QTableWidget, QTableView {{
    background-color: {c['bg_main']};
    gridline-color: rgba(255,255,255,0.03);
    color: {c['text_primary']};
}}
QHeaderView::section {{
    background-color: {c['bg_header']};
    color: {c['text_header']};
    padding: 6px;
    border: 1px solid {c['border']};
}}
QTableWidget QTableCornerButton::section {{
    background: {c['bg_header']};
}}

/* Scroll areas */
QScrollArea {{
    background: transparent;
}}

/* Tooltips */
QToolTip {{
    background-color: {c['bg_header']};
    color: {c['text_primary']};
    border: 1px solid {c['border']};
}}

/* Specific small tweaks for better contrast */
QLabel#summary_label {{
    color: {c['text_primary']};
    font-weight: bold;
}}

/* Make checked map buttons visually distinct */
QPushButton:checked {{
    outline: none;
    box-shadow: none;
}}

"""
        return qss


def get_star_count(win_rate: float) -> int:
    """Convert win rate to star count (1-5)."""
    for threshold, stars in reversed(STAR_THRESHOLDS):
        if win_rate >= threshold:
            return stars
    return 1
