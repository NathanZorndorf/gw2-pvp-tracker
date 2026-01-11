"""
Confidence/star rating calculation and display utilities.
"""

from .styles import SYMBOLS, COLORS, get_star_count


def get_stars_display(win_rate: float, total_matches: int) -> str:
    """
    Generate star rating display string.

    Args:
        win_rate: Win rate percentage (0-100)
        total_matches: Number of matches played

    Returns:
        String with filled/empty stars (e.g., "★★★☆☆")
    """
    if total_matches == 0:
        return "-"

    star_count = get_star_count(win_rate)
    filled = SYMBOLS['star_filled'] * star_count
    empty = SYMBOLS['star_empty'] * (5 - star_count)
    return filled + empty


def get_winrate_indicator(win_rate: float, total_matches: int) -> tuple:
    """
    Get win rate indicator symbol and color.

    Args:
        win_rate: Win rate percentage (0-100)
        total_matches: Number of matches played

    Returns:
        Tuple of (symbol, color)
    """
    if total_matches == 0:
        return ("-", COLORS['winrate_new'])

    if win_rate >= 50:
        return (SYMBOLS['check'], COLORS['winrate_high'])
    else:
        return (SYMBOLS['cross'], COLORS['winrate_low'])


def format_winrate(win_rate: float, total_matches: int) -> str:
    """
    Format win rate for display.

    Args:
        win_rate: Win rate percentage (0-100)
        total_matches: Number of matches played

    Returns:
        Formatted string (e.g., "52.3%" or "NEW")
    """
    if total_matches == 0:
        return "NEW"
    return f"{win_rate:.1f}%"


def format_matches(total_matches: int) -> str:
    """
    Format match count for display.

    Args:
        total_matches: Number of matches played

    Returns:
        Formatted string (e.g., "12" or "-")
    """
    if total_matches == 0:
        return "-"
    return str(total_matches)
