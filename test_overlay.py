"""
Test script to preview the win rate overlay GUI without running live capture.
Run: python test_overlay.py
Run: python test_overlay.py --live   (use real data from ranked-1 screenshot)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import time
import tkinter as tk
from ui import WinRateOverlay, PlayerStats


def get_sample_players():
    """Return hardcoded sample player data."""
    return [
        # Red team
        PlayerStats(name="Cameronz", team="red", win_rate=52.3, total_matches=44, is_user=False),
        PlayerStats(name="Fahrôs", team="red", win_rate=0.0, total_matches=0, is_user=False),
        PlayerStats(name="I Am The Weapon", team="red", win_rate=48.1, total_matches=27, is_user=False),
        PlayerStats(name="Darkliken", team="red", win_rate=61.2, total_matches=85, is_user=False),
        PlayerStats(name="Mira Phantom", team="red", win_rate=45.0, total_matches=12, is_user=False),

        # Blue team
        PlayerStats(name="Katrimus", team="blue", win_rate=55.0, total_matches=120, is_user=False),
        PlayerStats(name="General Huggles", team="blue", win_rate=0.0, total_matches=0, is_user=False),
        PlayerStats(name="Veiny Trunk", team="blue", win_rate=42.8, total_matches=35, is_user=False),
        PlayerStats(name="Grizzk Rainfall", team="blue", win_rate=73.3, total_matches=60, is_user=False),
        PlayerStats(name="Phantom Tithe", team="blue", win_rate=100.0, total_matches=3, is_user=True),
    ]


def get_live_players():
    """Extract players from ranked-1 screenshot using real OCR and database."""
    t_start = time.perf_counter()
    from config import Config
    from database.models import Database
    from automation.match_processor import MatchProcessor
    import cv2

    t_after_imports = time.perf_counter()
    print(f"[benchmark] imports: {t_after_imports - t_start:.3f}s")

    config = Config()
    db = Database("data/pvp_tracker.db")
    processor = MatchProcessor(config, db)
    t_after_init = time.perf_counter()
    print(f"[benchmark] init Config/DB/Processor: {t_after_init - t_after_imports:.3f}s")

    # Load ranked-4 start screenshot
    image_path = "data/samples/ranked-4/match_start_20260110_182154_full.png"
    image = cv2.imread(image_path)
    t_after_read = time.perf_counter()
    print(f"[benchmark] cv2.imread: {t_after_read - t_after_init:.3f}s")

    # Detect arena type and extract players
    t0 = time.perf_counter()
    processor.detect_arena_type(image)
    players_data = processor._extract_all_players(image, "start")
    t_after_extract = time.perf_counter()
    print(f"[benchmark] detect_arena_type+extract_all_players: {t_after_extract - t0:.3f}s")

    # Detect user (use loaded image to avoid re-reading and duplicate arena detection)
    t0 = time.perf_counter()
    user_char, user_team, _ = processor.detect_user_from_image(image=image)
    t_after_user = time.perf_counter()
    print(f"[benchmark] detect_user_from_image: {t_after_user - t0:.3f}s")

    # Build PlayerStats with real win rates from database
    t0 = time.perf_counter()
    players = []
    for p in players_data:
        win_rate, total_matches = db.get_player_winrate(p['name'])
        is_user = (p['name'] == user_char) if user_char else False
        players.append(PlayerStats(
            name=p['name'],
            team=p['team'],
            win_rate=win_rate,
            total_matches=total_matches,
            is_user=is_user
        ))
    t_after_dblookups = time.perf_counter()
    print(f"[benchmark] db.get_player_winrate loop: {t_after_dblookups - t0:.3f}s")

    db.close()
    t_end = time.perf_counter()
    print(f"[benchmark] total get_live_players: {t_end - t_start:.3f}s")
    return players


def main():
    # Check for --live flag
    use_live = "--live" in sys.argv

    if use_live:
        print("Loading real data from ranked-1 screenshot...")
        players = get_live_players()
    else:
        print("Using sample data (run with --live for real OCR data)")
        players = get_sample_players()

    print(f"Loaded {len(players)} players")
    print("Close the window or press ESC to exit.\n")

    # Create tkinter root
    root = tk.Tk()
    root.withdraw()

    def on_close():
        """Handle window close - exit the app."""
        root.quit()
        root.destroy()

    # Create and show overlay
    overlay = WinRateOverlay(root)
    overlay.show(players)

    # Override close behavior to exit app
    overlay.window.protocol("WM_DELETE_WINDOW", on_close)
    overlay.window.bind('<Escape>', lambda e: on_close())

    # Run tkinter mainloop
    root.mainloop()

    print("Done!")


if __name__ == "__main__":
    main()
