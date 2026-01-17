"""
Live screenshot capture for GW2 PvP matches.

Press F8 at match start to capture beginning screenshot.
Press F9 at match end to capture ending screenshot.

The app will save timestamped screenshots to the screenshots/ folder.
"""

import sys
import os
import argparse
from pathlib import Path
import keyboard
import time
from datetime import datetime
import tkinter as tk
from typing import Optional, List

# Add src to path so we can import modules
# File is at src/automation/live_capture.py
# We want to add 'src' to path.
# Path(__file__).parent = src/automation
# Path(__file__).parent.parent = src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.capture import ScreenCapture
from config import Config
from ui import WinRateOverlay, PlayerStats
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveCapture:
    """Live screenshot capture with hotkeys."""

    def __init__(self):
        # Allow overriding detected arena/map via CLI arg or env var
        self.selected_map = None
        try:
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--map', dest='map', help='Preselected map/arena name')
            args, _ = parser.parse_known_args()
            if args.map:
                self.selected_map = args.map
        except Exception:
            pass

        # Environment variable fallback
        if not self.selected_map:
            self.selected_map = os.environ.get('LIVE_CAPTURE_MAP')

        if self.selected_map:
            print(f"Using selected map override: {self.selected_map}")

        self.config = Config()
        self.screenshots_root = Path(self.config.get('paths.screenshots', 'screenshots'))
        self.current_match_dir = None
        # Don't initialize ScreenCapture here - create new instance per capture
        self.running = True
        self.match_start_time = None
        self.match_screenshots = {"start": None, "end": None}
        self.detected_user_character = None  # Stores (char_name, team) from F8 detection
        self.detected_players: List[dict] = []  # Stores player data from F8 for overlay

        # Initialize tkinter for overlay (must be in main thread)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the root window

        # Initialize overlay
        self.overlay_enabled = self.config.get('ui.overlay.enabled', True)
        self.overlay: Optional[WinRateOverlay] = None
        if self.overlay_enabled:
            self.overlay = WinRateOverlay(
                self.root, 
                on_profession_change=self.update_player_profession
            )
            logger.info("Win rate overlay initialized")

        # Pending overlay action (for thread-safe GUI updates)
        self._pending_overlay_action = None
        self._pending_action = None

        print("\n" + "=" * 70)
        print("  GW2 PvP Tracker - Live Capture Mode")
        print("=" * 70)
        print("\nHotkeys:")
        print("  F8  - Capture match START (press when scoreboard is visible)")
        print("  F9  - Capture match END (press when match is over)")
        print("\nTips:")
        print("  - Open the scoreboard (default: Tab key) before pressing F8/F9")
        print("  - Screenshots are saved to: screenshots/match_YYYYMMDD_HHMMSS/")
        print("  - Files are named with timestamps for easy identification")
        if self.overlay_enabled:
            print("  - Win rate overlay will appear on F8 (close with X or ESC)")
        print("\nListening for hotkeys... (Keep this window in background)")
        print("=" * 70 + "\n")

    def capture_match_start(self):
        """Capture match start screenshot."""
        try:
            # Check for retake
            if self.match_screenshots["start"] and os.path.exists(self.match_screenshots["start"]):
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] F8 pressed again - RETAKING match START...")
                try:
                    os.remove(self.match_screenshots["start"])
                    print(f"  -> Deleted previous capture: {os.path.basename(self.match_screenshots['start'])}")
                except OSError as e:
                    print(f"  -> WARNING: Could not delete previous capture: {e}")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] F8 pressed - Capturing match START...")

            # Small delay to ensure scoreboard is fully rendered
            time.sleep(0.3)

            # Create match folder if not already existing (not a retake)
            if not self.match_screenshots["start"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_match_dir = self.screenshots_root / f"match_{timestamp}"
                self.current_match_dir.mkdir(parents=True, exist_ok=True)

            # Create new ScreenCapture instance using the match folder
            capture = ScreenCapture(screenshots_dir=str(self.current_match_dir))

            # Capture full screen
            full_path = capture.capture_and_save_full("match_start")

            self.match_start_time = datetime.now()
            self.match_screenshots["start"] = full_path

            print(f"  -> Saved: {full_path}")

            # Detect user character and extract all players from F8 screenshot
            print("  -> Detecting players and your character...")
            self._detect_players_from_screenshot(full_path)

            # Show overlay with win rates (schedule for main thread)
            if self.overlay_enabled and self.overlay and self.detected_players:
                print("  -> Showing win rate overlay...")
                self._pending_overlay_action = 'show'

            print("  -> Ready! Press F9 when match ends.\n")

        except Exception as e:
            logger.error(f"Failed to capture match start: {e}")
            print(f"  -> ERROR: {e}\n")

    def update_player_profession(self, index: int, profession: str):
        """Update profession for a player (called from overlay)."""
        if 0 <= index < len(self.detected_players):
            old_prof = self.detected_players[index].get('profession', 'Unknown')
            self.detected_players[index]['profession'] = profession
            print(f"  -> Updated player {index} profession: {old_prof} -> {profession}")
            
            # Refresh overlay
            self._pending_overlay_action = 'show'

    def capture_match_end(self):
        """Capture match end screenshot."""
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] F9 pressed - Capturing match END...")

            # Hide overlay if configured
            if self.overlay_enabled and self.config.get('ui.overlay.auto_close_on_f9', True):
                self._pending_overlay_action = 'hide'

            # Small delay to ensure final scores are visible
            time.sleep(0.3)

            # Ensure we have a match folder (user might have skipped F8)
            if not self.current_match_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_match_dir = self.screenshots_root / f"match_{timestamp}"
                self.current_match_dir.mkdir(parents=True, exist_ok=True)

            # Create new ScreenCapture instance using the match folder
            capture = ScreenCapture(screenshots_dir=str(self.current_match_dir))

            # Capture full screen
            full_path = capture.capture_and_save_full("match_end")

            self.match_screenshots["end"] = full_path

            print(f"  -> Saved: {full_path}")

            # Show match summary
            if self.match_start_time:
                duration = datetime.now() - self.match_start_time
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                print(f"  -> Match duration: {minutes}m {seconds}s")

            # Queue processing in main thread
            self._pending_action = 'process_match_end'

        except Exception as e:
            logger.error(f"Failed to capture match end: {e}")
            print(f"  -> ERROR: {e}\n")

    def _detect_players_from_screenshot(self, screenshot_path: str):
        """Detect all players and user character from F8 screenshot."""
        try:
            from automation.match_processor import MatchProcessor
            from database.models import Database
            import cv2

            db = Database(self.config.get('database.path'))
            processor = MatchProcessor(self.config, db)

            # Load image
            image = cv2.imread(screenshot_path)
            if image is None:
                raise ValueError(f"Failed to load image: {screenshot_path}")

            # Detect arena type from image (map selection is separate)
            processor.detect_arena_type(image)

            # Extract all players
            players_data = processor._extract_all_players(image, "start")
            self.detected_players = players_data

            # Detect user character
            user_char, user_team, confidence = processor.detect_user_from_image(screenshot_path)

            if user_char:
                print(f"  -> Detected: {user_char} ({user_team.upper()} team) [confidence: {confidence:.2f}]")
                self.detected_user_character = (user_char, user_team)
            else:
                print(f"  -> WARNING: Could not detect your character (bold name not found)")
                self.detected_user_character = None

            # Fetch win rates for all players
            for player in self.detected_players:
                win_rate, total_matches = db.get_player_winrate(player['name'])
                player['win_rate'] = win_rate
                player['total_matches'] = total_matches
                player['is_user'] = (
                    self.detected_user_character and
                    player['name'] == self.detected_user_character[0]
                )

            print(f"  -> Extracted {len(self.detected_players)} players with win rates")

            db.close()

        except Exception as e:
            logger.error(f"Error detecting user from image: {e}")
            print(f"  -> WARNING: Could not detect your character (bold name not found)")
            self.detected_user_character = None
            self.detected_players = []

    def _build_player_stats(self) -> List[PlayerStats]:
        """Build PlayerStats list from detected players."""
        stats = []
        for idx, player in enumerate(self.detected_players):
            stats.append(PlayerStats(
                name=player.get('name', 'Unknown'),
                profession=player.get('profession', 'Unknown'),
                team=player.get('team', 'red'),
                win_rate=player.get('win_rate', 0.0),
                total_matches=player.get('total_matches', 0),
                is_user=player.get('is_user', False),
                index=idx
            ))
        return stats

    def _process_pending_overlay_action(self):
        """Process pending overlay actions in the main thread."""
        if self._pending_overlay_action == 'show' and self.overlay:
            player_stats = self._build_player_stats()
            if player_stats:
                self.overlay.show(player_stats)
            self._pending_overlay_action = None
        elif self._pending_overlay_action == 'hide' and self.overlay:
            self.overlay.hide()
            self._pending_overlay_action = None

    def _auto_process_match(self):
        """Process captured match and log to database."""
        try:
            from automation.match_processor import MatchProcessor
            from database.models import Database

            # Initialize processor
            db = Database(self.config.get('database.path'))
            processor = MatchProcessor(self.config, db)

            # Process screenshots (pass detected user from F8 if available)
            match_data = processor.process_match(
                self.match_screenshots['start'],
                self.match_screenshots['end'],
                detected_user=self.detected_user_character,
                map_name=self.selected_map,
                known_players=self.detected_players
            )

            # Display extraction results
            if match_data['success']:
                print(f"  -> Scores: Red {match_data['red_score']} - Blue {match_data['blue_score']}")
                print(f"  -> Your character: {match_data['user_character']} ({match_data['user_team'].upper()} team)")
                print(f"  -> Map: {match_data.get('map_name', 'Unknown')}")
                print(f"  -> Extracted {len(match_data['players'])} players")

                if match_data['validation_errors']:
                    print(f"  -> Warnings: {', '.join(match_data['validation_errors'])}")
            else:
                print(f"  -> ERROR: {match_data['error']}")
                print(f"  -> Match will be logged with partial data")

            # Log to database (best-effort)
            match_id = processor.log_match(
                match_data,
                self.match_screenshots['start'],
                self.match_screenshots['end'],
                map_name=match_data.get('map_name')
            )

            # Display result
            winner = 'Blue' if match_data.get('blue_score', 0) > match_data.get('red_score', 0) else 'Red'
            user_result = 'WIN' if match_data.get('user_team', '').lower() == winner.lower() else 'LOSS'

            print(f"\n  -> Match #{match_id} logged successfully!")
            print(f"  -> Result: {user_result}")
            print(f"  -> Run 'python view_stats.py' to see updated stats\n")

            db.close()

        except Exception as e:
            logger.error(f"Auto-processing failed: {e}", exc_info=True)
            print(f"\n  -> ERROR: Auto-processing failed: {e}")
            print(f"  -> You can manually log this match using log_latest_match.py\n")

    def exit_program(self):
        """Exit the program."""
        print("\n[ESC pressed] Exiting live capture mode...")
        print("Goodbye!\n")
        self.running = False

    def run(self):
        """Start listening for hotkeys."""
        try:
            # Test if we have permission to use global hotkeys
            print("\nTesting hotkey permissions...")

            try:
                keyboard.add_hotkey('f8', self.capture_match_start)
                keyboard.add_hotkey('f9', self.capture_match_end)
                # keyboard.add_hotkey('esc', self.exit_program)  # Disabled to prevent accidental exit
                print("Hotkeys registered successfully!\n")
            except Exception as e:
                print("\nERROR: Could not register hotkeys!")
                print("This usually means the program needs Administrator privileges.\n")
                print("Solution:")
                print("  1. Close this window")
                print("  2. Right-click on your terminal (Command Prompt or PowerShell)")
                print("  3. Select 'Run as administrator'")
                print("  4. Navigate back to this folder")
                print("  5. Run: python src/automation/live_capture.py\n")
                raise

            # Keep running until ESC is pressed
            # Use tkinter mainloop with polling for overlay updates
            while self.running:
                # Process pending overlay actions (thread-safe GUI updates)
                self._process_pending_overlay_action()

                # Process main workflow actions
                if self._pending_action == 'process_match_end':
                    self._pending_action = None
                    
                    # Process directly without asking for map
                    print("\n  -> Processing match data (OCR + logging)...")
                    self._auto_process_match()

                    # Reset
                    self.match_start_time = None
                    self.match_screenshots = {"start": None, "end": None}
                    self.current_match_dir = None
                    self.detected_user_character = None
                    self.detected_players = []

                # Update tkinter
                try:
                    self.root.update()
                except tk.TclError:
                    pass

                time.sleep(0.05)  # 50ms polling interval

        except Exception as e:
            logger.error(f"Error in live capture: {e}")
        finally:
            # Cleanup
            keyboard.unhook_all()
            print("Hotkeys unregistered.")

            # Cleanup overlay
            if self.overlay:
                self.overlay.destroy()
            if self.root:
                try:
                    self.root.destroy()
                except tk.TclError:
                    pass


if __name__ == "__main__":
    print("\nStarting GW2 PvP Live Capture...")
    print("Make sure GW2 is running and you're ready to capture!")
    print("\nPress any key to continue or Ctrl+C to cancel...")

    try:
        # Start live capture
        app = LiveCapture()
        app.run()

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
