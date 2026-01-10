"""
Live screenshot capture for GW2 PvP matches.

Press F8 at match start to capture beginning screenshot.
Press F9 at match end to capture ending screenshot.
Press ESC to exit the program.

The app will save timestamped screenshots to the screenshots/ folder.
"""

import sys
from pathlib import Path
import keyboard
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.capture import ScreenCapture
from config import Config
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
        self.config = Config()
        # Don't initialize ScreenCapture here - create new instance per capture
        self.running = True
        self.match_start_time = None
        self.match_screenshots = {"start": None, "end": None}
        self.detected_user_character = None  # Stores (char_name, team) from F8 detection

        print("\n" + "=" * 70)
        print("  GW2 PvP Tracker - Live Capture Mode")
        print("=" * 70)
        print("\nHotkeys:")
        print("  F8  - Capture match START (press when scoreboard is visible)")
        print("  F9  - Capture match END (press when match is over)")
        print("  ESC - Exit program")
        print("\nTips:")
        print("  - Open the scoreboard (default: Tab key) before pressing F8/F9")
        print("  - Screenshots are saved to: screenshots/")
        print("  - Files are named with timestamps for easy identification")
        print("\nListening for hotkeys... (Keep this window in background)")
        print("=" * 70 + "\n")

    def capture_match_start(self):
        """Capture match start screenshot."""
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] F8 pressed - Capturing match START...")

            # Small delay to ensure scoreboard is fully rendered
            time.sleep(0.3)

            # Create new ScreenCapture instance (thread-safe)
            capture = ScreenCapture()

            # Capture full screen
            full_path = capture.capture_and_save_full("match_start")

            self.match_start_time = datetime.now()
            self.match_screenshots["start"] = full_path

            print(f"  -> Saved: {full_path}")

            # Detect user character from F8 screenshot
            print("  -> Detecting your character...")
            self._detect_user_from_screenshot(full_path)

            print("  -> Ready! Press F9 when match ends.\n")

        except Exception as e:
            logger.error(f"Failed to capture match start: {e}")
            print(f"  -> ERROR: {e}\n")

    def capture_match_end(self):
        """Capture match end screenshot."""
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] F9 pressed - Capturing match END...")

            # Small delay to ensure final scores are visible
            time.sleep(0.3)

            # Create new ScreenCapture instance (thread-safe)
            capture = ScreenCapture()

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

            # Auto-process and log match
            print("\n  -> Processing match data (OCR + logging)...")
            self._auto_process_match()

            # Reset for next match
            self.match_start_time = None
            self.match_screenshots = {"start": None, "end": None}
            self.detected_user_character = None

        except Exception as e:
            logger.error(f"Failed to capture match end: {e}")
            print(f"  -> ERROR: {e}\n")

    def _detect_user_from_screenshot(self, screenshot_path: str):
        """Detect user character from F8/F9 screenshot using bold detection."""
        try:
            from automation.match_processor import MatchProcessor
            from database.models import Database

            db = Database(self.config.get('database.path'))
            processor = MatchProcessor(self.config, db)

            # Detect user from screenshot
            user_char, user_team, confidence = processor.detect_user_from_image(screenshot_path)

            if user_char:
                print(f"  -> Detected: {user_char} ({user_team.upper()} team) [confidence: {confidence:.2f}]")
                self.detected_user_character = (user_char, user_team)
            else:
                print(f"  -> WARNING: Could not detect your character (bold name not found)")
                self.detected_user_character = None

            db.close()

        except Exception as e:
            logger.error(f"User detection failed: {e}")
            print(f"  -> WARNING: User detection failed: {e}")
            self.detected_user_character = None

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
                detected_user=self.detected_user_character
            )

            # Display extraction results
            if match_data['success']:
                print(f"  -> Scores: Red {match_data['red_score']} - Blue {match_data['blue_score']}")
                print(f"  -> Your character: {match_data['user_character']} ({match_data['user_team'].upper()} team)")
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
                self.match_screenshots['end']
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
                keyboard.add_hotkey('esc', self.exit_program)
                print("Hotkeys registered successfully!\n")
            except Exception as e:
                print("\nERROR: Could not register hotkeys!")
                print("This usually means the program needs Administrator privileges.\n")
                print("Solution:")
                print("  1. Close this window")
                print("  2. Right-click on your terminal (Command Prompt or PowerShell)")
                print("  3. Select 'Run as administrator'")
                print("  4. Navigate back to this folder")
                print("  5. Run: python live_capture.py\n")
                raise

            # Keep running until ESC is pressed
            while self.running:
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in live capture: {e}")
        finally:
            # Cleanup
            keyboard.unhook_all()
            print("Hotkeys unregistered.")


if __name__ == "__main__":
    print("\nStarting GW2 PvP Live Capture...")
    print("Make sure GW2 is running and you're ready to capture!")
    print("\nPress any key to continue or Ctrl+C to cancel...")

    try:
        # Wait for user confirmation
        input()

        # Start live capture
        app = LiveCapture()
        app.run()

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
