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
        self.capture = ScreenCapture()
        self.running = True
        self.match_start_time = None
        self.match_screenshots = {"start": None, "end": None}

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

            # Capture full screen
            full_path = self.capture.capture_and_save_full("match_start")

            self.match_start_time = datetime.now()
            self.match_screenshots["start"] = full_path

            print(f"  -> Saved: {full_path}")
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

            # Capture full screen
            full_path = self.capture.capture_and_save_full("match_end")

            self.match_screenshots["end"] = full_path

            print(f"  -> Saved: {full_path}")

            # Show match summary
            if self.match_start_time:
                duration = datetime.now() - self.match_start_time
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                print(f"  -> Match duration: {minutes}m {seconds}s")

            print("\nMatch screenshots captured:")
            print(f"  Start: {self.match_screenshots['start']}")
            print(f"  End:   {self.match_screenshots['end']}")
            print("\nYou can now:")
            print("  1. Use manual_match_entry.py to log this match")
            print("  2. Press F8 to start capturing next match")
            print()

            # Reset for next match
            self.match_start_time = None
            self.match_screenshots = {"start": None, "end": None}

        except Exception as e:
            logger.error(f"Failed to capture match end: {e}")
            print(f"  -> ERROR: {e}\n")

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
