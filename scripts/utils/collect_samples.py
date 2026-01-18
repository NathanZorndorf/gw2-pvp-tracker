import mss
import mss.tools
import numpy as np
from PIL import Image
import os
import sys
import time
from pathlib import Path
import ctypes
from ctypes import wintypes

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from integration.mumble_link import MumbleLink

def get_gw2_window_rect():
    """Finds the GW2 window and returns its boundaries."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "Guild Wars 2")
    if not hwnd:
        print("Error: Guild Wars 2 window not found! Make sure it is running.")
        return None
    
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return {
        "left": rect.left,
        "top": rect.top,
        "width": rect.right - rect.left,
        "height": rect.bottom - rect.top
    }

def capture_sample(ml: MumbleLink):
    """Captures a screenshot of the GW2 window and saves it with metadata."""
    rect = get_gw2_window_rect()
    if not rect:
        return

    # Check for valid window (minimized/invalid check)
    if rect["width"] <= 0 or rect["height"] <= 0:
        print("Error: Invalid window size. Is the game minimized?")
        return

    # Create output directory
    output_dir = Path("data/samples/resolution_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDetected GW2 Window: {rect['width']}x{rect['height']}")
    
    # Try to get UI Scale from MumbleLink
    scale_name = "unknown"
    if ml.is_active:
        ml.read()
        identity = ml.get_identity()
        uisz = identity.get("uisz")
        if uisz is not None:
            scale_map = {0: 'small', 1: 'normal', 2: 'large', 3: 'larger'}
            scale_name = scale_map.get(uisz, 'unknown')
            print(f"Auto-detected UI Scale: {scale_name} (uisz: {uisz})")
    
    if scale_name == "unknown":
        ui_scale = input("Enter UI Scale (s=Small, n=Normal, l=Large, L=Larger) or 'q' to quit: ").strip().lower()
        if ui_scale == 'q':
            return False
        scale_map = {'s': 'small', 'n': 'normal', 'l': 'large', 'L': 'larger', 'larger': 'larger'}
        scale_name = scale_map.get(ui_scale, 'unknown')

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{rect['width']}x{rect['height']}_{scale_name}_{timestamp}.png"
    filepath = output_dir / filename

    with mss.mss() as sct:
        # Capture the specific region
        screenshot = sct.grab(rect)
        mss.tools.to_png(screenshot.rgb, screenshot.size, outputpath=str(filepath))
        
    print(f"Saved: {filepath}")
    return True

if __name__ == "__main__":
    print("--- GW2 Resolution Sample Collector ---")
    print("1. Set GW2 to 'Windowed' mode.")
    print("2. Select a resolution and UI scale in-game.")
    print("3. Run this script to capture.")
    
    ml = MumbleLink()
    if not ml.is_active:
        print("Warning: MumbleLink not detected. UI Scale will need to be entered manually.")

    try:
        while True:
            cont = capture_sample(ml)
            if not cont:
                break
            print("\nChange settings in-game and press enter here for the next one...")
            input()
    except KeyboardInterrupt:
        print("\nExiting.")
