"""
Screenshot capture module for GW2 PvP Tracker.
Handles screen capture functionality with timing and region support.
"""

import mss
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Handles screenshot capture operations."""

    def __init__(self, screenshots_dir: str = "screenshots"):
        """
        Initialize screen capture.

        Args:
            screenshots_dir: Directory to save screenshots
        """
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.sct = mss.mss()
        logger.info(f"Screen capture initialized. Saving to: {self.screenshots_dir}")

    def capture_full_screen(self, monitor_index: int = 0) -> np.ndarray:
        """
        Capture full screen.

        Args:
            monitor_index: Monitor to capture (0 for primary)

        Returns:
            Screenshot as numpy array (RGB format)
        """
        try:
            # Get monitor (1-indexed, 0 is all monitors combined)
            monitor = self.sct.monitors[monitor_index + 1]

            # Capture screenshot
            sct_img = self.sct.grab(monitor)

            # Convert to numpy array (RGB)
            img = np.array(sct_img)
            img = img[:, :, :3]  # Remove alpha channel if present

            logger.debug(f"Captured full screen: {img.shape}")
            return img

        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            raise

    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Capture specific screen region.

        Args:
            x: Left coordinate
            y: Top coordinate
            width: Region width
            height: Region height

        Returns:
            Screenshot as numpy array (RGB format)
        """
        try:
            monitor = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }

            sct_img = self.sct.grab(monitor)
            img = np.array(sct_img)
            img = img[:, :, :3]

            logger.debug(f"Captured region: ({x}, {y}, {width}, {height})")
            return img

        except Exception as e:
            logger.error(f"Failed to capture region: {e}")
            raise

    def save_screenshot(
        self,
        image: np.ndarray,
        prefix: str = "screenshot",
        suffix: str = ""
    ) -> str:
        """
        Save screenshot to disk with timestamp.

        Args:
            image: Image as numpy array
            prefix: Filename prefix
            suffix: Filename suffix (e.g., "_full", "_roster")

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}{suffix}.png"
        filepath = self.screenshots_dir / filename

        # Convert RGB to PIL Image and save
        pil_image = Image.fromarray(image)
        pil_image.save(filepath)

        logger.info(f"Saved screenshot: {filepath}")
        return str(filepath)

    def capture_and_save_full(
        self,
        prefix: str = "match",
        monitor_index: int = 0
    ) -> str:
        """
        Capture full screen and save immediately.

        Args:
            prefix: Filename prefix
            monitor_index: Monitor to capture

        Returns:
            Path to saved file
        """
        img = self.capture_full_screen(monitor_index)
        return self.save_screenshot(img, prefix, "_full")

    def capture_and_save_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        prefix: str = "match"
    ) -> str:
        """
        Capture region and save immediately.

        Args:
            x: Left coordinate
            y: Top coordinate
            width: Region width
            height: Region height
            prefix: Filename prefix

        Returns:
            Path to saved file
        """
        img = self.capture_region(x, y, width, height)
        return self.save_screenshot(img, prefix, "_region")

    def capture_match_start(
        self,
        monitor_index: int = 0,
        crop_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Capture match start screenshots (full + optional crop).

        Args:
            monitor_index: Monitor to capture
            crop_region: Optional (x, y, width, height) for cropped roster

        Returns:
            Tuple of (full_screenshot_path, cropped_screenshot_path)
        """
        logger.info("Capturing match start...")

        # Capture full screen
        full_path = self.capture_and_save_full("start", monitor_index)

        # Optionally capture cropped region
        cropped_path = None
        if crop_region:
            x, y, w, h = crop_region
            cropped_path = self.capture_and_save_region(x, y, w, h, "start_roster")

        return full_path, cropped_path

    def capture_match_end(
        self,
        monitor_index: int = 0,
        crop_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Capture match end screenshots (full + optional crop).

        Args:
            monitor_index: Monitor to capture
            crop_region: Optional (x, y, width, height) for cropped roster

        Returns:
            Tuple of (full_screenshot_path, cropped_screenshot_path)
        """
        logger.info("Capturing match end...")

        # Capture full screen
        full_path = self.capture_and_save_full("end", monitor_index)

        # Optionally capture cropped region
        cropped_path = None
        if crop_region:
            x, y, w, h = crop_region
            cropped_path = self.capture_and_save_region(x, y, w, h, "end_roster")

        return full_path, cropped_path

    def get_monitor_info(self) -> list:
        """Get information about available monitors."""
        monitors = []
        for i, monitor in enumerate(self.sct.monitors[1:], start=0):
            monitors.append({
                "index": i,
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"]
            })
        return monitors

    def __del__(self):
        """Cleanup MSS instance."""
        if hasattr(self, 'sct'):
            self.sct.close()
