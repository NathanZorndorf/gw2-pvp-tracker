#!/usr/bin/env python3
"""
Fix color inversion in existing screenshots by converting BGR to RGB.
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_screenshot_colors(filepath: Path, force: bool = False):
    """Fix color inversion in a single screenshot."""
    try:
        # Load image
        img = Image.open(filepath)
        img_array = np.array(img)

        # Swap channels: RGB -> BGR (to correct the inversion)
        fixed_array = img_array[:, :, ::-1]

        # Save back as RGB
        fixed_img = Image.fromarray(fixed_array)
        fixed_img.save(filepath, format='PNG', compress_level=0)

        logger.info(f"Fixed colors in: {filepath}")

    except Exception as e:
        logger.error(f"Failed to fix {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fix color inversion in screenshots")
    parser.add_argument("--path", required=True, help="Path to folder containing screenshots")
    parser.add_argument("--force", action="store_true", help="Force overwrite without backup")
    args = parser.parse_args()

    folder = Path(args.path)
    if not folder.exists():
        logger.error(f"Path does not exist: {folder}")
        return

    # Find all PNG files
    png_files = list(folder.glob("**/*.png"))
    logger.info(f"Found {len(png_files)} PNG files in {folder}")

    if not png_files:
        logger.info("No PNG files found.")
        return

    for png_file in png_files:
        fix_screenshot_colors(png_file, args.force)


if __name__ == "__main__":
    main()