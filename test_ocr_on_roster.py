"""
Test OCR specifically on GW2 roster screenshots.
This script will help us find the right preprocessing settings.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.ocr_engine import OCREngine
from config import Config


def display_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def extract_player_names_from_roster():
    """
    Extract player names from the roster screenshots.
    We'll manually define the regions where names appear.
    """
    display_section("GW2 Roster OCR Test")

    # Load roster image
    img = cv2.imread("roster - beginning.PNG")
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]} pixels")

    # Initialize OCR
    config = Config()
    ocr = OCREngine(tesseract_path=config.get('ocr.tesseract_path'))

    # Looking at your screenshot, the player names are in two columns:
    # Red team (left) and Blue team (right)
    # Let's extract some test regions

    display_section("Testing Different Regions")

    # Test regions (you can adjust these based on what we find)
    test_regions = [
        {
            "name": "Top-left (Score area)",
            "coords": (0, 0, 600, 100),
            "expected": "Should contain 'Score' and team scores"
        },
        {
            "name": "Red Team - First player name",
            "coords": (50, 100, 250, 140),
            "expected": "Dragon's Despair"
        },
        {
            "name": "Blue Team - First player name",
            "coords": (900, 100, 350, 140),
            "expected": "Terminal Gearning"
        },
        {
            "name": "Red Team - Second player",
            "coords": (50, 140, 250, 180),
            "expected": "Legendary L O K I"
        },
    ]

    for i, region_info in enumerate(test_regions):
        print(f"\n--- Test {i+1}: {region_info['name']} ---")
        print(f"Expected: {region_info['expected']}")

        x, y, w, h = region_info['coords']
        region = img[y:y+h, x:x+w]

        # Save region for visual inspection
        region_path = f"screenshots/test_region_{i+1}.png"
        cv2.imwrite(region_path, region)
        print(f"Saved region to: {region_path}")

        # Try different preprocessing approaches
        print("\nPreprocessing approach 1: Standard (invert for white text)")
        processed1 = ocr.preprocess_for_ocr(region, resize_factor=3.0, invert=True)
        cv2.imwrite(f"screenshots/test_region_{i+1}_processed1.png", processed1)
        text1 = ocr.extract_text(processed1, psm=7, preprocess=False)
        print(f"Result 1: '{text1}'")

        print("\nPreprocessing approach 2: No inversion (black text)")
        processed2 = ocr.preprocess_for_ocr(region, resize_factor=3.0, invert=False)
        cv2.imwrite(f"screenshots/test_region_{i+1}_processed2.png", processed2)
        text2 = ocr.extract_text(processed2, psm=7, preprocess=False)
        print(f"Result 2: '{text2}'")

        print("\nPreprocessing approach 3: Just grayscale + resize")
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"screenshots/test_region_{i+1}_processed3.png", resized)
        text3 = ocr.extract_text(resized, psm=7, preprocess=False)
        print(f"Result 3: '{text3}'")

    display_section("Summary")
    print("\nCheck the screenshots/ folder to see:")
    print("  - test_region_X.png: Original cropped regions")
    print("  - test_region_X_processedY.png: Different preprocessing attempts")
    print("\nNext steps:")
    print("  1. Review which preprocessing works best")
    print("  2. Adjust region coordinates if needed")
    print("  3. Fine-tune OCR settings based on results")


def test_score_extraction():
    """Test extracting the final scores from match end screenshot."""
    display_section("Score Extraction Test")

    img = cv2.imread("roster - end.PNG")
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]} pixels")

    config = Config()
    ocr = OCREngine(tesseract_path=config.get('ocr.tesseract_path'))

    # The scores should be in the center-top area
    # Looking at your screenshot: Red 308, Blue 500

    print("\n--- Testing Score Regions ---")

    # Red score region (approximate - adjust as needed)
    red_score_region = img[50:120, 200:350]
    cv2.imwrite("screenshots/red_score_region.png", red_score_region)

    # Blue score region
    blue_score_region = img[50:120, 450:600]
    cv2.imwrite("screenshots/blue_score_region.png", blue_score_region)

    print("\nRed score region:")
    red_score = ocr.extract_score(red_score_region)
    print(f"Extracted: {red_score} (Expected: 308)")

    print("\nBlue score region:")
    blue_score = ocr.extract_score(blue_score_region)
    print(f"Extracted: {blue_score} (Expected: 500)")

    if red_score == 308 and blue_score == 500:
        print("\nSUCCESS! Score extraction is working perfectly!")
    else:
        print("\nScores don't match - may need to adjust regions")
        print("Check screenshots/red_score_region.png and screenshots/blue_score_region.png")


if __name__ == "__main__":
    extract_player_names_from_roster()
    test_score_extraction()
