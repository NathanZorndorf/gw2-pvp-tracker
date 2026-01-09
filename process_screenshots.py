"""
Process existing GW2 screenshots to test OCR and database functionality.

This script will:
1. Load your roster screenshots
2. Attempt to extract player names using OCR
3. Extract scores from match end screenshot
4. Log the match to the database
5. Display the results
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.models import Database
from vision.ocr_engine import OCREngine
from config import Config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_section(title):
    """Display formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_image(filepath):
    """Load image from file."""
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    logger.info(f"Loaded image: {filepath} ({img.shape[1]}x{img.shape[0]})")
    return img


def extract_roster_manually(image_path, num_players=10):
    """
    Extract player names from roster screenshot.
    This is a simplified version - we'll need to adjust regions based on your screenshots.
    """
    print(f"\nProcessing: {image_path}")
    img = load_image(image_path)

    # Initialize OCR
    config = Config()
    ocr = OCREngine(tesseract_path=config.get('ocr.tesseract_path'))

    # For now, let's just try to extract text from the entire image
    # to see what Tesseract can detect
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")

    # Try OCR on full image first to see what we get
    print("\nAttempting OCR on full image...")
    text = ocr.extract_text(img, psm=6, preprocess=True)  # PSM 6 = uniform block of text

    if text:
        print(f"Detected text:\n{text[:500]}...")  # First 500 chars
    else:
        print("No text detected")

    return text


def manual_test_workflow():
    """
    Manual workflow to test OCR on your screenshots.
    """
    display_section("GW2 Screenshot Processing Test")

    # List available screenshots
    screenshots = [
        "roster - beginning.PNG",
        "roster - end.PNG",
        "full screen - beginning of match.PNG",
        "full screen - end of match.PNG"
    ]

    print("\nAvailable screenshots:")
    for i, name in enumerate(screenshots, 1):
        path = Path(name)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {i}. {name} ({size_mb:.1f} MB)")

    # Process roster screenshots
    display_section("Processing Roster Screenshots")

    # Check if Tesseract is installed
    config = Config()
    tesseract_path = config.get('ocr.tesseract_path')
    print(f"Tesseract path: {tesseract_path}")

    # Test on beginning roster
    try:
        print("\n--- ROSTER BEGINNING ---")
        extract_roster_manually("roster - beginning.PNG")
    except Exception as e:
        print(f"Error processing beginning roster: {e}")

    # Test on end roster
    try:
        print("\n--- ROSTER END ---")
        extract_roster_manually("roster - end.PNG")
    except Exception as e:
        print(f"Error processing end roster: {e}")


def interactive_region_test():
    """
    Interactive tool to help identify regions in your screenshots.
    """
    display_section("Interactive Region Identifier")

    print("\nThis will help you identify the coordinates for:")
    print("  - Player name columns")
    print("  - Score boxes")
    print("  - Profession icons")

    image_path = "roster - beginning.PNG"
    img = load_image(image_path)

    print(f"\nImage dimensions: {img.shape[1]}x{img.shape[0]} (width x height)")
    print("\nTo help locate regions, here's what we can do:")
    print("  1. Create annotated screenshots showing different regions")
    print("  2. Test OCR on specific regions you define")
    print("  3. Save annotated images to help you see what's being processed")

    # Create annotated version
    annotated = img.copy()
    height, width = img.shape[:2]

    # Draw some reference grids
    # Vertical lines every 10%
    for i in range(1, 10):
        x = int(width * i / 10)
        cv2.line(annotated, (x, 0), (x, height), (0, 255, 0), 1)
        cv2.putText(annotated, f"{i*10}%", (x+5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Horizontal lines every 10%
    for i in range(1, 10):
        y = int(height * i / 10)
        cv2.line(annotated, (0, y), (width, y), (0, 255, 0), 1)
        cv2.putText(annotated, f"{i*10}%", (10, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save annotated image
    output_path = "screenshots/annotated_roster_beginning.png"
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved annotated image to: {output_path}")
    print("Open this image to see grid references for defining regions")


def simple_ocr_test():
    """
    Simple test: just try OCR on a small region you can define.
    """
    display_section("Simple OCR Test")

    print("\nLet's test OCR on a small region of your roster screenshot.")
    print("We'll try the top-left corner first to see what Tesseract detects.")

    img = load_image("roster - beginning.PNG")

    # Test a small region (top-left 800x200 pixels)
    test_region = img[0:200, 0:800]

    config = Config()
    ocr = OCREngine(tesseract_path=config.get('ocr.tesseract_path'))

    print("\nTesting region: top-left 800x200 pixels")

    # Save the test region so you can see what we're processing
    cv2.imwrite("screenshots/test_region.png", test_region)
    print("Saved test region to: screenshots/test_region.png")

    # Try OCR
    try:
        text = ocr.extract_text(test_region, psm=6, preprocess=True)
        print(f"\nExtracted text:\n{text}")
    except Exception as e:
        print(f"\nOCR Error: {e}")
        print("\nNote: You may need to install Tesseract OCR:")
        print("  Download: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Install to: C:\\Program Files\\Tesseract-OCR\\")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  GW2 PvP Tracker - Screenshot Processing Tool")
    print("=" * 70)

    print("\nWhat would you like to do?")
    print("  1. Test OCR on full screenshots (see what text is detected)")
    print("  2. Create annotated image with grid (helps identify regions)")
    print("  3. Simple OCR test on small region")
    print("  4. All of the above")

    choice = input("\nEnter choice (1-4) or press Enter for option 4: ").strip()

    if not choice:
        choice = "4"

    if choice in ["1", "4"]:
        manual_test_workflow()

    if choice in ["2", "4"]:
        interactive_region_test()

    if choice in ["3", "4"]:
        simple_ocr_test()

    print("\n" + "=" * 70)
    print("  Processing Complete!")
    print("=" * 70)
    print("\nCheck the screenshots/ folder for generated images.")


if __name__ == "__main__":
    main()
