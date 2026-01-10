"""
Extract player names from GW2 roster using improved OCR techniques.

Based on Tesseract best practices:
- Rescaling to 300+ DPI equivalent
- Proper binarization
- Border addition
- Noise removal
- Correct PSM mode
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.ocr_engine import OCREngine
from config import Config


def add_border(image, border_size=10, border_color=255):
    """Add white border around image to improve OCR."""
    return cv2.copyMakeBorder(
        image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=border_color
    )


def advanced_preprocess(image, scale=3.0):
    """
    Advanced preprocessing based on Tesseract documentation.

    Steps:
    1. Convert to grayscale
    2. Rescale to 300 DPI equivalent
    3. Denoise
    4. Adaptive binarization
    5. Add border
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Rescale (simulate 300+ DPI)
    width = int(gray.shape[1] * scale)
    height = int(gray.shape[0] * scale)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(resized, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold (for white text on dark background, invert)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  # White background
        11, 2
    )

    # Invert if we have white text on dark background
    # Check if more pixels are dark (mean < 128)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    # Add border (helps Tesseract)
    bordered = add_border(binary, border_size=10, border_color=255)

    return bordered


def extract_player_names_improved():
    """
    Extract the 10 player names from roster screenshot.

    Player positions (approximate pixel coordinates):
    Red Team (left column):
    - Player 1: y=100-125
    - Player 2: y=125-150
    - Player 3: y=150-175
    - Player 4: y=175-200
    - Player 5: y=200-225

    Blue Team (right column):
    - Player 6: y=100-125
    - Player 7: y=125-150
    - Player 8: y=150-175
    - Player 9: y=175-200
    - Player 10: y=200-225
    """

    print("=" * 70)
    print("  GW2 Player Name Extraction (Improved OCR)")
    print("=" * 70)

    # Load image
    img = cv2.imread("roster - beginning.PNG")
    if img is None:
        print("Error: Could not load roster - beginning.PNG")
        return

    print(f"\nLoaded image: {img.shape[1]}x{img.shape[0]} pixels")

    # Initialize OCR
    config = Config()
    ocr = OCREngine(tesseract_path=config.get('ocr.tesseract_path'))

    # Define player name regions
    # Adjust these based on visual inspection
    player_regions = {
        # Red team (left side)
        "Red 1": (50, 100, 250, 30),   # x, y, width, height
        "Red 2": (50, 127, 250, 30),
        "Red 3": (50, 147, 250, 30),
        "Red 4": (50, 167, 250, 30),
        "Red 5": (50, 187, 250, 30),
        # Blue team (right side)
        "Blue 1": (900, 100, 300, 30),
        "Blue 2": (900, 127, 300, 30),
        "Blue 3": (900, 147, 300, 30),
        "Blue 4": (900, 167, 300, 30),
        "Blue 5": (900, 187, 300, 30),
    }

    extracted_names = []

    print("\nExtracting player names...\n")

    for player_id, (x, y, w, h) in player_regions.items():
        # Extract region
        region = img[y:y+h, x:x+w]

        # Save original region
        region_path = f"screenshots/player_{player_id.replace(' ', '_')}_original.png"
        cv2.imwrite(region_path, region)

        # Apply advanced preprocessing
        processed = advanced_preprocess(region, scale=3.0)

        # Save processed region
        processed_path = f"screenshots/player_{player_id.replace(' ', '_')}_processed.png"
        cv2.imwrite(processed_path, processed)

        # OCR with PSM 7 (single line)
        text = ocr.extract_text(
            processed,
            psm=7,  # Treat as single line
            whitelist=ocr.name_whitelist,
            preprocess=False  # Already preprocessed
        ).strip()

        team = "Red" if player_id.startswith("Red") else "Blue"
        print(f"{player_id:8} ({team} Team): '{text}'")

        extracted_names.append({
            "position": player_id,
            "team": team,
            "name": text,
            "original_img": region_path,
            "processed_img": processed_path
        })

    print("\n" + "=" * 70)
    print("  Extraction Complete")
    print("=" * 70)

    print("\nReview the processed images in screenshots/ folder:")
    print("  - player_*_original.png: Original cropped regions")
    print("  - player_*_processed.png: After preprocessing")

    # Count successful extractions
    successful = sum(1 for p in extracted_names if p['name'] and len(p['name']) > 2)
    print(f"\nSuccessfully extracted {successful}/10 player names")

    if successful < 10:
        print("\nTip: If extraction is poor, try adjusting:")
        print("  1. Region coordinates (player_regions dict)")
        print("  2. Scaling factor (currently 3.0)")
        print("  3. Binarization threshold parameters")

    return extracted_names


def test_single_name_region():
    """Test extraction on a single name to fine-tune settings."""
    print("=" * 70)
    print("  Testing Single Name Extraction")
    print("=" * 70)

    img = cv2.imread("roster - beginning.PNG")

    # Test on "Legendary L O K I" (Red Team, row 2)
    # Adjust coordinates to precisely capture just the name
    test_region = img[127:157, 50:300]  # y=127-157, x=50-300

    cv2.imwrite("screenshots/test_single_name_original.png", test_region)

    # Try different preprocessing approaches
    config = Config()
    ocr = OCREngine(tesseract_path=config.get('ocr.tesseract_path'))

    print("\nTesting different preprocessing methods on 'Legendary L O K I':\n")

    # Method 1: Advanced preprocessing
    processed1 = advanced_preprocess(test_region, scale=3.0)
    cv2.imwrite("screenshots/test_single_name_method1.png", processed1)
    text1 = ocr.extract_text(processed1, psm=7, preprocess=False)
    print(f"Method 1 (Advanced): '{text1}'")

    # Method 2: Higher scaling
    processed2 = advanced_preprocess(test_region, scale=4.0)
    cv2.imwrite("screenshots/test_single_name_method2.png", processed2)
    text2 = ocr.extract_text(processed2, psm=7, preprocess=False)
    print(f"Method 2 (Scale 4x): '{text2}'")

    # Method 3: Simple grayscale + resize
    gray = cv2.cvtColor(test_region, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    bordered = add_border(resized)
    cv2.imwrite("screenshots/test_single_name_method3.png", bordered)
    text3 = ocr.extract_text(bordered, psm=7, preprocess=False)
    print(f"Method 3 (Simple): '{text3}'")

    # Method 4: PSM 8 (single word)
    text4 = ocr.extract_text(processed1, psm=8, preprocess=False)
    print(f"Method 4 (PSM 8): '{text4}'")

    print("\nExpected: 'Legendary L O K I'")
    print("\nCheck screenshots/test_single_name_*.png to see preprocessing results")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test on single name first
        test_single_name_region()
    else:
        # Extract all 10 names
        extract_player_names_improved()
