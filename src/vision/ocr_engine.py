"""
OCR engine module for text extraction from screenshots.
Handles image preprocessing and Tesseract OCR integration.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Optional
from thefuzz import process
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """Handles OCR text extraction with preprocessing."""

    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        name_whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. ",
        score_whitelist: str = "0123456789"
    ):
        """
        Initialize OCR engine.

        Args:
            tesseract_path: Path to tesseract executable
            name_whitelist: Characters allowed in player names
            score_whitelist: Characters allowed in scores
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.name_whitelist = name_whitelist
        self.score_whitelist = score_whitelist
        logger.info("OCR engine initialized")

    def preprocess_for_ocr(
        self,
        image: np.ndarray,
        resize_factor: float = 2.0,
        invert: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image: Input image (BGR or grayscale)
            resize_factor: Scale factor for resizing
            invert: Whether to invert (for white text on dark bg)

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize for better OCR
        if resize_factor != 1.0:
            width = int(gray.shape[1] * resize_factor)
            height = int(gray.shape[0] * resize_factor)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        return denoised

    def extract_text(
        self,
        image: np.ndarray,
        psm: int = 7,
        whitelist: Optional[str] = None,
        preprocess: bool = True
    ) -> str:
        """
        Extract text from image using Tesseract.

        Args:
            image: Input image
            psm: Page segmentation mode (7 = single line)
            whitelist: Character whitelist
            preprocess: Whether to preprocess image

        Returns:
            Extracted text (stripped)
        """
        try:
            # Preprocess if requested
            if preprocess:
                processed = self.preprocess_for_ocr(image)
            else:
                processed = image

            # Build Tesseract config
            config_parts = [f"--psm {psm}"]
            if whitelist:
                config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
            config = " ".join(config_parts)

            # Run OCR
            text = pytesseract.image_to_string(processed, config=config).strip()
            logger.debug(f"OCR extracted: '{text}'")

            return text

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def extract_player_name(
        self,
        image: np.ndarray,
        known_names: Optional[List[str]] = None,
        fuzzy_threshold: int = 80
    ) -> str:
        """
        Extract player name with fuzzy matching correction.

        Args:
            image: Image region containing player name
            known_names: List of known player names for fuzzy matching
            fuzzy_threshold: Minimum similarity score (0-100)

        Returns:
            Extracted and corrected player name
        """
        # OCR with name whitelist
        raw_text = self.extract_text(
            image,
            psm=7,
            whitelist=self.name_whitelist,
            preprocess=True
        )

        # If no known names, return raw OCR
        if not known_names or not raw_text:
            return raw_text

        # Fuzzy match against known names
        match, score = process.extractOne(raw_text, known_names)
        if score >= fuzzy_threshold:
            logger.debug(f"Fuzzy matched '{raw_text}' -> '{match}' (score: {score})")
            return match

        return raw_text

    def extract_score(
        self,
        image: np.ndarray,
        validate_range: bool = True
    ) -> Optional[int]:
        """
        Extract numeric score from image.

        Args:
            image: Image region containing score
            validate_range: Whether to validate score is 0-500

        Returns:
            Extracted score or None if invalid
        """
        # OCR with digits only
        text = self.extract_text(
            image,
            psm=7,
            whitelist=self.score_whitelist,
            preprocess=True
        )

        # Parse integer
        try:
            score = int(text)

            # Validate GW2 PvP score range
            if validate_range and not (0 <= score <= 500):
                logger.warning(f"Score {score} outside valid range (0-500)")
                return None

            return score

        except ValueError:
            logger.warning(f"Could not parse score from text: '{text}'")
            return None

    def extract_player_names_from_roster(
        self,
        roster_image: np.ndarray,
        num_rows: int = 10,
        name_region_x: Tuple[int, int] = (0, 300),
        known_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract all player names from roster.

        Args:
            roster_image: Image of full roster
            num_rows: Number of player rows (default 10)
            name_region_x: (start_x, end_x) for name column
            known_names: Known player names for fuzzy matching

        Returns:
            List of 10 player names
        """
        names = []
        row_height = roster_image.shape[0] // num_rows
        x_start, x_end = name_region_x

        for row_idx in range(num_rows):
            # Extract row region
            y_start = row_idx * row_height
            y_end = y_start + row_height
            name_region = roster_image[y_start:y_end, x_start:x_end]

            # OCR name
            name = self.extract_player_name(
                name_region,
                known_names=known_names
            )

            names.append(name if name else f"Unknown_{row_idx}")
            logger.debug(f"Row {row_idx}: {names[-1]}")

        return names

    def extract_scores_from_boxes(
        self,
        image: np.ndarray,
        blue_box: Tuple[int, int, int, int],
        red_box: Tuple[int, int, int, int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract both team scores from specific regions.

        Args:
            image: Full screenshot
            blue_box: (x, y, width, height) for blue score
            red_box: (x, y, width, height) for red score

        Returns:
            Tuple of (blue_score, red_score)
        """
        # Extract blue score region
        bx, by, bw, bh = blue_box
        blue_region = image[by:by+bh, bx:bx+bw]
        blue_score = self.extract_score(blue_region)

        # Extract red score region
        rx, ry, rw, rh = red_box
        red_region = image[ry:ry+rh, rx:rx+rw]
        red_score = self.extract_score(red_region)

        logger.info(f"Extracted scores: Blue={blue_score}, Red={red_score}")
        return blue_score, red_score

    def extract_with_retry(
        self,
        image: np.ndarray,
        extraction_func,
        max_attempts: int = 3,
        **kwargs
    ):
        """
        Retry extraction with different preprocessing settings.

        Args:
            image: Input image
            extraction_func: Function to call for extraction
            max_attempts: Maximum retry attempts
            **kwargs: Additional arguments for extraction_func

        Returns:
            Result from extraction_func or None
        """
        for attempt in range(max_attempts):
            try:
                # Vary preprocessing on retries
                resize_factor = 2.0 + (attempt * 0.5)  # 2.0, 2.5, 3.0
                processed = self.preprocess_for_ocr(image, resize_factor=resize_factor)

                result = extraction_func(processed, **kwargs)
                if result:
                    return result

            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")

        logger.error("All extraction attempts failed")
        return None

    def validate_extracted_data(
        self,
        names: List[str],
        blue_score: Optional[int],
        red_score: Optional[int]
    ) -> Tuple[bool, List[str]]:
        """
        Validate extracted match data.

        Args:
            names: List of player names
            blue_score: Blue team score
            red_score: Red team score

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check player names
        if len(names) != 10:
            errors.append(f"Expected 10 players, got {len(names)}")

        empty_names = [i for i, name in enumerate(names) if not name or name.startswith("Unknown_")]
        if empty_names:
            errors.append(f"Missing names at rows: {empty_names}")

        # Check for duplicate names
        if len(names) != len(set(names)):
            errors.append("Duplicate player names detected")

        # Check scores
        if blue_score is None:
            errors.append("Blue score not extracted")
        if red_score is None:
            errors.append("Red score not extracted")

        if blue_score is not None and red_score is not None:
            if blue_score + red_score == 0:
                errors.append("Both scores are zero")

        is_valid = len(errors) == 0
        return is_valid, errors
