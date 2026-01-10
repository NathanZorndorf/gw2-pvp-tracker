"""
OCR engine module for text extraction from screenshots.
Handles image preprocessing and OCR integration (EasyOCR or Tesseract).
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
        engine: str = "easyocr",
        tesseract_path: Optional[str] = None,
        name_whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. ",
        score_whitelist: str = "0123456789",
        use_clahe: bool = True,
        resize_factor: float = 2.0
    ):
        """
        Initialize OCR engine.

        Args:
            engine: OCR engine to use ("easyocr" or "tesseract")
            tesseract_path: Path to tesseract executable
            name_whitelist: Characters allowed in player names
            score_whitelist: Characters allowed in scores
            use_clahe: Whether to use CLAHE contrast enhancement
            resize_factor: Scale factor for OCR preprocessing
        """
        self.engine = engine.lower()
        self.use_clahe = use_clahe
        self.resize_factor = resize_factor
        self.name_whitelist = name_whitelist
        self.score_whitelist = score_whitelist
        self.easyocr_reader = None

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Initialize EasyOCR if selected
        if self.engine == "easyocr":
            self._init_easyocr()

        logger.info(f"OCR engine initialized (engine={self.engine}, clahe={use_clahe})")

    def _init_easyocr(self):
        """Initialize EasyOCR reader (lazy loading)."""
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized")
        except ImportError:
            logger.warning("EasyOCR not installed, falling back to Tesseract")
            self.engine = "tesseract"
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}, falling back to Tesseract")
            self.engine = "tesseract"

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input BGR image

        Returns:
            Contrast-enhanced BGR image
        """
        if len(image.shape) == 2:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

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
        Extract text from image using configured OCR engine.

        Args:
            image: Input image
            psm: Page segmentation mode (7 = single line, Tesseract only)
            whitelist: Character whitelist
            preprocess: Whether to preprocess image

        Returns:
            Extracted text (stripped)
        """
        try:
            # Preprocess image
            if preprocess:
                processed = self._preprocess_for_engine(image)
            else:
                processed = image

            # Use appropriate OCR engine
            if self.engine == "easyocr" and self.easyocr_reader:
                return self._extract_with_easyocr(processed, whitelist)
            else:
                return self._extract_with_tesseract(processed, psm, whitelist)

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def _preprocess_for_engine(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for the current OCR engine."""
        processed = image.copy()

        # Apply CLAHE if enabled
        if self.use_clahe:
            processed = self.apply_clahe(processed)

        # Resize
        if self.resize_factor != 1.0:
            processed = cv2.resize(
                processed, None,
                fx=self.resize_factor, fy=self.resize_factor,
                interpolation=cv2.INTER_CUBIC
            )

        return processed

    def _extract_with_easyocr(
        self,
        image: np.ndarray,
        whitelist: Optional[str] = None
    ) -> str:
        """Extract text using EasyOCR."""
        try:
            results = self.easyocr_reader.readtext(
                image,
                allowlist=whitelist if whitelist else None
            )

            if results:
                # Combine all detected text with confidence > 0.3
                texts = [text for _, text, conf in results if conf > 0.3]
                extracted = ' '.join(texts)
                logger.debug(f"EasyOCR extracted: '{extracted}'")
                return extracted

            return ""

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""

    def _extract_with_tesseract(
        self,
        image: np.ndarray,
        psm: int = 7,
        whitelist: Optional[str] = None
    ) -> str:
        """Extract text using Tesseract."""
        # For Tesseract, apply additional preprocessing
        processed = self.preprocess_for_ocr(image)

        # Build Tesseract config
        config_parts = [f"--psm {psm}"]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        config = " ".join(config_parts)

        # Run OCR
        text = pytesseract.image_to_string(processed, config=config).strip()
        logger.debug(f"Tesseract extracted: '{text}'")

        return text

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

        # Clean up extracted text (remove spaces, non-digits)
        text = ''.join(c for c in text if c.isdigit())

        # Parse integer
        try:
            if not text:
                logger.warning("No digits extracted from score region")
                return None

            score = int(text)

            # Validate GW2 PvP score range
            if validate_range and not (0 <= score <= 500):
                logger.warning(f"Score {score} outside valid range (0-500)")
                return None

            logger.debug(f"Extracted score: {score}")
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

    def extract_player_name_regions(
        self,
        image: np.ndarray,
        regions_config: dict
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Extract individual player name regions from roster screenshot.

        Args:
            image: Full screenshot image
            regions_config: Dictionary with red_team_names and blue_team_names config

        Returns:
            List of (cropped_image, team) tuples for all 10 players
        """
        regions = []

        # Extract red team names
        red_config = regions_config['red_team_names']
        for i in range(red_config['num_players']):
            y_start = red_config['y_start'] + (i * red_config['row_height'])
            y_end = y_start + red_config['row_height']
            x_start = red_config['x_start']
            x_end = red_config['x_end']

            name_region = image[y_start:y_end, x_start:x_end]
            regions.append((name_region, 'red'))

        # Extract blue team names
        blue_config = regions_config['blue_team_names']
        for i in range(blue_config['num_players']):
            y_start = blue_config['y_start'] + (i * blue_config['row_height'])
            y_end = y_start + blue_config['row_height']
            x_start = blue_config['x_start']
            x_end = blue_config['x_end']

            name_region = image[y_start:y_end, x_start:x_end]
            regions.append((name_region, 'blue'))

        logger.debug(f"Extracted {len(regions)} player name regions")
        return regions

    def _calculate_bold_score(self, region: np.ndarray) -> float:
        """
        Calculate bold score using multiple metrics.

        Args:
            region: Image region containing text

        Returns:
            Composite score (0.0-1.0, higher = more bold)
        """
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # Metric 1: Stroke width (white pixel density after thresholding)
        # Bold text has thicker strokes → more white pixels
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(binary == 255) / binary.size

        # Metric 2: Edge density (Canny edge detection)
        # Bold text has more defined edges
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Metric 3: Mean brightness
        # Bold text appears slightly brighter in GW2 UI
        brightness = np.mean(gray) / 255.0

        # Weighted composite score
        score = (white_ratio * 0.4) + (edge_ratio * 0.3) + (brightness * 0.3)

        logger.debug(f"Bold score: {score:.3f} (white={white_ratio:.3f}, edge={edge_ratio:.3f}, bright={brightness:.3f})")
        return score

    def detect_bold_text(
        self,
        name_regions: List[Tuple[np.ndarray, str]]
    ) -> Tuple[int, float]:
        """
        Detect which name region contains bold text (user's character).

        Args:
            name_regions: List of (cropped_image, team) tuples for all 10 players

        Returns:
            Tuple of (index, confidence_score)
            index: Index (0-9) of the bold name region
            confidence_score: Difference between highest and second-highest scores
        """
        # Calculate bold scores for each region
        bold_scores = []
        for region, team in name_regions:
            score = self._calculate_bold_score(region)
            bold_scores.append(score)

        # Find index with highest bold score
        bold_index = int(np.argmax(bold_scores))
        max_score = bold_scores[bold_index]

        # Calculate confidence: difference between highest and second-highest
        sorted_scores = sorted(bold_scores, reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else max_score

        logger.info(f"Bold text detected at index {bold_index} (score: {max_score:.3f}, confidence: {confidence:.3f})")
        logger.debug(f"All scores: {[f'{s:.3f}' for s in bold_scores]}")

        return bold_index, confidence
