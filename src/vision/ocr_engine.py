"""
OCR engine module for text extraction from screenshots.
Handles image preprocessing and OCR integration (EasyOCR or Tesseract).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """Handles OCR text extraction with preprocessing."""

    def __init__(
        self,
        name_whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. ",
        score_whitelist: str = "0123456789",
        use_clahe: bool = True,
        resize_factor: float = 2.0
    ):
        """
        Initialize OCR engine.

        Args:
            name_whitelist: Characters allowed in player names
            score_whitelist: Characters allowed in scores
            use_clahe: Whether to use CLAHE contrast enhancement
            resize_factor: Scale factor for OCR preprocessing
        """
        self.use_clahe = use_clahe
        self.resize_factor = resize_factor
        self.name_whitelist = name_whitelist
        self.score_whitelist = score_whitelist
        self.easyocr_reader = None

        self._init_easyocr()

        logger.info(f"OCR engine initialized (clahe={use_clahe})")

    def _init_easyocr(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            # Use multiple languages to support accented characters (ô, ë, ä, ö, á, etc.)
            # Common in GW2 player names
            self.easyocr_reader = easyocr.Reader(
                ['en', 'fr', 'de', 'es', 'pt'],
                gpu=True,
                verbose=False
            )
            logger.info("EasyOCR reader initialized with multi-language support (GPU enabled)")
        except ImportError:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")

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
        whitelist: Optional[str] = None,
        preprocess: bool = True
    ) -> str:
        """
        Extract text from image using EasyOCR.

        Args:
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

            # Use EasyOCR
            if self.easyocr_reader:
                return self._extract_with_easyocr(processed, whitelist)
            else:
                logger.error("EasyOCR reader not initialized")
                return ""

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

    def extract_player_name(
        self,
        image: np.ndarray
    ) -> str:
        """
        Extract player name from image.

        Args:
            image: Image region containing player name

        Returns:
            Extracted player name
        """
        # OCR with name whitelist
        raw_text = self.extract_text(
            image,
            whitelist=self.name_whitelist,
            preprocess=True
        )

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

            # Validate GW2 PvP score (allow values above 500 — some modes go higher)
            if validate_range:
                if score < 0:
                    logger.warning(f"Score {score} is negative, invalid")
                    return None
                # Accept reasonably large scores; reject obviously garbage values
                if score > 9999:
                    logger.warning(f"Score {score} outside reasonable range (0-9999)")
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
                name_region
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
        regions_config: dict,
        arena_type: Optional[str] = None
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Extract individual player name regions from roster screenshot.

        Args:
            image: Full screenshot image
            regions_config: Dictionary with roster region config. Can be either:
                - New format: {'ranked': {...}, 'unranked': {...}}
                - Legacy format: {'red_team_names': {...}, 'blue_team_names': {...}}
            arena_type: Optional arena type ('ranked' or 'unranked'). If None and
                using new config format, will auto-detect.

        Returns:
            List of (cropped_image, team) tuples for all 10 players
        """
        regions = []

        # Determine which config to use
        if 'ranked' in regions_config or 'unranked' in regions_config:
            # New format with arena-specific regions
            if arena_type is None:
                # Auto-detect arena type
                arena_type, _ = self.detect_arena_type(image, regions_config)

            if arena_type in regions_config:
                config = regions_config[arena_type]
            else:
                # Fallback to default
                default = regions_config.get('default_arena', 'unranked')
                config = regions_config.get(default, regions_config.get('unranked', {}))
                logger.warning(f"Arena type '{arena_type}' not found, using {default}")
        else:
            # Legacy format - use directly
            config = regions_config

        # Extract red team names
        red_config = config.get('red_team_names', {})
        if red_config:
            for i in range(red_config.get('num_players', 5)):
                y_start = red_config['y_start'] + (i * red_config['row_height'])
                y_end = y_start + red_config['row_height']
                x_start = red_config['x_start']
                x_end = red_config['x_end']

                name_region = image[y_start:y_end, x_start:x_end]
                regions.append((name_region, 'red'))

        # Extract blue team names
        blue_config = config.get('blue_team_names', {})
        if blue_config:
            for i in range(blue_config.get('num_players', 5)):
                y_start = blue_config['y_start'] + (i * blue_config['row_height'])
                y_end = y_start + blue_config['row_height']
                x_start = blue_config['x_start']
                x_end = blue_config['x_end']

                name_region = image[y_start:y_end, x_start:x_end]
                regions.append((name_region, 'blue'))

        logger.debug(f"Extracted {len(regions)} player name regions (arena: {arena_type})")
        return regions

    def _calculate_bold_score(self, region: np.ndarray) -> float:
        """
        Calculate bold score by detecting gold/bright text (user highlight).

        In GW2, the user's name is displayed in gold/bright white color 
        while other players have gray text. This method detects those highlights.

        Args:
            region: Image region containing text

        Returns:
            Score (higher = more likely user's name)
        """
        if len(region.shape) != 3:
            return 0.0

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 1. Gold/Yellow Detection
        # Hue: 20-35 (Yellow/Gold)
        # Saturation: > 50 (Not white)
        # Value: > 150 (Bright)
        gold_lower = np.array([20, 50, 150])
        gold_upper = np.array([35, 255, 255])
        gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)
        gold_ratio = np.sum(gold_mask > 0) / gold_mask.size

        # 2. Brightness/Whiteness Detection
        # User name is significantly brighter than others (max V ~255 vs ~187)
        _, _, v = cv2.split(hsv)
        max_v = np.max(v)
        
        # Legacy: Cyan check (kept with low weight just in case)
        cyan_lower = np.array([80, 30, 100])
        cyan_upper = np.array([110, 255, 255])
        cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)
        cyan_ratio = np.sum(cyan_mask > 0) / cyan_mask.size
        
        # Scoring
        brightness_score = 0.0
        if max_v > 240:
            brightness_score = 5.0  # Strong signal for bright highlight
        elif max_v > 200:
            brightness_score = 0.5  # Weak signal
            
        gold_score = gold_ratio * 10  # Scale up gold presence
        
        # Combined score: Favor Gold or High Brightness
        final_score = max(gold_score, brightness_score, cyan_ratio * 2)

        logger.debug(f"Highlight score: {final_score:.4f} (gold={gold_ratio:.4f}, max_v={max_v}, cyan={cyan_ratio:.4f})")
        return final_score

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

    def detect_arena_type(
        self,
        image: np.ndarray,
        roster_regions: dict,
        detection_region: Optional[dict] = None
    ) -> Tuple[Optional[str], float]:
        """
        Detect arena type (ranked vs unranked) by testing OCR with both region sets.

        The method extracts player name regions using both ranked and unranked
        bounding boxes, runs OCR on each, and determines which produces more
        valid text results.

        Args:
            image: Full screenshot image
            roster_regions: Config dict containing 'ranked' and 'unranked' region sets
            detection_region: Optional region dict {'x', 'y', 'width', 'height'} to OCR first

        Returns:
            Tuple of (arena_type, confidence)
            arena_type: "ranked" or "unranked"
            confidence: Score difference (higher = more confident)
        """
        # 1. Try explicit detection region if provided
        if detection_region:
            try:
                x, y = detection_region['x'], detection_region['y']
                w, h = detection_region['width'], detection_region['height']
                
                # Check bounds
                if y + h <= image.shape[0] and x + w <= image.shape[1]:
                    roi = image[y:y+h, x:x+w]
                    # Use Tesseract normally or specific settings if needed
                    # We don't use strict whitelist here as we look for "Ranked Arena" or "Unranked Arena"
                    text = self.extract_text(roi, preprocess=True).lower()
                    logger.info(f"Arena detection ROI text: '{text}'")
                    
                    if 'unranked' in text:
                        return 'unranked', 1.0
                    if 'ranked' in text: # "Ranked Arena" or just "Ranked"
                        return 'ranked', 1.0
            except Exception as e:
                logger.warning(f"Failed to use detection region: {e}")

        # 2. Fallback to legacy heuristic method
        def _run_pass(preprocess_flag: bool) -> dict:
            scores = {}
            for arena_type in ['ranked', 'unranked']:
                if arena_type not in roster_regions:
                    continue

                regions_config = roster_regions[arena_type]
                total_score = 0.0
                valid_extractions = 0

                # Extract and OCR red team names
                red_cfg = regions_config['red_team_names']
                for i in range(red_cfg['num_players']):
                    y_start = red_cfg['y_start'] + (i * red_cfg['row_height'])
                    y_end = y_start + red_cfg['row_height']
                    x_start = red_cfg['x_start']
                    x_end = red_cfg['x_end']

                    # Bounds check
                    if y_end > image.shape[0] or x_end > image.shape[1]:
                        continue

                    region = image[y_start:y_end, x_start:x_end]
                    text = self.extract_text(region, whitelist=self.name_whitelist, preprocess=preprocess_flag)

                    # Score based on text quality
                    if text and len(text.strip()) >= 2:
                        valid_extractions += 1
                        # Bonus for longer names (more likely valid)
                        total_score += min(len(text.strip()) / 15.0, 1.0)

                # Extract and OCR blue team names
                blue_cfg = regions_config['blue_team_names']
                for i in range(blue_cfg['num_players']):
                    y_start = blue_cfg['y_start'] + (i * blue_cfg['row_height'])
                    y_end = y_start + blue_cfg['row_height']
                    x_start = blue_cfg['x_start']
                    x_end = blue_cfg['x_end']

                    # Bounds check
                    if y_end > image.shape[0] or x_end > image.shape[1]:
                        continue

                    region = image[y_start:y_end, x_start:x_end]
                    text = self.extract_text(region, whitelist=self.name_whitelist, preprocess=preprocess_flag)

                    if text and len(text.strip()) >= 2:
                        valid_extractions += 1
                        total_score += min(len(text.strip()) / 15.0, 1.0)

                scores[arena_type] = (valid_extractions, total_score)
                logger.debug(f"Arena detection - {arena_type}: {valid_extractions} valid names, score={total_score:.2f}")

            return scores

        # Run detection with full preprocessing for accuracy
        arena_scores = _run_pass(preprocess_flag=True)

        # Determine winner from first pass
        if not arena_scores:
            default = roster_regions.get('default_arena', 'unranked')
            logger.warning(f"No arena regions found, using default: {default}")
            return default, 0.0

        # Compare by valid extractions first, then by total score
        ranked_valid, ranked_score = arena_scores.get('ranked', (0, 0))
        unranked_valid, unranked_score = arena_scores.get('unranked', (0, 0))

        if ranked_valid == 0 and unranked_valid == 0:
            logger.warning("Arena detection failed: No valid player names found in either layout.")
            return None, 0.0

        if ranked_valid > unranked_valid:
            confidence = (ranked_valid - unranked_valid) / 10.0
            arena_choice = 'ranked'
        elif unranked_valid > ranked_valid:
            confidence = (unranked_valid - ranked_valid) / 10.0
            arena_choice = 'unranked'
        else:
            # Tie on valid extractions, use total score
            if ranked_score > unranked_score:
                confidence = (ranked_score - unranked_score) / 10.0
                arena_choice = 'ranked'
            else:
                confidence = (unranked_score - ranked_score) / 10.0
                arena_choice = 'unranked'

        logger.info(f"Arena type detected: {arena_choice} (confidence: {confidence:.2f})")
        return arena_choice, confidence
