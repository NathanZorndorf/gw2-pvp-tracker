"""
Profession detector module for icon-based profession recognition.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from config import Config

logger = logging.getLogger(__name__)


class ProfessionDetector:
    """Detects player professions from scoreboard icons using template matching."""

    REFERENCE_ICON_SIZE = (32, 32)

    def __init__(self, config: Config):
        """
        Initialize profession detector.

        Args:
            config: Configuration object
        """
        self.config = config
        self.enabled = config.get('profession_detection.enabled', True)

        # Get preprocessing parameters
        preproc = config.get('profession_detection.preprocessing', {})
        self.clahe_clip = preproc.get('clahe_clip', 1.0)
        self.clahe_tile = tuple(preproc.get('clahe_tile', [2, 2]))
        self.canny_min = preproc.get('canny_min', 50)
        self.canny_max = preproc.get('canny_max', 150)
        self.matching_method = config.get('profession_detection.matching_method', 'template')
        self.confidence_threshold = config.get('profession_detection.confidence_threshold', 0.5)

        # Load and preprocess reference icons
        self.reference_icons: Dict[str, np.ndarray] = {}
        self.reference_dist_maps: Dict[str, np.ndarray] = {}
        if self.enabled:
            self._load_reference_icons()
            logger.info(f"Loaded {len(self.reference_icons)} reference profession icons (method: {self.matching_method})")

    def _load_reference_icons(self) -> None:
        """Load and preprocess all reference icons from icons-white directory."""
        icons_path = Path(self.config.get(
            'profession_detection.reference_icons_path',
            'data/reference-icons/icons-white'
        ))

        if not icons_path.exists():
            logger.warning(f"Reference icons path not found: {icons_path}")
            return

        # Load all PNG files
        for icon_file in icons_path.glob('*.png'):
            try:
                # Load icon
                icon = cv2.imread(str(icon_file))
                if icon is None:
                    logger.warning(f"Failed to load icon: {icon_file}")
                    continue

                # Preprocess the reference icon
                preprocessed = self._preprocess_icon(icon)

                # Store with profession name (filename without extension)
                profession_name = icon_file.stem
                self.reference_icons[profession_name] = preprocessed

                # If using chamfer matching, pre-calculate distance transform map
                if self.matching_method == 'chamfer':
                    # Convert to binary
                    if len(preprocessed.shape) == 3:
                        gray = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = preprocessed
                    
                    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    # DIST_L2 distance transform. Distances are calculated from non-zero (white) pixels.
                    # We want distances from edges, so we invert it.
                    dist_map = cv2.distanceTransform(cv2.bitwise_not(binary), cv2.DIST_L2, 3)
                    self.reference_dist_maps[profession_name] = dist_map

            except Exception as e:
                logger.error(f"Error loading reference icon {icon_file}: {e}")

    def _letterbox_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image preserving aspect ratio with letterboxing.

        Args:
            img: Input image
            target_size: Target size (width, height)

        Returns:
            Letterboxed image
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor (fit inside box)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize preserving aspect ratio
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create black canvas
        if len(img.shape) == 3:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)

        # Center the resized image on canvas
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def _preprocess_icon(self, icon: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply CLAHE + Canny preprocessing pipeline.

        Based on grid-searched optimal parameters achieving 92% accuracy:
        - CLAHE clipLimit=1.0, tileGridSize=(2,2)
        - Canny min=50, max=150

        Args:
            icon: Input icon image
            target_size: Optional target size override (width, height). 
                        Defaults to REFERENCE_ICON_SIZE if None.

        Returns:
            Preprocessed icon (RGB format for consistency)
        """
        if target_size is None:
            target_size = self.REFERENCE_ICON_SIZE

        # 1. Convert to grayscale
        if len(icon.shape) == 3:
            gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        else:
            gray = icon.copy()

        # 2. Resize with aspect-ratio-preserving letterbox
        gray = self._letterbox_image(gray, target_size)

        # 3. Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
        enhanced = clahe.apply(gray)

        # 4. Extract edges with Canny
        edges = cv2.Canny(enhanced, self.canny_min, self.canny_max)

        # 5. Apply circular mask to remove corner noise
        h, w = edges.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        cv2.circle(mask, center, radius, 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        # 6. Convert back to RGB for template matching consistency
        rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return rgb

    def _extract_icon_regions(
        self,
        image: np.ndarray,
        arena_type: str
    ) -> List[Tuple[np.ndarray, str, int]]:
        """
        Extract 10 icon regions from screenshot.

        Args:
            image: Full screenshot image
            arena_type: 'ranked' or 'unranked'

        Returns:
            List of (icon_image, team, player_index) tuples
            player_index is 0-4 for each team
        """
        icon_regions = []

        # Get icon region config for this arena type
        regions_config = self.config.get('roster_regions', {})
        if arena_type not in regions_config:
            logger.error(f"No icon regions configured for arena type: {arena_type}")
            return icon_regions

        icon_config = regions_config[arena_type].get('icon_regions')
        if not icon_config:
            logger.error(f"No icon_regions in config for arena type: {arena_type}")
            return icon_regions

        # Process red team and blue team
        for team_name in ['red_team', 'blue_team']:
            team_cfg = icon_config.get(team_name)
            if not team_cfg:
                logger.warning(f"No config for {team_name}")
                continue

            # Extract rectangle region containing all 5 icons
            x = team_cfg['x']
            y = team_cfg['y']
            width = team_cfg['width']
            height = team_cfg['height']

            rect_region = image[y:y + height, x:x + width].copy()

            # Split into 5 icons vertically
            num_icons = 5
            icon_height = height // num_icons
            square_size = min(width, icon_height)

            team = 'red' if team_name == 'red_team' else 'blue'

            for i in range(num_icons):
                # Calculate vertical position
                y_start = i * icon_height
                y_end = y_start + square_size

                # Center horizontally
                x_start = (width - square_size) // 2
                x_end = x_start + square_size

                # Crop the square icon
                icon_img = rect_region[y_start:y_end, x_start:x_end].copy()

                icon_regions.append((icon_img, team, i))

        return icon_regions

    def _match_template(self, target: np.ndarray, reference: np.ndarray) -> float:
        """
        Compare preprocessed icons using template matching.

        Args:
            target: Preprocessed target icon
            reference: Preprocessed reference icon

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Convert to grayscale for matching
        if len(target.shape) == 3:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target

        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference

        # Use TM_CCOEFF_NORMED (best performing method from grid search)
        result = cv2.matchTemplate(target_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return float(max_val)

    def _match_chamfer(self, target: np.ndarray, pref_dist_map: np.ndarray) -> float:
        """
        Compare target icon against reference distance map using Chamfer matching.

        Args:
            target: Preprocessed target icon (RGB)
            pref_dist_map: Distance transform map of reference icon

        Returns:
            Similarity score normalized to (0.0 to 1.0), higher is better
        """
        # 1. Convert target to binary edges
        if len(target.shape) == 3:
            target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target
        
        _, binary = cv2.threshold(target_gray, 1, 255, cv2.THRESH_BINARY)
        
        # 2. Sum distances for all edge pixels in target
        edge_pixels = binary > 0
        num_edge_pixels = np.sum(edge_pixels)
        
        if num_edge_pixels == 0:
            return 0.0
            
        total_distance = np.sum(pref_dist_map[edge_pixels])
        avg_distance = total_distance / num_edge_pixels
        
        # 3. Normalize to 0.0-1.0 (Exp decay or simple inverse)
        # Using 1 / (1 + avg_distance) so a perfect match (avg=0) is 1.0
        # and avg distance of 1px is 0.5, 2px is 0.33, etc.
        similarity = 1.0 / (1.0 + (avg_distance * 0.5)) # Scaled slightly to be more forgiving
        
        return float(similarity)

    def detect_professions(
        self,
        image: np.ndarray,
        arena_type: str
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: detect all 10 player professions from screenshot.

        Args:
            image: Full match screenshot image
            arena_type: 'ranked' or 'unranked'

        Returns:
            List of profession detection results:
            [{
                'profession': str,
                'confidence': float,
                'team': str ('red' or 'blue'),
                'player_index': int (0-4)
            }, ...]
        """
        if not self.enabled:
            logger.warning("Profession detection is disabled in config")
            return []

        if not self.reference_icons:
            logger.warning("No reference icons loaded")
            return []

        results = []

        try:
            # Extract icon regions
            icon_regions = self._extract_icon_regions(image, arena_type)

            if not icon_regions:
                logger.warning("No icon regions extracted")
                return []

            # Process each icon
            for icon_img, team, player_idx in icon_regions:
                # Preprocess target icon
                # We use a slightly larger 36x36 window for the target to allow 
                # template matching to find the best alignment (translation robust)
                search_size = (36, 36) if self.matching_method == 'template' else self.REFERENCE_ICON_SIZE
                preprocessed = self._preprocess_icon(icon_img, target_size=search_size)

                # Match against all reference icons
                best_profession = 'Unknown'
                best_score = 0.0

                for profession_name, ref_icon in self.reference_icons.items():
                    if self.matching_method == 'chamfer' and profession_name in self.reference_dist_maps:
                        score = self._match_chamfer(preprocessed, self.reference_dist_maps[profession_name])
                    else:
                        score = self._match_template(preprocessed, ref_icon)

                    if score > best_score:
                        best_score = score
                        best_profession = profession_name

                # Only accept if above confidence threshold
                if best_score < self.confidence_threshold:
                    best_profession = 'Unknown'
                    logger.debug(
                        f"Low confidence for {team} player {player_idx}: "
                        f"best={best_profession} score={best_score:.3f}"
                    )

                results.append({
                    'profession': best_profession,
                    'confidence': best_score,
                    'team': team,
                    'player_index': player_idx
                })

                logger.debug(
                    f"Detected {team} player {player_idx}: "
                    f"{best_profession} (conf={best_score:.3f})"
                )

            logger.info(f"Profession detection complete: {len(results)} players processed")

        except Exception as e:
            logger.error(f"Profession detection failed: {e}", exc_info=True)

        return results
