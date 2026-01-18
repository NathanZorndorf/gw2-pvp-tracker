"""
Match processor module for automated OCR extraction and database logging.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

from vision.ocr_engine import OCREngine
from vision.profession_detector import ProfessionDetector
from database.models import Database
from integration.mumble_link import MumbleLink
from config import Config

logger = logging.getLogger(__name__)


class MatchProcessor:
    """Processes match screenshots and extracts all data for logging."""

    def __init__(self, config: Config, db: Database):
        """
        Initialize match processor.

        Args:
            config: Configuration object
            db: Database object
        """
        self.config = config
        self.db = db
        self.debug_enabled = config.get('debug.save_regions', True)
        self.debug_dir = Path(config.get('debug.output_dir', 'data/debug'))

        # Initialize OCR engine with new settings
        self.ocr = OCREngine(
            name_whitelist=config.get('ocr.name_whitelist'),
            score_whitelist=config.get('ocr.score_whitelist'),
            use_clahe=config.get('ocr.use_clahe', True),
            resize_factor=config.get('ocr.resize_factor', 2.0)
        )

        # Initialize profession detector
        self.profession_detector = ProfessionDetector(config)

        # Initialize MumbleLink
        self.mumble_link = MumbleLink('data/map_ids.csv')

        # Arena type detected from current match (set during processing)
        self._current_arena_type: Optional[str] = None

        # Create debug directory if enabled
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug mode enabled. Saving regions to: {self.debug_dir}")

    def _get_arena_config(self, arena_type: Optional[str] = None) -> dict:
        """
        Get the region config for the specified or detected arena type.

        Args:
            arena_type: 'ranked', 'unranked', or None to use detected type

        Returns:
            Region config dict for the arena type
        """
        regions_config = self.config.get('roster_regions')

        # Use provided arena_type, or fall back to detected, or default
        arena = arena_type or self._current_arena_type
        if arena is None:
            arena = regions_config.get('default_arena', 'unranked')

        # Check if using new format with arena-specific regions
        if arena in regions_config:
            return regions_config[arena]
        else:
            # Legacy format - return the whole config
            return regions_config

    def detect_arena_type(self, image: np.ndarray) -> Optional[str]:
        """
        Detect and store the arena type from a screenshot.

        Args:
            image: Screenshot image

        Returns:
            Detected arena type ('ranked' or 'unranked') or None if detection failed
        """
        regions_config = self.config.get('roster_regions')
        detection_region = self.config.get('arena_type_detection')
        
        arena_type, confidence = self.ocr.detect_arena_type(image, regions_config, detection_region)
        
        if arena_type is None:
            logger.warning("Arena type detection failed (confidence: 0.0)")
            self._current_arena_type = None
            return None

        self._current_arena_type = arena_type
        logger.info(f"Arena type set to: {arena_type} (confidence: {confidence:.2f})")
        return arena_type

    def _save_debug_region(
        self,
        image: np.ndarray,
        name: str,
        timestamp: str
    ) -> str:
        """
        Save a debug region image.

        Args:
            image: Region image to save
            name: Descriptive name for the region
            timestamp: Timestamp string for grouping

        Returns:
            Path to saved file
        """
        if not self.debug_enabled:
            return ""

        filename = f"{timestamp}_{name}.png"
        filepath = self.debug_dir / filename
        cv2.imwrite(str(filepath), image)
        logger.debug(f"Saved debug region: {filepath}")
        return str(filepath)

    def detect_user_from_image(self, image_path: Optional[str] = None, image: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        Detect user character from screenshot (for F8/F9).

        Args:
            image_path: Path to screenshot image

        Returns:
            Tuple of (character_name, team, confidence_score)
        """
        try:
            # Load image if not provided
            if image is None:
                if not image_path:
                    logger.error("No image or image_path provided to detect_user_from_image")
                    return None, None, 0.0
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return None, None, 0.0

            # Detect arena type from this screenshot only if not already detected
            if self._current_arena_type is None:
                self.detect_arena_type(image)

            # Try to get user name from MumbleLink first
            self.mumble_link.read()
            mumble_name = None
            if self.mumble_link.is_active:
                identity = self.mumble_link.get_identity()
                mumble_name = identity.get('name')
                if mumble_name:
                    logger.info(f"MumbleLink detected user name: {mumble_name}")

            # Extract player name regions using detected arena type
            regions_config = self.config.get('roster_regions')
            name_regions = self.ocr.extract_player_name_regions(
                image, regions_config, arena_type=self._current_arena_type
            )
            
            # If we have MumbleLink name, find which team they are on
            if mumble_name:
                # We need to run OCR on the regions to find which one matches MumbleLink name
                # Or we can just run a quick check if we assume the user is rendering the MumbleLink data? NO.
                # We need to match MumbleLink name to one of the regions to ID the team.
                
                # Check matching against all regions
                # This could be expensive, so we only do it if we have the name
                
                # Extract text from all regions
                extracted_names = []
                for idx, (region, team) in enumerate(name_regions):
                    text = self.ocr.extract_text(region, preprocess=True)
                    if text:
                        extracted_names.append((text, team, idx))
                
                # Find best match for mumble_name
                if extracted_names:
                    # Simple exact-match check (case-insensitive)
                    for text, team, idx in extracted_names:
                        if text.lower() == mumble_name.lower():
                            logger.info(f"Matched MumbleLink name '{mumble_name}' to OCR '{text}' on {team} team")
                            return mumble_name, team, 1.0
                    
                    # Substring check as secondary (e.g. OCR missed a letter)
                    for text, team, idx in extracted_names:
                        if mumble_name.lower() in text.lower() or text.lower() in mumble_name.lower():
                            logger.info(f"Partial match MumbleLink name '{mumble_name}' to OCR '{text}' on {team} team")
                            return mumble_name, team, 0.9

            # Fallback to Bold Text Detection if MumbleLink failed or couldn't be matched
            # Detect bold text
            bold_index, confidence = self.ocr.detect_bold_text(name_regions)

            # Extract the bold player's name
            bold_region, team = name_regions[bold_index]

            # OCR the name
            user_name = self.ocr.extract_player_name(bold_region)

            if not user_name:
                logger.warning("Failed to extract bold player name")
                return None, None, confidence

            logger.info(f"Detected user: {user_name} on {team} team (confidence: {confidence:.3f})")
            return user_name, team, confidence

        except Exception as e:
            logger.error(f"Error detecting user from image: {e}")
            return None, None, 0.0

    def process_match(
        self,
        start_path: str,
        end_path: str,
        detected_user: Optional[Tuple[str, str]] = None,
        map_name: Optional[str] = None,
        known_players: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Extract all match data from screenshots.

        Args:
            start_path: Path to match start screenshot
            end_path: Path to match end screenshot
            detected_user: Optional (char_name, team) from F8 detection
            map_name: Optional map name override
            known_players: Optional list of manually corrected player data from F8 overlay

        Returns:
            Dict with extracted data or error information
        """
        try:
            # Generate timestamp for debug images
            self._debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Load screenshots
            end_img = cv2.imread(end_path)
            if end_img is None:
                raise ValueError(f"Failed to load end screenshot: {end_path}")

            start_img = cv2.imread(start_path) if start_path else None

            # Detect arena type if not already detected (from F8)
            # Prefer start screenshot for detection since it has cleaner player names
            if self._current_arena_type is None:
                detection_img = start_img if start_img is not None else end_img
                detected_type = self.detect_arena_type(detection_img)
                
                if detected_type is None:
                    error_msg = "Arena type detection failed - cannot proceed with extraction. Check your resolution settings."
                    logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg
                    }

            # Extract scores (from end screenshot)
            red_score, blue_score = self._extract_scores(end_img, "end")

            # Extract all player names (from start screenshot for better accuracy)
            # If we have known players (from F8 overlay), use them but refresh winrates/stats if needed
            # Or just use them as the source of truth for name/profession/team
            if known_players:
                logger.info("Using metadata from F8 overlay (manually verified/corrected)")
                players_data = known_players
            elif start_img is not None:
                players_data = self._extract_all_players(start_img, "start")
            else:
                players_data = self._extract_all_players(end_img, "end")

            # Detect user character (use F8 detection if available, otherwise detect from F9)
            if detected_user:
                user_char_name, user_team = detected_user
                confidence = 1.0  # High confidence since it came from F8
                logger.info(f"Using F8 detected user: {user_char_name} ({user_team})")
            else:
                logger.info("No F8 detection available, detecting from F9 screenshot")
                user_char_name, user_team, confidence = self._detect_user_character(end_img, players_data)

            # Validate extraction
            is_valid, errors = self._validate_extraction(
                red_score, blue_score, players_data, user_char_name
            )

            # Get Map Name from MumbleLink if not provided and active
            if not map_name:
                self.mumble_link.read()
                if self.mumble_link.is_active:
                    map_name = self.mumble_link.get_map_name()
            
            if not map_name:
                map_name = "Unknown"

            return {
                'success': True,
                'red_score': red_score if red_score is not None else 0,
                'blue_score': blue_score if blue_score is not None else 0,
                'user_character': user_char_name if user_char_name else 'Unknown',
                'user_team': user_team if user_team else 'red',
                'map_name': map_name,
                'players': players_data,
                'validation_errors': errors,
                'is_valid': is_valid,
                'user_confidence': confidence,
                'arena_type': self._current_arena_type
            }

        except Exception as e:
            logger.error(f"Match processing failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_scores(self, image: np.ndarray, source: str = "end") -> Tuple[int, int]:
        """
        Extract red and blue scores from screenshot.

        Args:
            image: Full screenshot image
            source: Source identifier for debug images ("start" or "end")

        Returns:
            Tuple of (red_score, blue_score)
        """
        try:
            # Get arena-specific config
            arena_config = self._get_arena_config()

            # Extract score regions
            red_box = arena_config['red_score_box']
            blue_box = arena_config['blue_score_box']

            # Extract the actual image regions
            red_region = image[
                red_box['y']:red_box['y']+red_box['height'],
                red_box['x']:red_box['x']+red_box['width']
            ]
            blue_region = image[
                blue_box['y']:blue_box['y']+blue_box['height'],
                blue_box['x']:blue_box['x']+blue_box['width']
            ]

            # Save debug images
            if self.debug_enabled and hasattr(self, '_debug_timestamp'):
                self._save_debug_region(red_region, f"{source}_score_red", self._debug_timestamp)
                self._save_debug_region(blue_region, f"{source}_score_blue", self._debug_timestamp)

            # Extract scores using OCR
            red_score = self.ocr.extract_score(red_region)
            blue_score = self.ocr.extract_score(blue_region)

            logger.info(f"Extracted scores: Red={red_score}, Blue={blue_score} (arena: {self._current_arena_type})")
            return red_score if red_score is not None else 0, blue_score if blue_score is not None else 0

        except Exception as e:
            logger.error(f"Score extraction failed: {e}")
            return 0, 0

    def _detect_professions_from_image(self, image: np.ndarray) -> List[str]:
        """
        Detect professions from match screenshot.

        Args:
            image: Full screenshot image

        Returns:
            List of 10 profession names in player order (red 0-4, blue 0-4)
        """
        if not self.config.get('profession_detection.enabled', True):
            return ['Unknown'] * 10

        try:
            results = self.profession_detector.detect_professions(
                image,
                arena_type=self._current_arena_type
            )

            # Initialize with Unknown
            professions = ['Unknown'] * 10

            # Map results to player indices
            # Red team: indices 0-4, Blue team: indices 5-9
            for result in results:
                player_idx = result['player_index']
                if result['team'] == 'red':
                    idx = player_idx
                elif result['team'] == 'blue':
                    idx = player_idx + 5
                else:
                    continue

                if 0 <= idx < 10:
                    confidence_threshold = self.config.get('profession_detection.confidence_threshold', 0.5)
                    if result['confidence'] >= confidence_threshold:
                        professions[idx] = result['profession']
                    else:
                        logger.debug(
                            f"Low confidence for {result['team']} player {player_idx}: "
                            f"{result['profession']} ({result['confidence']:.3f})"
                        )

            return professions

        except Exception as e:
            logger.warning(f"Profession detection failed: {e}")
            return ['Unknown'] * 10

    def _extract_all_players(self, image: np.ndarray, source: str = "end") -> List[Dict]:
        """
        Extract all 10 player names and professions.

        Args:
            image: Full screenshot image
            source: Source identifier for debug images ("start" or "end")

        Returns:
            List of {name, profession, team} dicts
        """
        players = []

        try:
            regions_config = self.config.get('roster_regions')
            name_regions = self.ocr.extract_player_name_regions(
                image, regions_config, arena_type=self._current_arena_type
            )

            # Detect professions from image
            professions = self._detect_professions_from_image(image)

            # Preprocess each name region once (reduce repeated expensive ops)
            preprocessed_regions = []
            for region, team in name_regions:
                try:
                    processed = self.ocr._preprocess_for_engine(region)
                except Exception:
                    processed = region
                preprocessed_regions.append((processed, team))

            # Extract each player's name from preprocessed crops
            for idx, (region, team) in enumerate(preprocessed_regions):
                # Save debug image for this player region
                if self.debug_enabled and hasattr(self, '_debug_timestamp'):
                    player_num = idx + 1 if team == 'red' else idx - 4  # 1-5 for each team
                    self._save_debug_region(
                        region,
                        f"{source}_name_{team}_{player_num}",
                        self._debug_timestamp
                    )

                # Run OCR on the already-preprocessed crop (skip internal preprocessing)
                name = self.ocr.extract_text(
                    region,
                    whitelist=self.ocr.name_whitelist,
                    preprocess=False
                )

                if not name:
                    name = f"Unknown_{idx}"
                    logger.warning(f"Failed to extract name for player {idx}, using {name}")

                # Get profession for this player
                profession = professions[idx] if idx < len(professions) else 'Unknown'

                players.append({
                    'name': name,
                    'profession': profession,
                    'team': team
                })

            logger.info(f"Extracted {len(players)} players (arena: {self._current_arena_type})")
            return players

        except Exception as e:
            logger.error(f"Player extraction failed: {e}")
            # Return 10 unknown players as fallback
            return [
                {'name': f'Unknown_{i}', 'profession': 'Unknown', 'team': 'red' if i < 5 else 'blue'}
                for i in range(10)
            ]

    def _detect_user_character(
        self,
        image: np.ndarray,
        players: List[Dict]
    ) -> Tuple[str, str, float]:
        """
        Use bold detection to identify user's character.

        Args:
            image: Full screenshot image
            players: List of extracted player data

        Returns:
            Tuple of (character_name, team, confidence_score)
        """
        try:
            regions_config = self.config.get('roster_regions')
            name_regions = self.ocr.extract_player_name_regions(
                image, regions_config, arena_type=self._current_arena_type
            )

            # Detect bold text
            bold_index, confidence = self.ocr.detect_bold_text(name_regions)

            # Get the player at that index
            if bold_index < len(players):
                user_char = players[bold_index]['name']
                user_team = players[bold_index]['team']
                logger.info(f"Detected user character: {user_char} ({user_team}) with confidence {confidence:.3f}")
                return user_char, user_team, confidence
            else:
                logger.error(f"Bold index {bold_index} out of range for players list")
                return players[0]['name'], players[0]['team'], 0.0

        except Exception as e:
            logger.error(f"User character detection failed: {e}")
            # Fallback: use first player on winning team
            return players[0]['name'], players[0]['team'], 0.0

    def _validate_extraction(
        self,
        red_score: int,
        blue_score: int,
        players: List[Dict],
        user_char_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate extracted data.

        Args:
            red_score: Red team score
            blue_score: Blue team score
            players: List of player data
            user_char_name: User's character name

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check scores
        if red_score == 0 and blue_score == 0:
            errors.append("Both scores are zero")

        if red_score < 0 or red_score > 9999:
            errors.append(f"Invalid red score: {red_score}")

        if blue_score < 0 or blue_score > 9999:
            errors.append(f"Invalid blue score: {blue_score}")

        # Check players
        if len(players) != 10:
            errors.append(f"Expected 10 players, got {len(players)}")

        unknown_count = sum(1 for p in players if p['name'].startswith('Unknown_'))
        if unknown_count > 0:
            errors.append(f"{unknown_count} players could not be identified")

        # Check user character
        if not user_char_name or user_char_name == 'Unknown':
            errors.append("User character not identified")

        is_valid = len(errors) == 0
        return is_valid, errors

    def log_match(
        self,
        match_data: Dict,
        start_path: str,
        end_path: str,
        map_name: Optional[str] = None
    ) -> int:
        """
        Log processed match to database with best-effort approach.

        Args:
            match_data: Dict with extracted match data
            start_path: Path to start screenshot
            end_path: Path to end screenshot

        Returns:
            Match ID of logged match
        """
        try:
            # Extract data with defaults for missing values
            red_score = match_data.get('red_score', 0)
            blue_score = match_data.get('blue_score', 0)
            user_char = match_data.get('user_character', 'Unknown')
            user_team = match_data.get('user_team', 'red')
            players = match_data.get('players', [])

            # Ensure we have 10 players (fill with unknowns if needed)
            while len(players) < 10:
                team = 'red' if len(players) < 5 else 'blue'
                players.append({
                    'name': f'Unknown_{len(players)}',
                    'profession': 'Unknown',
                    'team': team
                })

            # Log to database
            match_id = self.db.log_match(
                red_score=red_score,
                blue_score=blue_score,
                user_team=user_team,
                user_char_name=user_char,
                players=players,
                screenshot_start_path=start_path,
                screenshot_end_path=end_path,
                arena_type=self._current_arena_type,
                map_name=map_name
            )

            logger.info(f"Match #{match_id} logged successfully")
            return match_id

        except Exception as e:
            logger.error(f"Failed to log match to database: {e}", exc_info=True)
            raise
