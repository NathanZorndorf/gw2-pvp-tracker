"""Test specifically for arena type detection logic."""
import sys
import pytest
from pathlib import Path
import cv2

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.config import Config
from src.vision.ocr_engine import OCREngine

RANKED_SAMPLE = Path("data/samples/ranked-1/match_start_20260110_102550_full.png")

@pytest.fixture
def config():
    return Config("config.yaml")

@pytest.mark.skipif(not RANKED_SAMPLE.exists(), reason="Ranked sample not found")
def test_arena_type_detection_tiebreaker(config):
    """
    Test how the system determines arena type when bounding boxes are identical.
    With the new unified bounding boxes, both ranked and unranked configs are identical.
    """
    ocr = OCREngine(
        engine=config.get('ocr.engine', 'easyocr'),
        tesseract_path=config.get('ocr.tesseract_path')
    )
    
    image = cv2.imread(str(RANKED_SAMPLE))
    assert image is not None
    
    # Get the raw config that has been overridden
    roster_regions = config.get('roster_regions')
    
    # Verify our hypothesis: are the regions identical?
    ranked_red = roster_regions['ranked']['red_team_names']
    unranked_red = roster_regions['unranked']['red_team_names']
    
    print(f"Ranked Red: {ranked_red}")
    print(f"Unranked Red: {unranked_red}")
    
    regions_match = (ranked_red == unranked_red)
    print(f"Regions match: {regions_match}")
    
    # Run detection
    detection_region = config.get('arena_type_detection')
    arena, confidence = ocr.detect_arena_type(image, roster_regions, detection_region)
    print(f"Detected: {arena} with confidence {confidence}")
    
    # If they are identical, it currently defaults to unranked.
    # The user might want us to use the explicit 'Arena Type' text if available,
    # OR we might accept that if they are geometrically identical, we can't distinguish 
    # strictly by "name region validity".
    
    # Ideally, we should check the "Arena Type" text region.
