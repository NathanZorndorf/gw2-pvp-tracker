"""Tests for Arena Type (Ranked/Unranked) detection."""
import sys
import pytest
from pathlib import Path
import cv2
import yaml

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from vision.ocr_engine import OCREngine

SAMPLES_ROOT = Path("data/samples")
ALL_FOLDERS = sorted([
    d for d in SAMPLES_ROOT.iterdir() 
    if d.is_dir() and (d.name.startswith("ranked") or d.name.startswith("unranked"))
])

@pytest.fixture
def config():
    return Config("config.yaml")

@pytest.fixture
def ocr_engine(config):
    return OCREngine(
        engine=config.get('ocr.engine', 'easyocr'),
        tesseract_path=config.get('ocr.tesseract_path')
    )

@pytest.mark.parametrize("folder", ALL_FOLDERS, ids=lambda d: d.name)
def test_arena_type_detection(folder, config, ocr_engine, stats_recorder):
    """Detect arena type for each valid screenshot in the folder."""
    
    expected_type = 'ranked' if folder.name.startswith('ranked') else 'unranked'
    
    # Find all PNGs
    images = list(folder.glob("*.png"))
    if not images:
        pytest.skip(f"No images in {folder}")

    roster_regions = config.get('roster_regions')
    
    total_checked = 0
    correct_checked = 0
    
    print(f"\nChecking arena type for {folder.name} (Expect: {expected_type})")
    
    for img_path in images:
        # Only check full screenshots
        if "full" not in img_path.name.lower():
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        detected_type, confidence = ocr_engine.detect_arena_type(image, roster_regions)
        
        is_correct = (detected_type == expected_type)
        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] {img_path.name}: detected='{detected_type}' (conf={confidence:.2f})")
        
        total_checked += 1
        if is_correct:
            correct_checked += 1
            
    stats_recorder.append({
        'category': 'Arena Type Recognition',
        'correct': correct_checked,
        'total': total_checked,
        'test': f'test_arena_type_detection[{folder.name}]'
    })
    
    if total_checked > 0:
        accuracy = correct_checked / total_checked
        # We expect high accuracy, but maybe not 100% on obscure cases?
        # Let's assert 100% for now as these are ground truth samples.
        assert accuracy == 1.0, f"Arena type detection failed for {folder.name}"
    else:
        pytest.skip("No valid full screenshots found")
