import pytest
import os
import yaml
import cv2
from pathlib import Path
from src.automation.match_processor import MatchProcessor
from src.config import Config
from src.database.models import Database

# Path to samples
SAMPLES_DIR = Path("data/samples")

def get_valid_samples():
    """Find all samples with a defined user_player in ground_truth.yaml."""
    samples = []
    if not SAMPLES_DIR.exists():
        return samples
        
    for sample_dir in SAMPLES_DIR.iterdir():
        if not sample_dir.is_dir():
            continue
            
        ground_truth_path = sample_dir / "ground_truth.yaml"
        if not ground_truth_path.exists():
            continue
            
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                user_player = data.get('user_player')
                
                # We need at least one start match screenshot
                start_image = None
                if 'samples' in data:
                    for s in data['samples']:
                        if 'match_start' in s['filename']:
                            start_image = s['filename']
                            break
                
                if user_player and start_image:
                    samples.append({
                        'id': sample_dir.name,
                        'dir': sample_dir,
                        'user_player': user_player,
                        'image_file': start_image
                    })
            except yaml.YAMLError:
                continue
                
    return samples

SAMPLES = get_valid_samples()

@pytest.fixture(scope="function")
def processor():
    """Initialize MatchProcessor for each test to avoid state leakage."""
    config = Config()
    # Use real DB for read access (name matching etc)
    db = Database("data/pvp_tracker.db") 
    return MatchProcessor(config, db)

@pytest.mark.parametrize("sample", SAMPLES, ids=[s['id'] for s in SAMPLES])
def test_user_detection(processor, sample):
    """Test that the user is correctly identified in the screenshot."""
    image_path = sample['dir'] / sample['image_file']
    assert image_path.exists(), f"Image file not found: {image_path}"
    
    image = cv2.imread(str(image_path))
    assert image is not None, "Failed to load image"

    # Run detection
    user_name, team, confidence = processor.detect_user_from_image(image=image)
    
    print(f"Sample: {sample['id']}")
    print(f"Expected: {sample['user_player']}")
    print(f"Detected: {user_name} (Team: {team}, Confidence: {confidence:.3f})")

    assert user_name is not None, "Failed to detect any user"
    
    # Check if we detected the correct user (allow for minor OCR adjustments)
    # The primary goal is ensuring we picked the right row (High Confidence)
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, user_name, sample['user_player']).ratio()
    
    print(f"Similarity: {similarity:.3f}")

    if similarity < 0.85:
         # Fallback: Check if words are just reordered (e.g. "Violet X Edge" vs "Violet Edge X")
         det_words = set(user_name.split())
         exp_words = set(sample['user_player'].split())
         if det_words == exp_words:
             print("Word set match (ignoring order)")
         else:
             assert user_name == sample['user_player'], \
                f"Detected '{user_name}' but expected '{sample['user_player']}'. Team: {team}, Conf: {confidence}"
