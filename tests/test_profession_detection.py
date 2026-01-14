"""Tests for profession icon detection on ranked gameplay screenshots."""
import sys
import pytest
import cv2
import yaml
from pathlib import Path

# Ensure project root and src are on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import after sys.path is set
from vision.profession_detector import ProfessionDetector
from config import Config


# Test against all sample folders with ground truth
SAMPLES_ROOT = Path("data/samples")
SAMPLE_FOLDERS = sorted([
    d for d in SAMPLES_ROOT.iterdir() 
    if d.is_dir() and (d / "ground_truth.yaml").exists()
])


@pytest.fixture
def config():
    """Load configuration."""
    config_path = project_root / "config.yaml"
    return Config(str(config_path))


@pytest.fixture
def profession_detector(config):
    """Create a profession detector instance."""
    return ProfessionDetector(config)


def get_ground_truth_professions(folder: Path) -> dict:
    """
    Load ground truth professions from ground_truth.yaml.

    Returns:
        {
            'red_team': ['Elementalist', 'Luminary', ...],
            'blue_team': ['Guardian', 'Amalgam', ...]
        }
    """
    gt_path = folder / "ground_truth.yaml"
    if not gt_path.exists():
        return {}

    with open(gt_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Find the match_start sample
    for sample in data.get('samples', []):
        if 'match_start' in sample.get('filename', ''):
            red_team = []
            blue_team = []

            # Extract professions from red team
            for player in sample.get('red_team', []):
                if isinstance(player, dict):
                    prof = player.get('profession', 'Unknown')
                    red_team.append(prof)
                else:
                    red_team.append('Unknown')

            # Extract professions from blue team
            for player in sample.get('blue_team', []):
                if isinstance(player, dict):
                    prof = player.get('profession', 'Unknown')
                    blue_team.append(prof)
                else:
                    blue_team.append('Unknown')

            return {
                'red_team': red_team,
                'blue_team': blue_team
            }

    return {}


@pytest.mark.parametrize("folder", SAMPLE_FOLDERS)
def test_profession_detection_per_folder(profession_detector, folder):
    """Test profession detection accuracy for each folder."""
    if not folder.exists():
        pytest.skip(f"Folder not found: {folder}")

    # Check if ground truth exists
    gt_path = folder / "ground_truth.yaml"
    if not gt_path.exists():
        pytest.skip(f"Ground truth not found: {gt_path}")

    # Find match_start screenshot
    start_images = list(folder.glob('match_start_*.png'))
    if not start_images:
        pytest.skip(f"No match_start screenshot in {folder}")

    image_path = start_images[0]
    image = cv2.imread(str(image_path))
    assert image is not None, f"Failed to load {image_path}"

    # Get ground truth
    gt_professions = get_ground_truth_professions(folder)
    if not gt_professions:
        pytest.skip(f"No profession ground truth in {folder}")

    # Skip if any profession is marked as NEEDS_MANUAL
    all_profs = gt_professions['red_team'] + gt_professions['blue_team']
    if 'NEEDS_MANUAL' in all_profs:
        pytest.skip(f"Professions need manual entry in {folder}")

    # Detect professions
    results = profession_detector.detect_professions(image, arena_type='ranked')

    assert len(results) == 10, f"Expected 10 results, got {len(results)}"

    # Build expected and detected lists
    expected_professions = gt_professions['red_team'] + gt_professions['blue_team']
    detected_professions = []

    # Sort results by team and index
    results_sorted = sorted(results, key=lambda x: (0 if x['team'] == 'red' else 1, x['player_index']))
    detected_professions = [r['profession'] for r in results_sorted]

    # Calculate accuracy
    if not expected_professions:
        pytest.skip(f"No ground truth professions found in {folder.name}")

    correct = sum(1 for exp, det in zip(expected_professions, detected_professions) if exp == det)
    accuracy = correct / len(expected_professions)

    # Print detailed results
    print(f"\n{folder.name}:")
    print(f"  Accuracy: {accuracy * 100:.1f}% ({correct}/{len(expected_professions)})")

    for i, (exp, det, result) in enumerate(zip(expected_professions, detected_professions, results_sorted)):
        team = result['team']
        idx = result['player_index']
        conf = result['confidence']
        status = 'OK' if exp == det else 'FAIL'
        print(f"  [{status}] {team} player {idx}: expected={exp:15s} got={det:15s} (conf={conf:.3f})")

    # Assert minimum accuracy threshold per folder (70%, overall target is 88% baseline)
    assert accuracy >= 0.70, f"Accuracy {accuracy*100:.1f}% below 70% threshold in {folder.name}"


def test_profession_detection_overall_accuracy(profession_detector, stats_recorder):
    """Test overall profession detection accuracy across all folders."""
    total_correct = 0
    total_tested = 0

    for folder in SAMPLE_FOLDERS:
        if not folder.exists():
            continue

        gt_path = folder / "ground_truth.yaml"
        if not gt_path.exists():
            continue

        start_images = list(folder.glob('match_start_*.png'))
        if not start_images:
            continue

        image_path = start_images[0]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        # Get ground truth
        gt_professions = get_ground_truth_professions(folder)
        if not gt_professions:
            continue

        # Skip if any profession is marked as NEEDS_MANUAL
        all_profs = gt_professions['red_team'] + gt_professions['blue_team']
        if 'NEEDS_MANUAL' in all_profs:
            continue

        # Determine arena type
        arena_type = 'ranked' if folder.name.startswith('ranked') else 'unranked'

        # Detect professions
        results = profession_detector.detect_professions(image, arena_type=arena_type)

        if len(results) != 10:
            continue

        # Build expected list
        expected_professions = gt_professions['red_team'] + gt_professions['blue_team']

        # Sort results by team and index
        results_sorted = sorted(results, key=lambda x: (0 if x['team'] == 'red' else 1, x['player_index']))
        detected_professions = [r['profession'] for r in results_sorted]

        # Count correct
        for exp, det in zip(expected_professions, detected_professions):
            if exp == det:
                total_correct += 1
            total_tested += 1

    # Calculate overall accuracy
    if total_tested > 0:
        overall_accuracy = total_correct / total_tested
        
        # Record stats
        stats_recorder.append({
            'category': 'Icon Matching',
            'correct': total_correct,
            'total': total_tested,
            'test': 'test_profession_detection_overall_accuracy'
        })
        
        print(f"\nOverall Profession Detection Accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_tested})")

        # Assert minimum 85% overall accuracy (92% achievable with grid-search tuning)
        assert overall_accuracy >= 0.85, (
            f"Overall accuracy {overall_accuracy*100:.1f}% below 85% threshold "
            f"(target: 88% baseline, 92% with optimization)"
        )
    else:
        pytest.skip("No valid test data found")


@pytest.mark.skipif(not Path("data/samples/ranked-1").exists(), reason="ranked-1 not found")
def test_profession_detection_confidence_scores(profession_detector):
    """Test that confidence scores are reasonable."""
    folder = Path("data/samples/ranked-1")
    start_images = list(folder.glob('match_start_*.png'))
    if not start_images:
        pytest.skip("No match_start screenshot")

    image = cv2.imread(str(start_images[0]))
    
    # Determine arena type
    arena_type = 'ranked' if folder.name.startswith('ranked') else 'unranked'
    
    results = profession_detector.detect_professions(image, arena_type=arena_type)

    assert len(results) > 0, "No professions detected"

    # Check confidence scores are in valid range
    for result in results:
        assert 0.0 <= result['confidence'] <= 1.0, (
            f"Invalid confidence {result['confidence']} for {result['team']} player {result['player_index']}"
        )

    # Check that at least some detections have reasonable confidence (> 0.3)
    # Template matching typically produces scores in 0.2-0.8 range
    reasonable_confidence_count = sum(1 for r in results if r['confidence'] > 0.3)
    assert reasonable_confidence_count >= len(results) // 2, (
        f"Only {reasonable_confidence_count}/{len(results)} detections have confidence > 0.3"
    )
