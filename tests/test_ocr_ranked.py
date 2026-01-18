"""Tests for OCR extraction on ranked gameplay screenshots."""
import sys
import pytest
from pathlib import Path
import yaml

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluation.ocr_benchmark import OCRBenchmark, EasyOCRMethod

SAMPLES_ROOT = Path("data/samples")
CONFIG_PATH = Path("config.yaml")

RANKED_FOLDERS = sorted([
    d for d in SAMPLES_ROOT.iterdir() 
    if d.is_dir() and d.name.startswith("ranked") and (d / "ground_truth.yaml").exists()
])

@pytest.fixture(scope="session")
def easyocr_method():
    """Share EasyOCR reader across all tests in the session."""
    return EasyOCRMethod(use_gpu=True, resize_factor=2.0, languages=['en', 'es', 'fr', 'pt', 'de'])

def create_benchmark(folder, method):
    if not (folder / "ground_truth.yaml").exists():
        pytest.skip("Ranked gameplay ground truth missing")
    
    # Use arena_type='ranked' to ensure correct bounding boxes
    bench = OCRBenchmark(str(folder), str(CONFIG_PATH), arena_type='ranked')
    bench.add_method(method)
    return bench

@pytest.mark.parametrize("folder", RANKED_FOLDERS, ids=lambda d: d.name)
def test_easyocr_recognizes_ranked_start_names(folder, easyocr_method, stats_recorder):
    """Test that EasyOCR correctly extracts all player names from start frame."""
    benchmark = create_benchmark(folder, easyocr_method)
    results = benchmark.run_all()
    assert results, "No benchmark results produced"

    # Find EasyOCR result
    easy_results = None
    for r in results:
        if r.method_name.startswith('EasyOCR'):
            easy_results = r
            break

    assert easy_results is not None, "EasyOCR result not found"

    # Filter to only start frame name results
    start_name_results = [
        nr for nr in easy_results.name_results
        if 'start' in nr.get('file', '').lower()
    ]

    if not start_name_results:
        # Some samples might not have a start screen (only end)
        return

    # Print detailed diagnostics
    print(f'\nDetailed name results (start frame) for {folder.name}:')
    for nr in start_name_results:
        expected = nr.get('expected')
        extracted = nr.get('extracted')
        similarity = nr.get('similarity', 0)
        correct = nr.get('correct')
        status = 'OK' if correct else f"FAIL ({similarity*100:.0f}% sim)"
        print(f"  [{status}] {nr.get('team')}[{nr.get('index')}]: expected='{expected}', got='{extracted}'")

    # Stats recording
    total = len(start_name_results)
    correct_count = sum(1 for nr in start_name_results if nr.get('correct'))
    stats_recorder.append({
        'category': 'Name Recognition',
        'correct': correct_count,
        'total': total,
        'test': f'test_easyocr_recognizes_ranked_start_names[{folder.name}]'
    })

    # Calculate accuracy for start frame only
    if start_name_results:
        accuracy = correct_count / total
        assert accuracy == 1.0, f"Start frame name accuracy not 100%: {accuracy*100:.1f}%"

@pytest.mark.parametrize("folder", RANKED_FOLDERS, ids=lambda d: d.name)
def test_easyocr_extracts_ranked_scores(folder, easyocr_method, stats_recorder):
    """Test that EasyOCR correctly extracts scores from end frame."""
    benchmark = create_benchmark(folder, easyocr_method)
    results = benchmark.run_all()
    assert results, "No benchmark results produced"

    # Find EasyOCR result
    easy_results = None
    for r in results:
        if r.method_name.startswith('EasyOCR'):
            easy_results = r
            break

    assert easy_results is not None, "EasyOCR result not found"

    # Filter to end frame score results
    end_score_results = [
        sr for sr in easy_results.score_results
        if 'end' in sr.get('file', '').lower()
    ]
    
    if not end_score_results:
        # Some samples might not have an end screen
        return

    # Print detailed diagnostics
    print(f'\nDetailed score results (end frame) for {folder.name}:')
    for sr in end_score_results:
        expected = sr.get('expected')
        extracted = sr.get('extracted')
        correct = sr.get('correct')
        status = 'OK' if correct else 'FAIL'
        print(f"  [{status}] {sr.get('team')}: expected={expected}, got={extracted}")

    # Stats recording
    total = len(end_score_results)
    correct_count = sum(1 for sr in end_score_results if sr.get('correct'))
    stats_recorder.append({
        'category': 'Score Identification',
        'correct': correct_count,
        'total': total,
        'test': f'test_easyocr_extracts_ranked_scores[{folder.name}]'
    })

    # Verify scores
    for sr in end_score_results:
        assert sr.get('correct'), f"Score extraction failed for {sr.get('team')}: expected {sr.get('expected')}, got {sr.get('extracted')}"
