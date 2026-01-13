"""Tests for OCR extraction on ranked gameplay screenshots."""
import sys
import pytest
from pathlib import Path
import yaml

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluation.ocr_benchmark import OCRBenchmark, EasyOCRMethod


SAMPLES_DIR = Path("data/samples/ranked-1")
CONFIG_PATH = Path("config.yaml")


@pytest.fixture
def ocr_benchmark():
    """Create an OCR benchmark instance for ranked gameplay samples."""
    if not (SAMPLES_DIR / "ground_truth.yaml").exists():
        pytest.skip("Ranked gameplay ground truth missing")

    try:
        import easyocr  # noqa: F401
    except ImportError:
        pytest.skip("EasyOCR not installed in test environment")

    # Use arena_type='ranked' to ensure correct bounding boxes
    bench = OCRBenchmark(str(SAMPLES_DIR), str(CONFIG_PATH), arena_type='ranked')
    # Use EasyOCR with additional languages for accented characters
    method = EasyOCRMethod(use_gpu=False, resize_factor=2.0, languages=['en', 'es', 'fr', 'pt', 'de'])
    bench.add_method(method)
    return bench


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Ranked samples directory missing")
def test_easyocr_recognizes_ranked_start_names(ocr_benchmark):
    """Test that EasyOCR correctly extracts all player names from start frame."""
    results = ocr_benchmark.run_all()
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

    # Print detailed diagnostics
    print('\nDetailed name results (start frame):')
    for nr in start_name_results:
        expected = nr.get('expected')
        extracted = nr.get('extracted')
        similarity = nr.get('similarity', 0)
        correct = nr.get('correct')
        status = 'OK' if correct else f"FAIL ({similarity*100:.0f}% sim)"
        print(f"  [{status}] {nr.get('team')}[{nr.get('index')}]: expected='{expected}', got='{extracted}'")

    # Calculate accuracy for start frame only
    if start_name_results:
        correct_count = sum(1 for nr in start_name_results if nr.get('correct'))
        accuracy = correct_count / len(start_name_results)
        assert accuracy == 1.0, f"Start frame name accuracy not 100%: {accuracy*100:.1f}%"


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="Ranked samples directory missing")
def test_easyocr_extracts_ranked_scores(ocr_benchmark):
    """Test that EasyOCR correctly extracts scores from end frame."""
    results = ocr_benchmark.run_all()
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

    # Print detailed diagnostics
    print('\nDetailed score results (end frame):')
    for sr in end_score_results:
        expected = sr.get('expected')
        extracted = sr.get('extracted')
        correct = sr.get('correct')
        status = 'OK' if correct else 'FAIL'
        print(f"  [{status}] {sr.get('team')}: expected={expected}, got={extracted}")

    # Verify scores
    for sr in end_score_results:
        assert sr.get('correct'), f"Score extraction failed for {sr.get('team')}: expected {sr.get('expected')}, got {sr.get('extracted')}"
