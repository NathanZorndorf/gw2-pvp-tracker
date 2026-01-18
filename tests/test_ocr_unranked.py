"""Tests for OCR extraction on unranked gameplay screenshots."""
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

UNRANKED_FOLDERS = sorted([
    d for d in SAMPLES_ROOT.iterdir() 
    if d.is_dir() and d.name.startswith("unranked") and (d / "ground_truth.yaml").exists()
])

@pytest.fixture(scope="session")
def easyocr_method():
    """Share EasyOCR reader across all tests in the session."""
    return EasyOCRMethod(use_gpu=True, resize_factor=2.0, languages=['en', 'es', 'fr', 'pt', 'de'])

@pytest.mark.parametrize("folder", UNRANKED_FOLDERS, ids=lambda d: d.name)
def test_easyocr_recognizes_all_names_unranked(folder, easyocr_method, stats_recorder):
    """Run EasyOCR on the unranked sample folder and assert all names are recognized exactly."""
    if not (folder / "ground_truth.yaml").exists():
        pytest.skip("Sample ground truth missing")
        
    # Use arena_type='unranked' to ensure correct bounding boxes
    bench = OCRBenchmark(str(folder), str(CONFIG_PATH), arena_type='unranked')
    bench.add_method(easyocr_method)

    results = bench.run_all()
    assert results, "No benchmark results produced"

    # Find the EasyOCR result
    easy_results = None
    for r in results:
        if r.method_name.startswith('EasyOCR'):
            easy_results = r
            break

    assert easy_results is not None, "EasyOCR result not found"

    # All names must be correct — print formatted per-name diagnostics
    print(f'\nDetailed name results for {folder.name}:')
    for nr in easy_results.name_results:
        # nr is a dict with keys: file, team, index, expected, extracted, correct, similarity
        expected = nr.get('expected')
        extracted = nr.get('extracted')
        similarity = nr.get('similarity')
        correct = nr.get('correct')
        status = 'OK' if correct else f"FAIL ({similarity*100:.0f}% sim)"
        print(f"  [{status}] {nr.get('file')} {nr.get('team')}[{nr.get('index')}]: expected='{expected}', got='{extracted}'")

    # Record stats
    total = len(easy_results.name_results)
    if total == 0:
        pytest.skip(f"No name ground truth available for {folder.name}")

    correct_count = sum(1 for nr in easy_results.name_results if nr.get('correct'))
    stats_recorder.append({
        'category': 'Name Recognition',
        'correct': correct_count,
        'total': total,
        'test': f'test_easyocr_recognizes_all_names_unranked[{folder.name}]'
    })

    assert easy_results.name_accuracy == 1.0, f"Name accuracy not 100%: {easy_results.name_accuracy}"
