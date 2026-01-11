"""Tests for OCR extraction on unranked gameplay screenshots."""
import sys
import pytest
from pathlib import Path
import yaml

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluation.ocr_benchmark import OCRBenchmark, EasyOCRMethod


SAMPLES_DIR = Path("data/samples/unranked-1")
CONFIG_PATH = Path("config.yaml")


@pytest.mark.skipif(not SAMPLES_DIR.exists() or not (SAMPLES_DIR / "ground_truth.yaml").exists(),
                    reason="Unranked sample ground truth missing")
def test_easyocr_recognizes_all_names(tmp_path):
    """Run EasyOCR on the unranked sample folder and assert all names are recognized exactly."""
    # Load config to get fuzzy threshold (but we enforce exact matching here)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Skip test if EasyOCR not installed in the environment
    try:
        import easyocr  # noqa: F401
    except Exception:
        pytest.skip("EasyOCR not installed in test environment")

    # Use arena_type='unranked' to ensure correct bounding boxes
    bench = OCRBenchmark(str(SAMPLES_DIR), str(CONFIG_PATH), arena_type='unranked')

    # Use EasyOCR with additional languages to improve accented-character recognition
    method = EasyOCRMethod(use_gpu=False, resize_factor=2.0, languages=['en', 'es', 'fr', 'pt'])
    bench.add_method(method)

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
    print('\nDetailed name results:')
    for nr in easy_results.name_results:
        # nr is a dict with keys: file, team, index, expected, extracted, correct, similarity
        expected = nr.get('expected')
        extracted = nr.get('extracted')
        similarity = nr.get('similarity')
        correct = nr.get('correct')
        status = 'OK' if correct else f"FAIL ({similarity*100:.0f}% sim)"
        print(f"  [{status}] {nr.get('file')} {nr.get('team')}[{nr.get('index')}]: expected='{expected}', got='{extracted}'")

    assert easy_results.name_accuracy == 1.0, f"Name accuracy not 100%: {easy_results.name_accuracy}"
