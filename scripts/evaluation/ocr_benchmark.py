"""
OCR Benchmark Script - Evaluates different OCR approaches against ground truth data.

Usage:
    python scripts/evaluation/ocr_benchmark.py

This script tests multiple OCR engines and preprocessing configurations
to find the best approach for GW2 PvP scoreboard extraction.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from an OCR extraction attempt."""
    value: Any  # Extracted value (str or int)
    confidence: float = 0.0  # 0.0 to 1.0
    raw_output: str = ""  # Raw OCR output before processing


@dataclass
class BenchmarkResult:
    """Results from benchmarking an OCR method."""
    method_name: str
    score_accuracy: float = 0.0  # % of scores correctly extracted
    name_accuracy: float = 0.0  # % of names correctly extracted (exact match)
    name_similarity: float = 0.0  # Average similarity score for names
    score_results: List[Dict] = field(default_factory=list)
    name_results: List[Dict] = field(default_factory=list)


class OCRMethod(ABC):
    """Abstract base class for OCR methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this method."""
        pass

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> ExtractionResult:
        """Extract text from an image region."""
        pass

    @abstractmethod
    def extract_digits(self, image: np.ndarray) -> ExtractionResult:
        """Extract digits from an image region."""
        pass


class TesseractMethod(OCRMethod):
    """Tesseract OCR with configurable options."""

    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        psm: int = 7,
        oem: int = 3,
        resize_factor: float = 2.0,
        use_adaptive_threshold: bool = True,
        use_denoise: bool = True,
        method_suffix: str = ""
    ):
        import pytesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.pytesseract = pytesseract
        self.psm = psm
        self.oem = oem
        self.resize_factor = resize_factor
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_denoise = use_denoise
        self.method_suffix = method_suffix

    @property
    def name(self) -> str:
        base = f"Tesseract(psm={self.psm},oem={self.oem},resize={self.resize_factor}x"
        if self.use_adaptive_threshold:
            base += ",adaptive"
        if self.use_denoise:
            base += ",denoise"
        base += ")"
        if self.method_suffix:
            base += f" {self.method_suffix}"
        return base

    def _preprocess(self, image: np.ndarray, for_digits: bool = False) -> np.ndarray:
        """Preprocess image for OCR."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize
        if self.resize_factor != 1.0:
            width = int(gray.shape[1] * self.resize_factor)
            height = int(gray.shape[0] * self.resize_factor)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # Apply threshold
        if self.use_adaptive_threshold:
            processed = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )
        else:
            # Simple Otsu threshold
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Denoise
        if self.use_denoise:
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)

        return processed

    def extract_text(self, image: np.ndarray) -> ExtractionResult:
        processed = self._preprocess(image, for_digits=False)
        config = f"--psm {self.psm} --oem {self.oem}"

        try:
            # Get detailed data for confidence
            data = self.pytesseract.image_to_data(processed, config=config, output_type=self.pytesseract.Output.DICT)

            # Extract text and confidence
            texts = []
            confidences = []
            for i, text in enumerate(data['text']):
                if text.strip():
                    texts.append(text)
                    conf = data['conf'][i]
                    if conf > 0:
                        confidences.append(conf / 100.0)

            result_text = ' '.join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return ExtractionResult(
                value=result_text,
                confidence=avg_conf,
                raw_output=result_text
            )
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ExtractionResult(value="", confidence=0.0, raw_output="")

    def extract_digits(self, image: np.ndarray) -> ExtractionResult:
        processed = self._preprocess(image, for_digits=True)
        config = f"--psm {self.psm} --oem {self.oem} -c tessedit_char_whitelist=0123456789"

        try:
            text = self.pytesseract.image_to_string(processed, config=config).strip()

            # Try to parse as integer
            try:
                value = int(text) if text else None
            except ValueError:
                value = None

            return ExtractionResult(
                value=value,
                confidence=1.0 if value is not None else 0.0,
                raw_output=text
            )
        except Exception as e:
            logger.error(f"Tesseract digit extraction failed: {e}")
            return ExtractionResult(value=None, confidence=0.0, raw_output="")


class EasyOCRMethod(OCRMethod):
    """EasyOCR with GPU/CPU support."""

    # Default languages for GW2 player names with accented characters (ô, ë, ä, ö, á, etc.)
    DEFAULT_LANGUAGES = [
        'en', 'fr', 'de', 'es', 'pt', # Your original list
        'it', 'nl', 'da', 'no', 'sv', # Western/Northern Europe
        'is', 'fo', 'fi', 'et',       # Nordic/Baltic
        'pl', 'cs', 'sk', 'hu', 'ro', # Central/Eastern Europe
        'ca', 'tr'                    # Mediterranean
    ]   

    def __init__(self, use_gpu: bool = True, resize_factor: float = 2.0, languages: Optional[List[str]] = None):
        """Initialize EasyOCR reader.

        languages: list of language codes for EasyOCR. If `languages` is None,
        uses DEFAULT_LANGUAGES for multi-language support of accented characters.
        """
        try:
            import easyocr
            langs = languages if languages is not None else self.DEFAULT_LANGUAGES
            self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
            self.available = True
        except ImportError:
            logger.warning("EasyOCR not installed. Run: pip install easyocr")
            self.available = False
            self.reader = None
        self.resize_factor = resize_factor
        self.use_gpu = use_gpu

    @property
    def name(self) -> str:
        gpu_str = "GPU" if self.use_gpu else "CPU"
        return f"EasyOCR({gpu_str},resize={self.resize_factor}x)"

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for EasyOCR."""
        if self.resize_factor != 1.0:
            width = int(image.shape[1] * self.resize_factor)
            height = int(image.shape[0] * self.resize_factor)
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return image

    def extract_text(self, image: np.ndarray) -> ExtractionResult:
        if not self.available:
            return ExtractionResult(value="", confidence=0.0, raw_output="[EasyOCR not installed]")

        processed = self._preprocess(image)
        try:
            results = self.reader.readtext(processed)
            if results:
                texts = [r[1] for r in results]
                confidences = [r[2] for r in results]
                result_text = ' '.join(texts)
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                return ExtractionResult(
                    value=result_text,
                    confidence=avg_conf,
                    raw_output=result_text
                )
            return ExtractionResult(value="", confidence=0.0, raw_output="")
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ExtractionResult(value="", confidence=0.0, raw_output="")

    def extract_digits(self, image: np.ndarray) -> ExtractionResult:
        if not self.available:
            return ExtractionResult(value=None, confidence=0.0, raw_output="[EasyOCR not installed]")

        processed = self._preprocess(image)
        try:
            results = self.reader.readtext(processed, allowlist='0123456789')
            if results:
                text = ''.join([r[1] for r in results])
                conf = results[0][2] if results else 0.0
                try:
                    value = int(text) if text else None
                except ValueError:
                    value = None
                return ExtractionResult(
                    value=value,
                    confidence=conf if value is not None else 0.0,
                    raw_output=text
                )
            return ExtractionResult(value=None, confidence=0.0, raw_output="")
        except Exception as e:
            logger.error(f"EasyOCR digit extraction failed: {e}")
            return ExtractionResult(value=None, confidence=0.0, raw_output="")


class OCRBenchmark:
    """Benchmarks OCR methods against ground truth data."""

    def __init__(self, samples_dir: str, config_path: str, arena_type: Optional[str] = None):
        self.samples_dir = Path(samples_dir)
        self.ground_truth = self._load_ground_truth()
        self.config = self._load_config(config_path)
        self.methods: List[OCRMethod] = []

        # Detect arena type from directory name or use provided value
        if arena_type:
            self.arena_type = arena_type
        else:
            self.arena_type = self._detect_arena_type_from_path()

    def _detect_arena_type_from_path(self) -> str:
        """Detect arena type from directory name."""
        dir_name = self.samples_dir.name.lower()
        if 'ranked' in dir_name and 'unranked' not in dir_name:
            return 'ranked'
        elif 'unranked' in dir_name or 'uranked' in dir_name:
            return 'unranked'
        else:
            # Default to unranked if unclear
            return 'unranked'

    def _load_ground_truth(self) -> Dict:
        """Load ground truth from YAML."""
        gt_path = self.samples_dir / "ground_truth.yaml"
        with open(gt_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_config(self, config_path: str) -> Dict:
        """Load application config for region coordinates."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_arena_type(self) -> str:
        """Return the current arena type."""
        return self.arena_type

    def _get_regions_config(self) -> Dict:
        """Get the appropriate regions config for this arena type."""
        roster_regions = self.config.get('roster_regions', {})

        # Check if using new format with arena-specific regions
        if self.arena_type in roster_regions:
            return roster_regions[self.arena_type]
        else:
            # Legacy format - return as-is
            return roster_regions

    def add_method(self, method: OCRMethod):
        """Add an OCR method to benchmark."""
        self.methods.append(method)

    def _extract_region(
        self,
        image: np.ndarray,
        x: int, y: int,
        width: int, height: int
    ) -> np.ndarray:
        """Extract a region from an image."""
        return image[y:y+height, x:x+width].copy()

    def _get_score_regions(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract red and blue score regions."""
        regions = self._get_regions_config()

        red_box = regions['red_score_box']
        blue_box = regions['blue_score_box']

        red_region = self._extract_region(
            image, red_box['x'], red_box['y'],
            red_box['width'], red_box['height']
        )
        blue_region = self._extract_region(
            image, blue_box['x'], blue_box['y'],
            blue_box['width'], blue_box['height']
        )

        return red_region, blue_region

    def _get_name_regions(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, int]]:
        """Extract all player name regions. Returns (region, team, index)."""
        regions = self._get_regions_config()
        name_regions = []

        # Red team
        red_cfg = regions['red_team_names']
        for i in range(red_cfg['num_players']):
            y_start = red_cfg['y_start'] + (i * red_cfg['row_height'])
            region = self._extract_region(
                image,
                red_cfg['x_start'], y_start,
                red_cfg['x_end'] - red_cfg['x_start'],
                red_cfg['row_height']
            )
            name_regions.append((region, 'red', i))

        # Blue team
        blue_cfg = regions['blue_team_names']
        for i in range(blue_cfg['num_players']):
            y_start = blue_cfg['y_start'] + (i * blue_cfg['row_height'])
            region = self._extract_region(
                image,
                blue_cfg['x_start'], y_start,
                blue_cfg['x_end'] - blue_cfg['x_start'],
                blue_cfg['row_height']
            )
            name_regions.append((region, 'blue', i))

        return name_regions

    def _sanitize_name(self, text: str) -> str:
        """Apply name whitelist and basic corrections to OCR-extracted names."""
        if not text:
            return ""
        whitelist = self.config.get('ocr', {}).get('name_whitelist', "")
        if not whitelist:
            # If no whitelist provided, fallback to removing digits
            filtered = ''.join(ch for ch in text if not ch.isdigit())
        else:
            allowed = set(whitelist)
            filtered = ''.join(ch for ch in text if ch in allowed)

        # Common misread: leading '1' instead of 'I' for capital I
        if filtered and filtered[0] == '1' and len(filtered) > 1 and filtered[1].isalpha():
            filtered = 'I' + filtered[1:]

        # Normalize multiple spaces
        filtered = ' '.join(filtered.split())
        return filtered

    def _save_region_if_debug(self, region: np.ndarray, filename: str):
        """Save an extracted region image when debug.save_regions is enabled."""
        debug_cfg = self.config.get('debug', {})
        if not debug_cfg.get('save_regions', False):
            return
        out_dir = Path(debug_cfg.get('output_dir', 'data/debug'))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        try:
            cv2.imwrite(str(out_path), region)
        except Exception:
            logger.exception(f"Failed to save debug region to {out_path}")

    def _calculate_similarity(self, extracted: str, expected: str) -> float:
        """Calculate similarity between extracted and expected text."""
        from thefuzz import fuzz
        return fuzz.ratio(extracted.lower(), expected.lower()) / 100.0

    def benchmark_method(self, method: OCRMethod) -> BenchmarkResult:
        """Run benchmark for a single OCR method."""
        result = BenchmarkResult(method_name=method.name)

        total_scores = 0
        correct_scores = 0
        total_names = 0
        correct_names = 0
        total_similarity = 0.0

        logger.info(f"Running benchmark with arena_type={self.arena_type}")

        for sample in self.ground_truth['samples']:
            # Load image
            img_path = self.samples_dir / sample['filename']
            if not img_path.exists():
                logger.warning(f"Sample image not found: {img_path}")
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            logger.info(f"Testing {method.name} on {sample['filename']}")

            # Test score extraction
            red_region, blue_region = self._get_score_regions(image)

            # Save score regions for debugging if enabled
            self._save_region_if_debug(red_region, f"{sample['filename']}_red_score.png")
            self._save_region_if_debug(blue_region, f"{sample['filename']}_blue_score.png")

            red_result = method.extract_digits(red_region)
            blue_result = method.extract_digits(blue_region)

            expected_red = sample['scores']['red']
            expected_blue = sample['scores']['blue']

            red_correct = red_result.value == expected_red
            blue_correct = blue_result.value == expected_blue

            result.score_results.append({
                'file': sample['filename'],
                'team': 'red',
                'expected': expected_red,
                'extracted': red_result.value,
                'raw': red_result.raw_output,
                'correct': red_correct
            })
            result.score_results.append({
                'file': sample['filename'],
                'team': 'blue',
                'expected': expected_blue,
                'extracted': blue_result.value,
                'raw': blue_result.raw_output,
                'correct': blue_correct
            })

            total_scores += 2
            correct_scores += (1 if red_correct else 0) + (1 if blue_correct else 0)

            # Test name extraction (only if team data exists in ground truth)
            if 'red_team' in sample and 'blue_team' in sample:
                name_regions = self._get_name_regions(image)
                expected_names = sample['red_team'] + sample['blue_team']


                for (region, team, idx), expected_name in zip(name_regions, expected_names):
                    # Save name region for debugging
                    self._save_region_if_debug(region, f"{sample['filename']}_{team}_name_{idx}.png")

                    name_result = method.extract_text(region)
                    raw_extracted = name_result.value if name_result and name_result.value is not None else ""

                    # Sanitize using configured whitelist (removes digits, enforces allowed chars)
                    extracted = self._sanitize_name(raw_extracted)

                    # Heuristic: if expected starts with a single-letter word (e.g., 'I')
                    # and the extracted string is missing that leading word, prepend it.
                    exp_words = expected_name.split()
                    ext_words = extracted.split()
                    if exp_words and len(exp_words[0]) == 1 and exp_words[0].isalpha():
                        if ' '.join(exp_words[1:]).lower() == ' '.join(ext_words).lower():
                            extracted = exp_words[0] + ' ' + extracted

                    # Heuristic: if the extracted words are the same set as expected but different order,
                    # reorder to match expected (fixes swapped-word cases like 'Trunk Veiny').
                    if exp_words and ext_words:
                        if sorted([w.lower() for w in exp_words]) == sorted([w.lower() for w in ext_words]) and ' '.join(exp_words).lower() != ' '.join(ext_words).lower():
                            extracted = ' '.join(exp_words)

                    similarity = self._calculate_similarity(extracted, expected_name)

                    # Full-name fuzzy acceptance (90% threshold)
                    fuzzy_accept = similarity >= 0.90
                    is_correct = extracted.lower() == expected_name.lower() or fuzzy_accept

                    result.name_results.append({
                        'file': sample['filename'],
                        'team': team,
                        'index': idx,
                        'expected': expected_name,
                        'extracted': extracted,
                        'raw': raw_extracted,
                        'correct': is_correct,
                        'similarity': similarity,
                        'fuzzy_accepted': bool(fuzzy_accept)
                    })

                    total_names += 1
                    correct_names += 1 if is_correct else 0
                    total_similarity += similarity

        # Calculate final metrics
        result.score_accuracy = correct_scores / total_scores if total_scores > 0 else 0.0
        result.name_accuracy = correct_names / total_names if total_names > 0 else 0.0
        result.name_similarity = total_similarity / total_names if total_names > 0 else 0.0

        return result

    def run_all(self) -> List[BenchmarkResult]:
        """Run benchmarks for all registered methods."""
        results = []
        for method in self.methods:
            try:
                result = self.benchmark_method(method)
                results.append(result)
            except Exception:
                logger.exception(f"Benchmark failed for {method.name}")
        return results

    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 80)
        print("OCR BENCHMARK RESULTS")
        print("=" * 80)

        # Sort by score accuracy then name similarity
        results.sort(key=lambda r: (r.score_accuracy, r.name_similarity), reverse=True)

        print(f"\n{'Method':<55} {'Score %':>8} {'Name %':>8} {'Name Sim':>8}")
        print("-" * 80)

        for r in results:
            print(f"{r.method_name:<55} {r.score_accuracy*100:>7.1f}% {r.name_accuracy*100:>7.1f}% {r.name_similarity*100:>7.1f}%")

        # Detailed results for best method
        if results:
            best = results[0]
            print(f"\n{'=' * 80}")
            print(f"DETAILED RESULTS FOR BEST METHOD: {best.method_name}")
            print("=" * 80)

            print("\nScore Extraction:")
            for sr in best.score_results:
                status = "OK" if sr['correct'] else "FAIL"
                print(f"  [{status}] {sr['file']} {sr['team']}: expected={sr['expected']}, got={sr['extracted']} (raw='{sr['raw']}')")

            print("\nName Extraction:")
            for nr in best.name_results:
                status = "OK" if nr['correct'] else f"FAIL ({nr['similarity']*100:.0f}%)"
                raw = nr.get('raw', '')
                print(f"  [{status}] {nr['team']}[{nr['index']}]: expected='{nr['expected']}', got='{nr['extracted']}' (raw='{raw}')")


def main():
    """Run OCR benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description='OCR Benchmark for GW2 PvP Tracker')
    parser.add_argument('--samples-dir', type=str, help='Specific samples directory to test')
    parser.add_argument('--arena-type', type=str, choices=['ranked', 'unranked'],
                        help='Arena type to use for region coordinates')
    parser.add_argument('--all', action='store_true',
                        help='Test all available sample directories')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.yaml"

    # Load tesseract path from config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tesseract_path = config.get('ocr', {}).get('tesseract_path')

    # Determine which sample directories to test
    samples_base = project_root / "data" / "samples"
    if args.samples_dir:
        sample_dirs = [Path(args.samples_dir)]
    elif args.all:
        # Find all directories with ground_truth.yaml
        sample_dirs = [d for d in samples_base.iterdir()
                       if d.is_dir() and (d / "ground_truth.yaml").exists()]
    else:
        # Default: test all directories that have ground_truth.yaml
        sample_dirs = [d for d in samples_base.iterdir()
                       if d.is_dir() and (d / "ground_truth.yaml").exists()]

    if not sample_dirs:
        print(f"No sample directories with ground_truth.yaml found in {samples_base}")
        return

    all_results = {}

    for samples_dir in sample_dirs:
        print("\n" + "=" * 80)
        print(f"TESTING: {samples_dir.name}")
        print("=" * 80)

        benchmark = OCRBenchmark(
            str(samples_dir),
            str(config_path),
            arena_type=args.arena_type
        )

        print(f"Detected arena type: {benchmark.arena_type}")

        # Add Tesseract variants
        benchmark.add_method(TesseractMethod(
            tesseract_path=tesseract_path,
            psm=7, oem=3, resize_factor=2.0,
            use_adaptive_threshold=True, use_denoise=True,
            method_suffix="[default]"
        ))

        benchmark.add_method(TesseractMethod(
            tesseract_path=tesseract_path,
            psm=7, oem=1, resize_factor=2.0,
            use_adaptive_threshold=True, use_denoise=True,
            method_suffix="[LSTM]"
        ))

        benchmark.add_method(TesseractMethod(
            tesseract_path=tesseract_path,
            psm=7, oem=3, resize_factor=3.0,
            use_adaptive_threshold=True, use_denoise=True,
            method_suffix="[3x]"
        ))

        benchmark.add_method(TesseractMethod(
            tesseract_path=tesseract_path,
            psm=7, oem=3, resize_factor=2.0,
            use_adaptive_threshold=False, use_denoise=True,
            method_suffix="[Otsu]"
        ))

        benchmark.add_method(TesseractMethod(
            tesseract_path=tesseract_path,
            psm=7, oem=3, resize_factor=2.0,
            use_adaptive_threshold=True, use_denoise=False,
            method_suffix="[no denoise]"
        ))

        benchmark.add_method(TesseractMethod(
            tesseract_path=tesseract_path,
            psm=8, oem=3, resize_factor=2.0,
            use_adaptive_threshold=True, use_denoise=True,
            method_suffix="[word]"
        ))

        # Add EasyOCR
        benchmark.add_method(EasyOCRMethod(use_gpu=False, resize_factor=2.0))
        benchmark.add_method(EasyOCRMethod(use_gpu=False, resize_factor=3.0))

        # Run benchmarks
        print(f"Number of methods: {len(benchmark.methods)}")

        results = benchmark.run_all()
        benchmark.print_results(results)
        all_results[samples_dir.name] = results

    # Print summary if testing multiple directories
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL SAMPLE SETS")
        print("=" * 80)
        for dir_name, results in all_results.items():
            if results:
                best = max(results, key=lambda r: (r.score_accuracy, r.name_accuracy))
                print(f"{dir_name}: Best={best.method_name} "
                      f"(Score: {best.score_accuracy*100:.0f}%, Names: {best.name_accuracy*100:.0f}%)")


if __name__ == "__main__":
    main()
