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

    def __init__(self, use_gpu: bool = False, resize_factor: float = 2.0):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
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

    def __init__(self, samples_dir: str, config_path: str):
        self.samples_dir = Path(samples_dir)
        self.ground_truth = self._load_ground_truth()
        self.config = self._load_config(config_path)
        self.methods: List[OCRMethod] = []

    def _load_ground_truth(self) -> Dict:
        """Load ground truth from YAML."""
        gt_path = self.samples_dir / "ground_truth.yaml"
        with open(gt_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_config(self, config_path: str) -> Dict:
        """Load application config for region coordinates."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

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
        regions = self.config['roster_regions']

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
        regions = self.config['roster_regions']
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

            # Test name extraction
            name_regions = self._get_name_regions(image)
            expected_names = sample['red_team'] + sample['blue_team']

            for (region, team, idx), expected_name in zip(name_regions, expected_names):
                name_result = method.extract_text(region)
                extracted = name_result.value

                is_correct = extracted.lower() == expected_name.lower()
                similarity = self._calculate_similarity(extracted, expected_name)

                result.name_results.append({
                    'file': sample['filename'],
                    'team': team,
                    'index': idx,
                    'expected': expected_name,
                    'extracted': extracted,
                    'correct': is_correct,
                    'similarity': similarity
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
            except Exception as e:
                logger.error(f"Benchmark failed for {method.name}: {e}")
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
                print(f"  [{status}] {nr['team']}[{nr['index']}]: expected='{nr['expected']}', got='{nr['extracted']}'")


def main():
    """Run OCR benchmark."""
    project_root = Path(__file__).parent.parent.parent
    samples_dir = project_root / "data" / "samples"
    config_path = project_root / "config.yaml"

    # Load tesseract path from config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tesseract_path = config.get('ocr', {}).get('tesseract_path')

    benchmark = OCRBenchmark(str(samples_dir), str(config_path))

    # Add Tesseract variants
    # Default configuration
    benchmark.add_method(TesseractMethod(
        tesseract_path=tesseract_path,
        psm=7, oem=3, resize_factor=2.0,
        use_adaptive_threshold=True, use_denoise=True,
        method_suffix="[default]"
    ))

    # LSTM-only engine
    benchmark.add_method(TesseractMethod(
        tesseract_path=tesseract_path,
        psm=7, oem=1, resize_factor=2.0,
        use_adaptive_threshold=True, use_denoise=True,
        method_suffix="[LSTM]"
    ))

    # Higher resize factor
    benchmark.add_method(TesseractMethod(
        tesseract_path=tesseract_path,
        psm=7, oem=3, resize_factor=3.0,
        use_adaptive_threshold=True, use_denoise=True,
        method_suffix="[3x]"
    ))

    # Otsu threshold instead of adaptive
    benchmark.add_method(TesseractMethod(
        tesseract_path=tesseract_path,
        psm=7, oem=3, resize_factor=2.0,
        use_adaptive_threshold=False, use_denoise=True,
        method_suffix="[Otsu]"
    ))

    # No denoising
    benchmark.add_method(TesseractMethod(
        tesseract_path=tesseract_path,
        psm=7, oem=3, resize_factor=2.0,
        use_adaptive_threshold=True, use_denoise=False,
        method_suffix="[no denoise]"
    ))

    # Single word mode (PSM 8)
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
    print("Starting OCR benchmark...")
    print(f"Samples directory: {samples_dir}")
    print(f"Number of methods: {len(benchmark.methods)}")

    results = benchmark.run_all()
    benchmark.print_results(results)


if __name__ == "__main__":
    main()
