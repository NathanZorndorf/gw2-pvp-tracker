
import time
import cv2
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import Config
from database.models import Database
from automation.match_processor import MatchProcessor

def main():
    config = Config()
    db = Database("data/pvp_tracker.db")
    processor = MatchProcessor(config, db)

    image_path = project_root / "data" / "samples" / "ranked-1" / "match_start_20260110_102550_full.png"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    print("Starting fine-grained profiling...")
    
    # 1. Arena Type Detection
    start = time.time()
    arena_type = processor.detect_arena_type(image)
    arena_time = time.time() - start
    print(f"Arena Type Detection: {arena_time:.3f}s (Result: {arena_type})")

    # 2. Score Extraction
    start = time.time()
    red, blue = processor._extract_scores(image, "profile")
    score_time = time.time() - start
    print(f"Score Extraction: {score_time:.3f}s (Red: {red}, Blue: {blue})")

    # 3. Profession Detection
    start = time.time()
    profs = processor._detect_professions_from_image(image)
    prof_time = time.time() - start
    print(f"Profession Detection: {prof_time:.3f}s")

    # 4. Name Extraction (Total for 10)
    # Get regions first
    regions_config = processor.config.get('roster_regions')
    name_regions = processor.ocr.extract_player_name_regions(image, regions_config, arena_type=processor._current_arena_type)
    
    start = time.time()
    names = []
    for region, team in name_regions:
        name = processor.ocr.extract_player_name(region)
        names.append(name)
    name_time = time.time() - start
    print(f"Name Extraction (10 players): {name_time:.3f}s (Avg: {name_time/10:.3f}s per player)")

    total_time = arena_time + score_time + prof_time + name_time
    print("-" * 30)
    print(f"Total processing time: {total_time:.3f}s")
    
    # Check if EasyOCR is the bottleneck
    if processor.ocr.easyocr_reader:
        print(f"\nEasyOCR is using GPU: {processor.ocr.easyocr_reader.gpu}")

if __name__ == "__main__":
    main()
