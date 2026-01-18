"""
Update ground_truth.yaml files with profession data from mappings.csv.

This script:
1. Parses data/target-icons/mappings.csv for profession mappings
2. Updates existing ground_truth.yaml files with profession data
3. Creates new ground_truth.yaml files for folders missing them (using OCR for names)
"""

import sys
import csv
import yaml
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluation.ocr_benchmark import EasyOCRMethod


def parse_mappings_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse mappings.csv into a structured dict.

    Returns:
        {
            'ranked-1': {
                'match_start_20260110_102550_full.png': {
                    'Red_Player_1': 'Elementalist',
                    'Blue_Player_1': 'Guardian',
                    ...
                }
            },
            ...
        }
    """
    mappings = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['file_name']
            profession = row['mapping_icon_name']

            # Parse filename: ranked-1_Red_Player_1___Class_match_start_20260110_102550_full.png
            parts = filename.split('_')
            folder = parts[0]  # e.g., 'ranked-1'
            team = parts[1]  # 'Red' or 'Blue'
            player_num = parts[3]  # '1', '2', etc.

            # Find the screenshot name (starts with 'match')
            screenshot = None
            for i, part in enumerate(parts):
                if part == 'match':
                    screenshot = '_'.join(parts[i:])
                    break

            if not screenshot:
                print(f"Warning: Could not parse screenshot name from {filename}")
                continue

            # Build the key
            player_key = f"{team}_Player_{player_num}"

            # Store mapping
            if folder not in mappings:
                mappings[folder] = {}
            if screenshot not in mappings[folder]:
                mappings[folder][screenshot] = {}

            mappings[folder][screenshot][player_key] = profession

    return mappings


def extract_names_via_ocr(samples_dir: Path, config_path: Path) -> Tuple[List[str], List[str]]:
    """
    Extract player names from match_start screenshot using OCR.

    Returns:
        (red_team_names, blue_team_names) - lists of 5 names each
    """
    try:
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Use EasyOCR
        method = EasyOCRMethod(use_gpu=True, resize_factor=2.0)

        # Find match_start screenshot
        start_images = list(samples_dir.glob('match_start_*.png'))
        if not start_images:
            print(f"  No match_start screenshot found in {samples_dir}")
            return [], []

        img_path = start_images[0]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Failed to load {img_path}")
            return [], []

        # Get region config for ranked arena
        regions = config['roster_regions']['ranked']

        # Extract name regions manually
        red_names = []
        blue_names = []

        # Red team
        red_cfg = regions['red_team_names']
        for i in range(red_cfg['num_players']):
            y_start = red_cfg['y_start'] + (i * red_cfg['row_height'])
            region = image[
                y_start:y_start + red_cfg['row_height'],
                red_cfg['x_start']:red_cfg['x_end']
            ]

            result = method.extract_text(region)
            name = result.value if result and result.value else f"RedPlayer{i+1}"
            name = name.strip()
            if not name:
                name = f"RedPlayer{i+1}"
            red_names.append(name)

        # Blue team
        blue_cfg = regions['blue_team_names']
        for i in range(blue_cfg['num_players']):
            y_start = blue_cfg['y_start'] + (i * blue_cfg['row_height'])
            region = image[
                y_start:y_start + blue_cfg['row_height'],
                blue_cfg['x_start']:blue_cfg['x_end']
            ]

            result = method.extract_text(region)
            name = result.value if result and result.value else f"BluePlayer{i+1}"
            name = name.strip()
            if not name:
                name = f"BluePlayer{i+1}"
            blue_names.append(name)

        print(f"  Extracted via OCR: {len(red_names)} red, {len(blue_names)} blue")
        return red_names, blue_names

    except Exception as e:
        print(f"  OCR extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def update_ground_truth_file(
    folder_path: Path,
    folder_name: str,
    profession_mappings: Dict[str, Dict[str, str]],
    config_path: Path
) -> None:
    """Update or create ground_truth.yaml for a folder."""

    gt_path = folder_path / 'ground_truth.yaml'
    screenshot_name = None

    # Find the screenshot name from profession mappings
    if folder_name in profession_mappings:
        screenshot_name = list(profession_mappings[folder_name].keys())[0]
    else:
        print(f"  No profession mappings found for {folder_name}")
        return

    # Check if ground_truth.yaml exists
    if gt_path.exists():
        print(f"  Updating existing ground_truth.yaml")

        # Load existing data
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Find the sample entry for match_start
        for sample in data.get('samples', []):
            if 'match_start' in sample.get('filename', ''):
                # Get professions for this screenshot
                prof_data = profession_mappings[folder_name][screenshot_name]

                # Update red team
                red_team = sample.get('red_team', [])
                if isinstance(red_team, list) and red_team and isinstance(red_team[0], str):
                    # Old format: list of strings
                    new_red_team = []
                    for i, name in enumerate(red_team, start=1):
                        player_key = f"Red_Player_{i}"
                        profession = prof_data.get(player_key, 'NEEDS_MANUAL')
                        new_red_team.append({'name': name, 'profession': profession})
                    sample['red_team'] = new_red_team
                elif isinstance(red_team, list) and red_team and isinstance(red_team[0], dict):
                    # Already dict format, just add professions
                    for i, player in enumerate(red_team, start=1):
                        player_key = f"Red_Player_{i}"
                        player['profession'] = prof_data.get(player_key, 'NEEDS_MANUAL')
                else:
                    # Create from scratch with OCR
                    red_names, _ = extract_names_via_ocr(folder_path, config_path)
                    new_red_team = []
                    for i, name in enumerate(red_names, start=1):
                        player_key = f"Red_Player_{i}"
                        profession = prof_data.get(player_key, 'NEEDS_MANUAL')
                        new_red_team.append({'name': name, 'profession': profession})
                    sample['red_team'] = new_red_team

                # Update blue team
                blue_team = sample.get('blue_team', [])
                if isinstance(blue_team, list) and blue_team and isinstance(blue_team[0], str):
                    # Old format: list of strings
                    new_blue_team = []
                    for i, name in enumerate(blue_team, start=1):
                        player_key = f"Blue_Player_{i}"
                        profession = prof_data.get(player_key, 'NEEDS_MANUAL')
                        new_blue_team.append({'name': name, 'profession': profession})
                    sample['blue_team'] = new_blue_team
                elif isinstance(blue_team, list) and blue_team and isinstance(blue_team[0], dict):
                    # Already dict format, just add professions
                    for i, player in enumerate(blue_team, start=1):
                        player_key = f"Blue_Player_{i}"
                        player['profession'] = prof_data.get(player_key, 'NEEDS_MANUAL')
                else:
                    # Create from scratch with OCR
                    _, blue_names = extract_names_via_ocr(folder_path, config_path)
                    new_blue_team = []
                    for i, name in enumerate(blue_names, start=1):
                        player_key = f"Blue_Player_{i}"
                        profession = prof_data.get(player_key, 'NEEDS_MANUAL')
                        new_blue_team.append({'name': name, 'profession': profession})
                    sample['blue_team'] = new_blue_team

                break

        # Save updated data
        with open(gt_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"  OK Updated {gt_path}")

    else:
        print(f"  Creating new ground_truth.yaml (OCR for names)")

        # Extract names via OCR
        red_names, blue_names = extract_names_via_ocr(folder_path, config_path)

        if not red_names or not blue_names:
            print(f"  FAILED Failed to extract names for {folder_name}")
            return

        # Get professions
        prof_data = profession_mappings[folder_name][screenshot_name]

        # Build team data
        red_team = []
        for i, name in enumerate(red_names, start=1):
            player_key = f"Red_Player_{i}"
            profession = prof_data.get(player_key, 'NEEDS_MANUAL')
            red_team.append({'name': name, 'profession': profession})

        blue_team = []
        for i, name in enumerate(blue_names, start=1):
            player_key = f"Blue_Player_{i}"
            profession = prof_data.get(player_key, 'NEEDS_MANUAL')
            blue_team.append({'name': name, 'profession': profession})

        # Create ground truth data
        data = {
            'samples': [
                {
                    'filename': screenshot_name,
                    'description': 'Start of match scoreboard',
                    'scores': {'red': 0, 'blue': 0},
                    'red_team': red_team,
                    'blue_team': blue_team
                }
            ]
        }

        # Save
        with open(gt_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"  OK Created {gt_path}")


def main():
    """Main execution."""
    # Paths
    mappings_path = project_root / 'data' / 'target-icons' / 'mappings.csv'
    samples_dir = project_root / 'data' / 'samples'
    config_path = project_root / 'config.yaml'

    print("=" * 60)
    print("Updating Ground Truth Files with Profession Data")
    print("=" * 60)

    # Parse mappings
    print(f"\nParsing {mappings_path}...")
    profession_mappings = parse_mappings_csv(mappings_path)
    print(f"Found profession data for {len(profession_mappings)} folders")

    # Process each ranked folder
    for folder_name in sorted(profession_mappings.keys()):
        folder_path = samples_dir / folder_name

        if not folder_path.exists():
            print(f"\n{folder_name}: Folder not found, skipping")
            continue

        print(f"\n{folder_name}:")
        update_ground_truth_file(folder_path, folder_name, profession_mappings, config_path)

    print("\n" + "=" * 60)
    print("Done! Please verify the updated ground_truth.yaml files.")
    print("=" * 60)


if __name__ == '__main__':
    main()
