"""Tests for CSV config loading and logical splitting of bounding boxes."""
import pytest
from pathlib import Path
import sys

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

CONFIG_PATH = Path("config.yaml")

def test_config_loads_csv_overrides():
    """Test that Config loads overrides from data/all_bounding_boxes.csv."""
    config = Config(str(CONFIG_PATH))
    
    # Check that roster_regions exists
    roster_regions = config.get('roster_regions')
    assert roster_regions is not None
    
    # Check unranked config
    unranked = roster_regions.get('unranked')
    assert unranked is not None
    
    # Check Red Names
    red_names = unranked.get('red_team_names')
    assert red_names is not None
    assert 'row_height' in red_names
    assert 'num_players' in red_names
    assert red_names['num_players'] == 5
    
    # Check calculation of row_height
    # In CSV: Red Names height is 290. 290 // 5 = 58.
    # Note: If CSV changes, this test might need update, but validating the split logic is key.
    # We expect row_height to be roughly height/5.
    
    # Let's verify against the actual CSV file if it exists
    csv_path = Path("data/all_bounding_boxes.csv")
    if csv_path.exists():
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['label_name'] == 'Red Names':
                    height = int(row['bbox_height'])
                    expected_row_height = height // 5
                    assert red_names['row_height'] == expected_row_height
                    break

def test_config_propagates_to_ranked():
    """Test that CSV overrides are propagated to ranked arena config as well."""
    config = Config(str(CONFIG_PATH))
    ranked = config.get('roster_regions').get('ranked')
    
    # Ranked should have been populated by the CSV loader if it didn't exist or was overridden
    assert ranked is not None
    
    red_names = ranked.get('red_team_names')
    assert red_names is not None
    assert red_names['row_height'] > 0
    assert red_names['num_players'] == 5

if __name__ == "__main__":
    pytest.main([__file__])
