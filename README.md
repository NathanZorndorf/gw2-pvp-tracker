# GW2 PvP Tracker

A local Python application that uses computer vision and OCR to automatically track Guild Wars 2 PvP match performance.

## Project Status

### Completed Components

- **Database Module** ([src/database/models.py](src/database/models.py))
  - SQLite database with 3-table schema (players, matches, match_participants)
  - Full CRUD operations for players and matches
  - Win rate calculations and match history queries
  - Automatic player statistics updates

- **Screenshot Capture** ([src/vision/capture.py](src/vision/capture.py))
  - Full screen and region capture
  - Support for multiple monitors
  - Automatic timestamped file saving
  - Match start/end capture workflows

- **OCR Engine** ([src/vision/ocr_engine.py](src/vision/ocr_engine.py))
  - Image preprocessing for better OCR accuracy
  - Text extraction using Tesseract
  - Fuzzy name matching for error correction
  - Score validation (0-500 range)

- **Configuration System** ([src/config.py](src/config.py))
  - YAML-based configuration
  - Supports OCR settings, paths, and CV parameters

### Next Steps

- Template matching for scoreboard detection
- User highlight row detection (yellow highlight)
- Profession icon recognition
- Hotkey integration (F8/F9)
- Tactical briefing generator

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** (required for text extraction)
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to: `C:\Program Files\Tesseract-OCR\`
   - Or update `config.yaml` with your installation path

### Setup

```bash
# 1. Clone or navigate to the repository
cd gw2-pvp-tracker

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Testing

Run the component test suite:

```bash
python test_components.py
```

This will test:
- Configuration loading
- Database operations (creates test database in `data/test_pvp_tracker.db`)
- Screenshot capture (saves test screenshots to `screenshots/`)
- OCR engine (requires Tesseract)

### Test Output

```
============================================================
  GW2 PvP Tracker - Component Tests
============================================================

OK Config loaded
OK Database initialized
OK Screen capture initialized
OK OCR engine initialized
...
```

## Project Structure

```
gw2-pvp-tracker/
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   └── models.py           # SQLite database schema & operations
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── capture.py          # Screenshot capture
│   │   └── ocr_engine.py       # OCR text extraction
│   ├── config.py               # Configuration loader
│   └── __init__.py
├── screenshots/                # Captured screenshots
├── data/                       # SQLite database files
├── templates/                  # CV templates (scoreboard, icons)
├── logs/                       # Application logs
├── venv/                       # Virtual environment
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── test_components.py          # Component test suite
├── IMPLEMENTATION_PLAN.md      # Full implementation spec
└── README.md                   # This file
```

## Configuration

Edit [config.yaml](config.yaml) to customize:

- **Screen capture**: Monitor index, capture delay
- **OCR settings**: Tesseract path, character whitelists
- **Database**: Database file path
- **Paths**: Screenshots, templates, logs directories

## Database Schema

### Players Table
Tracks global statistics for each character:
- `char_name` (PK): Character name
- `global_wins`, `global_losses`, `total_matches`: Stats
- `most_played_profession`: Most common profession
- `first_seen`, `last_seen`: Timestamps

### Matches Table
Stores match metadata:
- `match_id` (PK): Unique match identifier
- `timestamp`: When match occurred
- `red_score`, `blue_score`: Final scores
- `winning_team`, `user_team`: Team info
- `user_char_name`: User's character
- `screenshot_*_path`: Screenshot paths

### Match Participants Table
Links players to matches:
- `match_id` (FK): Match reference
- `char_name` (FK): Player reference
- `profession`, `team_color`: Match-specific info
- `is_user`: Boolean flag for user

## Usage Examples

### Database Operations

```python
from src.database.models import Database

# Initialize database
db = Database("data/pvp_tracker.db")

# Add player
db.add_player("PlayerName.1234")

# Get player stats
player = db.get_player("PlayerName.1234")
print(f"{player['char_name']}: {player['global_wins']}W - {player['global_losses']}L")

# Log a match
players = [
    {"name": "Player1", "profession": "Guardian", "team": "blue"},
    # ... 9 more players
]
match_id = db.log_match(
    red_score=487,
    blue_score=500,
    user_team="blue",
    user_char_name="Player1",
    players=players
)
```

### Screenshot Capture

```python
from src.vision.capture import ScreenCapture

# Initialize
capture = ScreenCapture()

# Capture full screen
full_path = capture.capture_and_save_full("match_start")
print(f"Saved to: {full_path}")

# Capture specific region
region_path = capture.capture_and_save_region(
    x=500, y=300, width=800, height=600, prefix="roster"
)

# Capture match start (full + crop)
full, crop = capture.capture_match_start(crop_region=(500, 300, 800, 600))
```

### OCR Engine

```python
from src.vision.ocr_engine import OCREngine
import cv2

# Initialize
ocr = OCREngine(tesseract_path="C:/Program Files/Tesseract-OCR/tesseract.exe")

# Extract text from image
image = cv2.imread("screenshot.png")
text = ocr.extract_text(image)

# Extract score
score = ocr.extract_score(image, validate_range=True)

# Extract player name with fuzzy matching
known_names = ["Player1.1234", "Player2.5678"]
name = ocr.extract_player_name(image, known_names=known_names)
```

## Dependencies

Core libraries:
- **opencv-python**: Computer vision and image processing
- **pytesseract**: OCR text extraction
- **numpy**: Array operations
- **Pillow**: Image manipulation
- **mss**: Fast screenshot capture
- **PyYAML**: Configuration file parsing
- **thefuzz**: Fuzzy string matching
- **python-Levenshtein**: String distance calculations

Development:
- **pytest**: Testing framework

## Contributing

This is an early-stage project. Core components are implemented:
1. Database ✓
2. Screenshot capture ✓
3. OCR engine ✓

Remaining work:
- Computer vision (template matching, highlight detection)
- User interface (hotkeys, console output)
- Integration and testing with real GW2 screenshots

## License

MIT License - See [LICENSE](LICENSE) file

## Disclaimer

This is a third-party tool and is not affiliated with or endorsed by ArenaNet or NCSOFT. The tool operates purely through screenshot analysis and does not modify game files or memory.

## Documentation

For the complete implementation specification, see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
