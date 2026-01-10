# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GW2 PvP Tracker is a Windows desktop application that automatically tracks Guild Wars 2 PvP match performance using computer vision and OCR. It captures screenshots at match start/end, extracts player names and scores via Tesseract OCR, and maintains a SQLite database for historical analysis.

**Key Design Philosophy:** The app never asks who the user is - it automatically detects the user's character via bold text highlighting on the scoreboard.

## Tech Stack

- Python 3.10+ (Windows)
- OpenCV for computer vision
- Tesseract OCR for text extraction
- SQLite3 for local database
- keyboard library for global hotkeys (requires admin on Windows)
- thefuzz for fuzzy string matching

## Commands

```bash
# Activate venv (MANDATORY before any Python commands)
.\venv\Scripts\activate

# Run main application (requires admin privileges)
python live_capture.py

# Interactive match logging from screenshots
python log_latest_match.py

# Calibrate OCR region coordinates
python calibrate_coordinates.py

# Run tests
pytest

# Lint
pylint src/
flake8 src/
```

## Architecture

```
src/
├── config.py           # YAML config loader (Config class)
├── database/
│   └── models.py       # SQLite operations (Database class)
├── vision/
│   ├── capture.py      # Screenshot capture (ScreenCapture class)
│   └── ocr_engine.py   # OCR + preprocessing (OCREngine class)
└── automation/
    └── match_processor.py  # Match extraction pipeline (MatchProcessor class)
```

### Data Flow

1. **F8 hotkey** → `ScreenCapture` captures match start screenshot → `MatchProcessor.detect_user_from_image()` identifies user via bold text detection
2. **F9 hotkey** → `ScreenCapture` captures match end screenshot → `MatchProcessor.process_match()` extracts scores and all 10 player names → `Database.log_match()` updates player statistics

### Database Schema

Three tables: `players` (character stats), `matches` (match metadata), `match_participants` (junction table with profession/team). The `player_winrates` view aggregates win rate statistics.

### Configuration

`config.yaml` contains all tunable parameters:
- `roster_regions`: OCR extraction coordinates (calibrated for 4K resolution - divide by 2 for 1080p)
- `ocr`: Tesseract path, PSM modes, character whitelists
- `bold_detection`: Composite algorithm weights for user detection
- `hotkeys`: F8 (match start), F9 (match end), ESC (exit)

## Critical Rules

- **MANDATORY VENV USAGE**: Always activate venv before running any Python commands
- **Windows Admin Required**: Global hotkeys require admin privileges
- **Tesseract Required**: Must be installed at path specified in config.yaml
- **Resolution Sensitive**: Roster region coordinates in config.yaml are for 4K. Adjust for other resolutions
- **Keep Root Clean**: Put new files in appropriate subdirectories (`scripts/`, `src/`, `docs/`, etc.)
- **PEP 8 Style**: Use type hints, f-strings, clear docstrings
