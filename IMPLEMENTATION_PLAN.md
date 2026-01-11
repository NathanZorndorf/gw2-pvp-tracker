# GW2 PvP Tracker: OCR Testing & GUI Implementation Plan

## Overview

This plan covers four main objectives:
1. Create ground_truth.yaml for ranked gameplay sample images
2. Create pytests to validate OCR extraction for start/end frames
3. Verify database logging functionality
4. Implement a GUI overlay to display player win rates on F8 press

---

## Phase 1: Ground Truth & OCR Testing


**Important fixes needed in existing ground_truth.yaml:**
1. Fix filenames to match actual files (`match_start_...` and `match_end_...`)
2. team analysis not necessary in end frame - only score OCR recognition and score update are needed for the end frame... team player names are hanled in the start frame. End frame = score update, start frame = player names
3. Names contain special characters (ô, ë, ä) that EasyOCR must handle... yes we need to expand the whitelist to incldue all special characters....

### 1.2 Create Pytest for Ranked Gameplay

**File:** `tests/test_ocr_ranked_gameplay.py`

```python
# Test structure:
# 1. test_easyocr_recognizes_ranked_start_names() - Start frame name extraction
# 2. test_easyocr_extracts_ranked_scores() - End frame score extraction
```

Note that there is already one set of tests in tests/ we should expand on that...

**Key assertions:**
- 100% name accuracy (case-insensitive exact match)
- Score extraction: red=347, blue=500 for end frame

### 1.3 Implementation Steps

2. Run existing OCR benchmark to verify extraction works
4. Create/update pytest file with tests for ranked data
5. Run tests and iterate until 100% pass

**Commands:**
```bash
# Test OCR extraction
python scripts/evaluation/ocr_benchmark.py

# Run specific tests
pytest tests/test_ocr_ranked_gameplay.py -v
```

---

## Phase 2: Database Verification

### 2.1 Create Database Tests

**File:** `tests/test_database.py`

Tests to implement:
1. `test_log_match_creates_match_record()` - Verify match inserted
2. `test_log_match_creates_participants()` - Verify 10 participants created
3. `test_log_match_updates_player_stats()` - Verify win/loss counts update
4. `test_get_player_winrate_returns_correct_values()` - Verify calculation

### 2.2 Integration Test

**File:** `tests/test_match_processing_integration.py`

End-to-end test:
1. Load ranked sample images
2. Process with MatchProcessor
3. Log to test database
4. Verify all data persisted correctly

---

## Phase 3: GUI Overlay Implementation

### 3.1 File Structure

```
src/ui/
  __init__.py           # Export WinRateOverlay
  overlay.py            # Main overlay window class
  player_card.py        # Player row widget
  confidence.py         # Confidence calculation (1-5 stars)
  styles.py             # Colors, fonts, dimensions
```

### 3.2 Core Components

#### WinRateOverlay (`src/ui/overlay.py`)

- **Popup window style** (user preference)
- Centered dialog window with title bar
- On top of gameplay, however user will likely close it after a few seconds of observation so its not in the way during gameplay
- Shows 10 player rows (5 red, 5 blue)
- Shows their win rate, a green check mark emoji for > 50% win rate otherwise red X or crossed circle,
- Manual close via X button

#### PlayerStats Data Class

```python
@dataclass
class PlayerStats:
    name: str
    team: str           # "red" or "blue"
    win_rate: float     # 0.0 - 100.0
    total_matches: int
    is_user: bool       # Highlight user's row
```


### 3.3 Display Layout

```
+------------------------------------------+
|  Match Analysis                      [X] |
+------------------------------------------+
| RED TEAM                                 |
| [R] Cameronz           52.3%  ****-      |
| [R] Fahrts             NEW    -          |
| [R] I Am The Weapon    48.1%  ***--      |
| [R] Darklken           61.2%  *****      |
| [R] Mira Phantom       45.0%  **---      |
+------------------------------------------+
| BLUE TEAM                                |
| [B] Katrimus           55.0%  ****-      |
| [B] General Huggles    NEW    -          |
| [B] Vany Trunk         42.8%  ***--      |
| [B] Grizk Rainfall     58.3%  *****      |
| [B] Phantom Tithe      50.0%  **---      |
+------------------------------------------+
```

Percentages = Win Rate (%)
Stars = Win Rate Converted to 0 - 5 stars
ALso include an integer representing how many games they have appeared in.

The order of the columnss should be: stars, win rate, then "confidence" which right now is just an integer represenitng how many games you've seen them in.
### 3.4 Integration with LiveCapture

**Modify:** `live_capture.py`

1. Initialize tkinter root in main thread
2. Create WinRateOverlay instance
3. On F8: Extract players, query win rates, show overlay
4. On F9: Hide overlay, log match
5. Modify main loop to call `root.update()`

**Threading pattern:**
```python
# From keyboard callback (different thread)
self.root.after(0, lambda: self.overlay.show(player_stats))
```

### 3.5 Configuration Updates

**Add to:** `config.yaml`

```yaml
ui:
  overlay:
    enabled: true
    style: "popup"           # "popup" (centered dialog) or "side_panel"
    width: 400
    show_confidence: true
    auto_close_on_f9: true   # Close popup when match ends
```

---

## Phase 4: Implementation Order

### Step 1: Ground Truth & Tests (OCR)
1. Create `data/samples/actual-gameplay-ranked-1/ground_truth.yaml`
2. Run OCR benchmark to verify names
3. Create `tests/test_ocr_ranked_gameplay.py`
4. Iterate until tests pass

### Step 2: Database Tests
1. Create `tests/test_database.py`
2. Create `tests/test_match_processing_integration.py`
3. Run tests and verify database operations

### Step 3: GUI Implementation
1. Create `src/ui/styles.py` - styling constants
2. Create `src/ui/confidence.py` - confidence calculation
3. Create `src/ui/player_card.py` - player row widget
4. Create `src/ui/overlay.py` - main overlay window
5. Update `src/ui/__init__.py` - exports
6. Modify `live_capture.py` - integrate overlay
7. Update `config.yaml` - add overlay settings

### Step 4: End-to-End Testing
1. Run full application manually
2. Press F8, verify overlay appears with correct data
3. Press F9, verify match logged and overlay hides
4. Check database for correct entries

---

## Critical Files to Modify

| File | Action |
|------|--------|
| `data/samples/actual-gameplay-ranked-1/ground_truth.yaml` | CREATE |
| `tests/test_ocr_ranked_gameplay.py` | CREATE |
| `tests/test_database.py` | CREATE |
| `tests/test_match_processing_integration.py` | CREATE |
| `src/ui/overlay.py` | CREATE |
| `src/ui/player_card.py` | CREATE |
| `src/ui/confidence.py` | CREATE |
| `src/ui/styles.py` | CREATE |
| `src/ui/__init__.py` | MODIFY |
| `live_capture.py` | MODIFY |
| `config.yaml` | MODIFY |

---

## Verification Plan

### OCR Tests
```bash
pytest tests/test_ocr_ranked_gameplay.py -v
```
Expected: All tests pass with 100% name accuracy

### Database Tests
```bash
pytest tests/test_database.py -v
pytest tests/test_match_processing_integration.py -v
```
Expected: All database operations verified

### Manual GUI Test
1. Activate venv: `.\venv\Scripts\activate`
2. Run app: `python live_capture.py`
3. Open GW2, start ranked match
4. Press F8 at match start - verify overlay shows player stats
5. Press F9 at match end - verify overlay hides and data saved
6. Check database: `python scripts/utils/view_stats.py`

---

## Risk Mitigation

2. **Threading issues:** Use tkinter's `after()` method for all GUI updates from callbacks
4. **Admin privileges:** GUI must work alongside keyboard hooks (both require admin on Windows)
