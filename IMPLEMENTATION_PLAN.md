# GW2 PvP Tracker - Complete Implementation Plan

## Executive Summary
A local Python desktop application that uses computer vision and OCR to automatically track Guild Wars 2 PvP match performance. The app captures screenshots at match start and end, extracts player names, scores, and team assignments, then maintains a historical database to provide tactical insights.

---

## 1. Project Vision & Core Philosophy

### The Golden Rule
**The app never asks who the user is or what team they are on.**

Instead, it:
1. Detects the golden-yellow "Highlight" row to identify the current user
2. Compares Red vs. Blue scores to determine the winner
3. Automatically logs match results for all 10 players

### Key Benefits
- Zero manual data entry
- Instant tactical briefing at match start
- Historical win/loss tracking per player
- Profession trend analysis
- Threat assessment of opponents

---

## 2. Technology Stack

### Core Technologies
- **Python 3.10+**: Main language
- **OpenCV (cv2)**: Template matching and image processing
- **Tesseract OCR**: Text extraction from screenshots
- **pytesseract**: Python wrapper for Tesseract
- **SQLite3**: Local database (built-in with Python)
- **keyboard**: Global hotkey detection (F8/F9)
- **Pillow (PIL)**: Image manipulation
- **thefuzz**: Fuzzy string matching for OCR error correction
- **numpy**: Array operations for image processing

### Development Tools
- **pytest**: Unit testing
- **black**: Code formatting
- **pylint**: Code quality
- **git**: Version control

---

## 3. Database Schema (SQLite)

### Table 1: `players`
Tracks global statistics for each unique character encountered.

```sql
CREATE TABLE players (
    char_name TEXT PRIMARY KEY,
    global_wins INTEGER DEFAULT 0,
    global_losses INTEGER DEFAULT 0,
    total_matches INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    most_played_profession TEXT,
    CONSTRAINT valid_stats CHECK (total_matches = global_wins + global_losses)
);
```

**Indexes:**
```sql
CREATE INDEX idx_players_winrate ON players(global_wins, total_matches);
CREATE INDEX idx_players_last_seen ON players(last_seen);
```

### Table 2: `matches`
Stores metadata for each match played.

```sql
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    red_score INTEGER NOT NULL,
    blue_score INTEGER NOT NULL,
    winning_team TEXT NOT NULL CHECK(winning_team IN ('red', 'blue')),
    user_team TEXT NOT NULL CHECK(user_team IN ('red', 'blue')),
    user_char_name TEXT NOT NULL,
    match_result TEXT GENERATED ALWAYS AS (
        CASE WHEN winning_team = user_team THEN 'win' ELSE 'loss' END
    ) STORED,
    screenshot_start_path TEXT,
    screenshot_end_path TEXT,
    FOREIGN KEY (user_char_name) REFERENCES players(char_name)
);
```

**Indexes:**
```sql
CREATE INDEX idx_matches_timestamp ON matches(timestamp);
CREATE INDEX idx_matches_user ON matches(user_char_name);
CREATE INDEX idx_matches_result ON matches(match_result);
```

### Table 3: `match_participants`
Junction table linking players to specific matches with their team and profession.

```sql
CREATE TABLE match_participants (
    participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    char_name TEXT NOT NULL,
    profession TEXT NOT NULL,
    team_color TEXT NOT NULL CHECK(team_color IN ('red', 'blue')),
    is_user BOOLEAN DEFAULT 0,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    FOREIGN KEY (char_name) REFERENCES players(char_name),
    UNIQUE(match_id, char_name)
);
```

**Indexes:**
```sql
CREATE INDEX idx_participants_match ON match_participants(match_id);
CREATE INDEX idx_participants_char ON match_participants(char_name);
CREATE INDEX idx_participants_profession ON match_participants(profession);
```

### Database Views for Analytics

```sql
-- Player win rate view
CREATE VIEW player_winrates AS
SELECT
    char_name,
    global_wins,
    global_losses,
    total_matches,
    ROUND(CAST(global_wins AS FLOAT) / NULLIF(total_matches, 0) * 100, 1) as win_rate_pct,
    most_played_profession
FROM players
WHERE total_matches > 0;

-- Recent match history view
CREATE VIEW recent_matches AS
SELECT
    m.match_id,
    m.timestamp,
    m.user_char_name,
    m.match_result,
    m.red_score,
    m.blue_score,
    m.user_team
FROM matches m
ORDER BY m.timestamp DESC
LIMIT 50;
```

---

## 4. Computer Vision Pipeline

### Phase A: Screenshot Capture & Preprocessing

#### Screenshot Capture Strategy
The app captures **4 screenshots per match**:

1. **Match Start - Full Screen**: Captures entire game window for context
2. **Match Start - Cropped Roster**: Zoomed/cropped view of scoreboard area
3. **Match End - Full Screen**: Captures final state
4. **Match End - Cropped Roster**: Final scoreboard state

**File Naming Convention:**
```
screenshots/
├── YYYYMMDD_HHMMSS_start_full.png
├── YYYYMMDD_HHMMSS_start_roster.png
├── YYYYMMDD_HHMMSS_end_full.png
└── YYYYMMDD_HHMMSS_end_roster.png
```

#### Image Preprocessing Steps
1. **Capture**: Use `mss` or `PIL.ImageGrab` to capture screen
2. **Convert**: BGR to RGB (OpenCV compatibility)
3. **Denoise**: Apply Gaussian blur (kernel size 3x3) to reduce noise
4. **Store**: Save both original and preprocessed versions

### Phase B: Scoreboard Anchor Detection

#### Template Matching Algorithm
```python
# Multi-scale template matching to handle UI scaling
scales = [0.8, 0.9, 1.0, 1.1, 1.2]
best_match = None
best_score = 0.0

for scale in scales:
    # Resize template
    resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)

    # Template matching
    result = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > best_score:
        best_score = max_val
        best_match = (max_loc, scale)

# Confidence threshold
if best_score < 0.7:
    raise ScoreboardNotFoundError("Could not locate scoreboard")
```

#### Coordinate System Establishment
Once anchor is found, establish relative coordinates:
- **Score boxes**: (anchor_x + offset_x, anchor_y + offset_y)
- **Player name regions**: 10 rows below anchor
- **Profession icons**: Adjacent to names
- **Team divider**: Between rows 5 and 6

### Phase C: User Detection via Highlight Row

#### Yellow Highlight Detection Algorithm
```python
def find_user_row(roster_region, num_rows=10):
    """
    Scan each row for golden-yellow highlight pixels.
    Returns: row_index (0-9) where 0-4=Blue, 5-9=Red
    """
    # Define HSV range for GW2 highlight color
    lower_yellow = np.array([20, 50, 50])   # H:20, S:50, V:50
    upper_yellow = np.array([40, 255, 255]) # H:40, S:255, V:255

    row_height = roster_region.shape[0] // num_rows
    max_yellow_pixels = 0
    user_row = None

    for row_idx in range(num_rows):
        # Extract row region
        y_start = row_idx * row_height
        y_end = y_start + row_height
        row_img = roster_region[y_start:y_end, :]

        # Convert to HSV and count yellow pixels
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_count = cv2.countNonZero(mask)

        if yellow_count > max_yellow_pixels:
            max_yellow_pixels = yellow_count
            user_row = row_idx

    # Validate sufficient highlight pixels found
    if max_yellow_pixels < 100:  # Threshold to tune
        raise UserRowNotFoundError("No highlighted row detected")

    return user_row

def get_user_team(row_index):
    """Convert row index to team."""
    return 'blue' if row_index < 5 else 'red'
```

### Phase D: OCR Text Extraction

#### Character Name Extraction
```python
def extract_player_names(roster_region, num_rows=10):
    """
    OCR each of 10 player name regions.
    Returns: List of 10 character names
    """
    names = []
    row_height = roster_region.shape[0] // num_rows

    for row_idx in range(num_rows):
        # Define name region (left portion of row)
        y_start = row_idx * row_height
        y_end = y_start + row_height
        name_region = roster_region[y_start:y_end, 0:300]  # First 300px

        # Preprocess for OCR
        processed = preprocess_for_ocr(name_region)

        # OCR with Tesseract
        text = pytesseract.image_to_string(
            processed,
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        ).strip()

        # Fuzzy match correction against known database
        corrected_name = fuzzy_match_name(text)
        names.append(corrected_name)

    return names

def preprocess_for_ocr(image):
    """
    Standard OCR preprocessing pipeline.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize 2x for better OCR accuracy
    resized = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply adaptive threshold (inverted for white text on dark bg)
    thresh = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return thresh
```

#### Score Extraction
```python
def extract_scores(scoreboard_region, anchor_point):
    """
    Extract Red and Blue scores from top-center boxes.
    Returns: (blue_score, red_score)
    """
    # Define score box coordinates relative to anchor
    blue_box = (anchor_point[0] - 100, anchor_point[1] - 50, 80, 40)
    red_box = (anchor_point[0] + 20, anchor_point[1] - 50, 80, 40)

    # Extract and preprocess
    blue_region = extract_region(scoreboard_region, blue_box)
    red_region = extract_region(scoreboard_region, red_box)

    blue_processed = preprocess_for_ocr(blue_region)
    red_processed = preprocess_for_ocr(red_region)

    # OCR with digits-only mode
    blue_text = pytesseract.image_to_string(
        blue_processed,
        config='--psm 7 digits'
    ).strip()

    red_text = pytesseract.image_to_string(
        red_processed,
        config='--psm 7 digits'
    ).strip()

    # Parse and validate
    try:
        blue_score = int(blue_text)
        red_score = int(red_text)

        # Validate score range (0-500 in GW2 PvP)
        if not (0 <= blue_score <= 500 and 0 <= red_score <= 500):
            raise ValueError("Scores out of valid range")

        return blue_score, red_score
    except ValueError as e:
        raise ScoreExtractionError(f"Could not parse scores: {blue_text}, {red_text}") from e
```

### Phase E: Profession Detection

#### Icon-Based Classification
```python
def detect_professions(roster_region, num_rows=10):
    """
    Detect profession icons for each player.
    Uses template matching with pre-saved profession icons.
    Returns: List of 10 profession names
    """
    profession_templates = load_profession_templates()
    professions = []

    row_height = roster_region.shape[0] // num_rows

    for row_idx in range(num_rows):
        # Extract profession icon region (right side of name)
        y_start = row_idx * row_height
        y_end = y_start + row_height
        icon_region = roster_region[y_start:y_end, 300:350]  # Icon area

        best_match = None
        best_score = 0.0

        # Try matching against each profession template
        for prof_name, template in profession_templates.items():
            result = cv2.matchTemplate(icon_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_match = prof_name

        # Confidence threshold
        if best_score < 0.6:
            best_match = "Unknown"

        professions.append(best_match)

    return professions

# Profession list for GW2
PROFESSIONS = [
    "Guardian", "Warrior", "Engineer", "Ranger", "Thief",
    "Elementalist", "Mesmer", "Necromancer", "Revenant"
]
```

---

## 5. Application Architecture

### Directory Structure
```
gw2-pvp-tracker/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py           # Database schema definitions
│   │   ├── connection.py       # SQLite connection handler
│   │   └── queries.py          # Pre-defined SQL queries
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── capture.py          # Screenshot capture
│   │   ├── template_matcher.py # Scoreboard detection
│   │   ├── user_detector.py    # Highlight row detection
│   │   ├── ocr_engine.py       # Text extraction
│   │   └── profession_detector.py # Icon classification
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── stats.py            # Win rate calculations
│   │   └── briefing.py         # Tactical briefing generator
│   └── ui/
│       ├── __init__.py
│       ├── console.py          # Console output formatting
│       └── hotkeys.py          # Keyboard listener
├── templates/
│   ├── scoreboard_header.png  # User-provided template
│   └── professions/
│       ├── guardian.png
│       ├── warrior.png
│       └── ...                 # All 9 profession icons
├── screenshots/                # Auto-captured screenshots
├── tests/
│   ├── __init__.py
│   ├── test_vision.py
│   ├── test_database.py
│   └── fixtures/               # Test images
├── data/
│   └── pvp_tracker.db          # SQLite database
├── logs/
│   └── app.log                 # Application logs
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

### Core Module Descriptions

#### `main.py` - Application Entry Point
```python
"""
Main application orchestrator.
Handles:
- Hotkey registration
- Match workflow coordination
- Error handling and logging
"""
```

#### `config.py` - Configuration Management
```python
"""
Centralized configuration:
- Screen resolution
- OCR settings
- Template paths
- Database path
- Hotkey bindings
- CV confidence thresholds
"""
```

#### `database/models.py` - ORM Layer
```python
"""
SQLite schema definitions and CRUD operations.
Classes:
- Player
- Match
- MatchParticipant
"""
```

#### `vision/capture.py` - Screenshot Handler
```python
"""
Screen capture functionality:
- Detect game window
- Capture full screen
- Capture specific regions
- Save with timestamps
"""
```

#### `analysis/briefing.py` - Tactical Intelligence
```python
"""
Generate pre-match briefing:
- Team composition analysis
- Player threat assessment
- Historical performance lookup
- Formatted console output
"""
```

---

## 6. User Workflow

### Step 1: Application Launch
```bash
$ python src/main.py
```

**Console Output:**
```
╔══════════════════════════════════════════╗
║   GW2 PvP Tracker - Ready               ║
╚══════════════════════════════════════════╝

Hotkeys:
  F8 - Capture Match Start
  F9 - Capture Match End
  ESC - Exit Application

Listening for hotkeys...
```

### Step 2: Match Start (F8 Pressed)

**Process Flow:**
1. Capture 2 screenshots (full + roster crop)
2. Detect scoreboard anchor
3. Extract 10 player names via OCR
4. Detect user's highlighted row
5. Determine user's team (Blue/Red)
6. Extract professions for all players
7. Query database for player history
8. Generate tactical briefing

**Console Output:**
```
╔══════════════════════════════════════════╗
║   MATCH START - Tactical Briefing       ║
╚══════════════════════════════════════════╝

YOUR TEAM: Blue
Your Character: ExamplePlayer.1234

─────────────── BLUE TEAM ───────────────────
1. ExamplePlayer.1234 (Guardian) ★ YOU
   └─ No history

2. AllyOne (Thief)
   └─ 68% WR over 25 matches
   └─ Usually plays: Thief (18), Mesmer (7)

3. AllyTwo (Warrior)
   └─ 45% WR over 11 matches
   └─ Usually plays: Warrior (11)

4. AllyThree (Ranger)
   └─ NEW PLAYER (First time seen)

5. AllyFour (Necromancer)
   └─ 80% WR over 5 matches ⚠️ STRONG
   └─ Usually plays: Necromancer (5)

─────────────── RED TEAM (OPPONENTS) ────────
6. EnemyOne (Revenant)
   └─ 55% WR over 40 matches
   └─ Usually plays: Revenant (25), Guardian (15)

7. EnemyTwo (Elementalist)
   └─ 72% WR over 18 matches ⚠️ THREAT
   └─ Usually plays: Elementalist (18)

8. EnemyThree (Mesmer)
   └─ 30% WR over 10 matches

9. EnemyFour (Engineer)
   └─ NEW PLAYER (First time seen)

10. EnemyFive (Thief)
    └─ 50% WR over 4 matches

────────────────────────────────────────────
ANALYSIS:
  Blue Team Avg WR: 61.2% (excludes new players)
  Red Team Avg WR: 51.8% (excludes new players)

  Threats: EnemyTwo (72%), AllyFour (80%)

Press F9 when match ends to log results.
```

### Step 3: Match End (F9 Pressed)

**Process Flow:**
1. Capture 2 screenshots (full + roster crop)
2. Extract final scores (Blue/Red)
3. Determine winning team
4. Compare with user's team to get result (Win/Loss)
5. Insert match record into database
6. Update all 10 players' statistics
7. Display match summary

**Console Output:**
```
╔══════════════════════════════════════════╗
║   MATCH END - Results Logged            ║
╚══════════════════════════════════════════╝

Final Score: Blue 500 - Red 412

Result: VICTORY! 🎉

Match Details:
  Duration: 14:32
  Your Team: Blue
  Winning Team: Blue

Database Updated:
  ✓ Match #47 logged
  ✓ 10 player records updated

Your Stats:
  Overall Record: 23W - 24L (48.9% WR)
  Recent Form (Last 10): 6W - 4L

Ready for next match. Press F8 at match start.
```

---

## 7. Error Handling & Edge Cases

### Common Error Scenarios

#### 1. Scoreboard Not Found
**Cause:** User captures screenshot outside of scoreboard view
**Solution:**
```python
class ScoreboardNotFoundError(Exception):
    pass

try:
    anchor = find_scoreboard_anchor(screenshot)
except ScoreboardNotFoundError:
    print("❌ Error: Scoreboard not visible in screenshot.")
    print("Please ensure scoreboard is open (default: Tab key)")
    print("Then press F8 again.")
```

#### 2. User Highlight Not Detected
**Cause:** Screenshot timing before highlight renders
**Solution:**
- Add 0.5s delay after F8 press before capture
- Retry with adjusted HSV range
- Prompt user to provide their character name as fallback

#### 3. OCR Misreads
**Cause:** Low resolution, motion blur, unusual fonts
**Solution:**
```python
def fuzzy_match_name(ocr_text, threshold=80):
    """
    Correct OCR errors using fuzzy matching against known players.
    """
    from thefuzz import process

    # Get all known character names from database
    known_names = get_all_player_names()

    if not known_names:
        return ocr_text  # First time seeing any player

    # Find best match
    best_match, score = process.extractOne(ocr_text, known_names)

    if score >= threshold:
        return best_match
    else:
        return ocr_text  # Keep OCR result if no good match
```

#### 4. Score OCR Fails
**Cause:** Unusual UI elements overlapping score boxes
**Solution:**
```python
def extract_scores_with_retry(region, max_attempts=3):
    """
    Retry score extraction with different preprocessing.
    """
    for attempt in range(max_attempts):
        try:
            scores = extract_scores(region)
            validate_scores(scores)  # Check 0-500 range
            return scores
        except ScoreExtractionError:
            if attempt == max_attempts - 1:
                # Final attempt failed - prompt user
                print("❌ Could not read scores automatically.")
                blue = int(input("Enter Blue score (0-500): "))
                red = int(input("Enter Red score (0-500): "))
                return blue, red
            else:
                # Try different preprocessing
                region = adjust_preprocessing(region, attempt)
```

#### 5. Profession Detection Fails
**Cause:** New profession icon design after game update
**Solution:**
- Mark profession as "Unknown"
- Log warning to review templates
- Continue with match logging (profession is non-critical)

### Validation Rules

#### Match Data Validation
```python
def validate_match_data(match_data):
    """
    Ensure match data integrity before database insert.
    """
    assert len(match_data['players']) == 10, "Must have exactly 10 players"
    assert match_data['blue_score'] + match_data['red_score'] > 0, "Scores cannot both be zero"
    assert match_data['user_team'] in ['red', 'blue'], "Invalid user team"
    assert match_data['winning_team'] in ['red', 'blue'], "Invalid winning team"

    # Check for duplicate names
    names = [p['name'] for p in match_data['players']]
    assert len(names) == len(set(names)), "Duplicate player names detected"
```

---

## 8. Testing Strategy

### Unit Tests

#### `test_vision.py`
```python
def test_scoreboard_detection():
    """Test template matching across multiple scales."""
    screenshot = cv2.imread('tests/fixtures/match_start.png')
    template = cv2.imread('templates/scoreboard_header.png')
    anchor = find_scoreboard_anchor(screenshot, template)
    assert anchor is not None
    assert 0 <= anchor[0] < screenshot.shape[1]
    assert 0 <= anchor[1] < screenshot.shape[0]

def test_user_row_detection():
    """Test highlight detection."""
    roster = cv2.imread('tests/fixtures/roster_highlighted.png')
    row = find_user_row(roster)
    assert 0 <= row <= 9

def test_ocr_name_extraction():
    """Test name OCR accuracy."""
    name_region = cv2.imread('tests/fixtures/player_name.png')
    name = extract_player_name(name_region)
    assert len(name) > 0
    assert name == "ExpectedName.1234"

def test_score_extraction():
    """Test score OCR."""
    score_region = cv2.imread('tests/fixtures/scores.png')
    blue, red = extract_scores(score_region, (100, 100))
    assert 0 <= blue <= 500
    assert 0 <= red <= 500
```

#### `test_database.py`
```python
def test_player_insertion():
    """Test adding new player to database."""
    db = Database(':memory:')
    db.add_player('TestPlayer.1234')
    player = db.get_player('TestPlayer.1234')
    assert player['char_name'] == 'TestPlayer.1234'
    assert player['total_matches'] == 0

def test_match_logging():
    """Test complete match insertion."""
    db = Database(':memory:')
    match_data = create_test_match()
    match_id = db.log_match(match_data)
    assert match_id > 0

    # Verify player stats updated
    player = db.get_player('TestPlayer.1234')
    assert player['total_matches'] == 1

def test_win_rate_calculation():
    """Test statistics computation."""
    db = Database(':memory:')
    db.add_player('Player1')

    # Log 7 wins, 3 losses
    for _ in range(7):
        log_win(db, 'Player1')
    for _ in range(3):
        log_loss(db, 'Player1')

    winrate = db.get_win_rate('Player1')
    assert winrate == 70.0
```

### Integration Tests
```python
def test_full_match_workflow():
    """Test complete match start -> end flow."""
    # Simulate F8 press
    match_start_screenshots = capture_match_start()
    briefing = process_match_start(match_start_screenshots)
    assert len(briefing['players']) == 10
    assert briefing['user_team'] in ['red', 'blue']

    # Simulate F9 press
    match_end_screenshots = capture_match_end()
    result = process_match_end(match_end_screenshots)

    # Verify database update
    last_match = db.get_last_match()
    assert last_match['match_id'] is not None
```

### Test Fixtures Required
- `match_start.png`: Full screenshot with scoreboard visible
- `match_end.png`: Full screenshot with final scores
- `roster_highlighted.png`: Roster with user row highlighted
- `player_name.png`: Cropped player name region
- `scores.png`: Cropped score boxes region
- `profession_icons/`: Sample of each profession icon

---

## 9. Configuration File

### `config.yaml`
```yaml
# GW2 PvP Tracker Configuration

# Screen Capture
screen:
  monitor: 0  # Primary monitor
  capture_delay: 0.5  # Seconds to wait after hotkey press

# Computer Vision
cv:
  template_match_threshold: 0.7
  highlight_hsv_lower: [20, 50, 50]
  highlight_hsv_upper: [40, 255, 255]
  profession_match_threshold: 0.6
  scales: [0.8, 0.9, 1.0, 1.1, 1.2]

# OCR Settings
ocr:
  tesseract_path: "C:/Program Files/Tesseract-OCR/tesseract.exe"  # Windows
  name_psm: 7  # Single line mode
  score_psm: 7
  score_whitelist: "0123456789"
  name_whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. "

# Fuzzy Matching
fuzzy:
  name_match_threshold: 80  # 0-100

# Database
database:
  path: "data/pvp_tracker.db"
  backup_on_startup: true

# Hotkeys
hotkeys:
  match_start: "F8"
  match_end: "F9"
  exit: "esc"

# Paths
paths:
  templates: "templates/"
  screenshots: "screenshots/"
  logs: "logs/"

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "logs/app.log"
  max_size_mb: 10
  backup_count: 5

# UI
ui:
  console_width: 60
  show_ascii_art: true
  color_enabled: true
```

---

## 10. Installation & Setup Guide

### Prerequisites
1. **Python 3.10+**
2. **Tesseract OCR**
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Install to `C:\Program Files\Tesseract-OCR\`
   - Add to PATH

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/NathanZorndorf/gw2-pvp-tracker.git
cd gw2-pvp-tracker

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python src/database/init_db.py

# 5. Setup templates (one-time)
python src/setup_templates.py
# This will guide you through capturing the scoreboard template
```

### First-Time Setup Wizard
```bash
$ python src/setup_templates.py

╔══════════════════════════════════════════╗
║   GW2 PvP Tracker - First Time Setup    ║
╚══════════════════════════════════════════╝

Step 1: Scoreboard Template Capture
────────────────────────────────────
1. Launch Guild Wars 2
2. Enter a PvP match or use PvP lobby
3. Open the scoreboard (default: Tab key)
4. Press F8 when ready

Waiting for F8...

✓ Screenshot captured!

Please draw a box around the "Scoreboard" header text.
(Click and drag with mouse)

✓ Template saved to templates/scoreboard_header.png

Step 2: Profession Icons
────────────────────────────────────
The app includes default profession icons.
If they don't work after a game update, you can
recapture them using: python src/update_profession_templates.py

Setup complete! Run the tracker with:
  python src/main.py
```

---

## 11. Advanced Features (Future Enhancements)

### Phase 2 Features
1. **Web Dashboard**
   - Flask/FastAPI backend
   - Interactive charts (win rate trends)
   - Player search and filtering
   - Export to CSV/JSON

2. **Team Synergy Analysis**
   - Track win rates when playing WITH specific players
   - Identify strong duo/trio combinations

3. **Map-Specific Stats**
   - Detect which PvP map is being played
   - Track performance per map

4. **Build Detection**
   - OCR skill bar to infer build archetypes
   - Track meta build prevalence

5. **Live Match Overlay**
   - Real-time stats displayed over game
   - Uses transparent overlay window

### Phase 3 Features
1. **Cloud Sync**
   - Optional account to sync data across devices
   - Community aggregated statistics

2. **Automated VOD Clipping**
   - Record highlights of close matches
   - Auto-generate replay clips

3. **Voice Callouts**
   - TTS announcements of high-threat players
   - Audio briefing instead of console text

---

## 12. Performance Optimization

### Target Metrics
- **Screenshot to Briefing**: < 3 seconds
- **Match End Processing**: < 2 seconds
- **Database Query Time**: < 50ms
- **Memory Usage**: < 200MB

### Optimization Strategies

#### 1. Async Screenshot Processing
```python
import asyncio

async def process_match_start(screenshot_path):
    """Process match start in parallel tasks."""
    tasks = [
        asyncio.create_task(detect_anchor(screenshot_path)),
        asyncio.create_task(detect_user_row(screenshot_path)),
        asyncio.create_task(extract_names(screenshot_path)),
        asyncio.create_task(detect_professions(screenshot_path))
    ]
    results = await asyncio.gather(*tasks)
    return combine_results(results)
```

#### 2. Template Caching
```python
# Load templates once at startup, not per-match
TEMPLATES_CACHE = {
    'scoreboard': cv2.imread('templates/scoreboard_header.png'),
    'professions': {prof: cv2.imread(f'templates/professions/{prof}.png')
                    for prof in PROFESSIONS}
}
```

#### 3. Database Connection Pooling
```python
# Reuse database connection
class Database:
    _connection = None

    @classmethod
    def get_connection(cls):
        if cls._connection is None:
            cls._connection = sqlite3.connect('data/pvp_tracker.db')
        return cls._connection
```

#### 4. Region of Interest (ROI) Processing
```python
# Only process relevant screen regions
def extract_roster_region(screenshot, anchor):
    """Crop to roster area only, ignore rest of screen."""
    x, y = anchor
    roster = screenshot[y:y+400, x:x+600]  # Fixed roster size
    return roster
```

---

## 13. Security & Privacy

### Data Privacy
- **100% Local**: All data stored locally, no cloud upload
- **No Telemetry**: Application does not send usage data
- **No Account Required**: Fully offline functionality

### Screenshot Storage
- Screenshots contain **only** scoreboard region (no chat, party, guild info)
- Optional auto-deletion after 7 days to save disk space
- User can configure screenshot retention policy

### Database Security
- SQLite file has standard file permissions
- No sensitive personal data stored (only character names, which are public in-game)

---

## 14. Troubleshooting Guide

### Issue: "Scoreboard Not Found"
**Symptoms:** App cannot detect scoreboard in screenshot
**Solutions:**
1. Recapture scoreboard template with current UI scale
2. Check `config.yaml` → `cv.scales` includes your UI scale
3. Verify scoreboard is fully visible (not cut off)

### Issue: OCR Returns Gibberish
**Symptoms:** Player names are garbled or incorrect
**Solutions:**
1. Increase OCR preprocessing resize factor
2. Check Tesseract installation and PATH
3. Verify `config.yaml` → `ocr.tesseract_path` is correct
4. Try manual calibration: `python src/calibrate_ocr.py`

### Issue: User Row Not Detected
**Symptoms:** App cannot find highlighted row
**Solutions:**
1. Increase capture delay: `config.yaml` → `screen.capture_delay = 1.0`
2. Adjust HSV range in config to match your highlight color
3. Ensure you're capturing AFTER selecting scoreboard

### Issue: Database Locked Error
**Symptoms:** `sqlite3.OperationalError: database is locked`
**Solutions:**
1. Close any other database connections
2. Restart application
3. If persistent, run: `python src/database/repair_db.py`

### Issue: Hotkeys Not Working
**Symptoms:** F8/F9 do not trigger captures
**Solutions:**
1. Run application as Administrator (Windows)
2. Check if another app is using same hotkeys
3. Change hotkey bindings in `config.yaml`

---

## 15. Development Roadmap

### Milestone 1: Core Functionality (v0.1.0)
- [x] Database schema implementation
- [ ] Screenshot capture system
- [ ] Scoreboard template matching
- [ ] User row detection
- [ ] Basic OCR for names and scores
- [ ] Match logging to database

### Milestone 2: Intelligence Layer (v0.2.0)
- [ ] Player statistics queries
- [ ] Tactical briefing generator
- [ ] Win rate calculations
- [ ] Profession tracking

### Milestone 3: Polish & UX (v0.3.0)
- [ ] Error handling and retries
- [ ] Fuzzy name matching
- [ ] Console UI formatting
- [ ] Configuration file support
- [ ] Setup wizard

### Milestone 4: Testing & Stability (v0.4.0)
- [ ] Unit test suite
- [ ] Integration tests
- [ ] Performance profiling
- [ ] Bug fixes from user testing

### Milestone 5: Public Release (v1.0.0)
- [ ] Documentation
- [ ] Installation guide
- [ ] Video tutorial
- [ ] GitHub release

---

## 16. Contributing Guidelines

### Code Style
- **Formatter**: Black (line length 100)
- **Linter**: Pylint + Flake8
- **Type Hints**: Required for all functions
- **Docstrings**: Google style

### Git Workflow
1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add: description"`
4. Run tests: `pytest`
5. Push: `git push origin feature/your-feature`
6. Open Pull Request

### Commit Message Format
```
Type: Brief description

Detailed explanation (if needed)

Types: Add, Update, Fix, Remove, Refactor, Test, Docs
```

---

## 17. License & Credits

### License
MIT License - See LICENSE file

### Credits
- **Computer Vision**: OpenCV Team
- **OCR Engine**: Tesseract OCR
- **Game**: ArenaNet / Guild Wars 2

### Disclaimer
This is a third-party tool and is not affiliated with or endorsed by ArenaNet or NCSOFT. Use at your own risk. The tool does not modify game files or memory and operates purely through screenshot analysis.

---

## Appendix A: File Structure Reference

```
gw2-pvp-tracker/
├── src/                        # Source code
│   ├── main.py                 # Entry point (hotkey listener)
│   ├── config.py               # Config loader
│   ├── database/
│   │   ├── models.py           # SQLite schema & ORM
│   │   ├── connection.py       # DB connection manager
│   │   ├── queries.py          # Prepared SQL queries
│   │   └── init_db.py          # Database initialization
│   ├── vision/
│   │   ├── capture.py          # Screenshot capture (mss/PIL)
│   │   ├── template_matcher.py # Scoreboard anchor detection
│   │   ├── user_detector.py    # Highlight row finder
│   │   ├── ocr_engine.py       # Tesseract wrapper + preprocessing
│   │   └── profession_detector.py # Icon classification
│   ├── analysis/
│   │   ├── stats.py            # Win rate, match history queries
│   │   └── briefing.py         # Tactical briefing formatter
│   └── ui/
│       ├── console.py          # Rich console output
│       └── hotkeys.py          # Keyboard event handler
├── templates/
│   ├── scoreboard_header.png  # User-captured template
│   └── professions/            # 9 profession icons
├── screenshots/                # Auto-captured match screenshots
├── data/
│   └── pvp_tracker.db          # SQLite database
├── logs/
│   └── app.log                 # Application logs
├── tests/
│   ├── test_vision.py
│   ├── test_database.py
│   └── fixtures/               # Test images
├── config.yaml                 # User configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installer
├── README.md                   # Quick start guide
├── IMPLEMENTATION_PLAN.md      # This document
└── .gitignore
```

---

## Appendix B: Dependencies

### `requirements.txt`
```txt
# Core
opencv-python==4.8.1.78
pytesseract==0.3.10
numpy==1.24.3
Pillow==10.0.0

# OCR Enhancement
thefuzz==0.20.0
python-Levenshtein==0.21.1

# Screenshot Capture
mss==9.0.1

# Hotkeys
keyboard==0.13.5

# Database
# (sqlite3 is built-in)

# Configuration
PyYAML==6.0.1

# UI Enhancement (optional)
rich==13.5.2
colorama==0.4.6

# Development
pytest==7.4.0
black==23.7.0
pylint==2.17.5
flake8==6.1.0
```

---

## Appendix C: SQL Query Reference

### Frequently Used Queries

#### Get Player Stats
```sql
SELECT
    char_name,
    global_wins,
    global_losses,
    total_matches,
    ROUND(CAST(global_wins AS FLOAT) / total_matches * 100, 1) as win_rate,
    most_played_profession
FROM players
WHERE char_name = ?;
```

#### Get Player Match History
```sql
SELECT
    m.timestamp,
    m.match_result,
    m.red_score,
    m.blue_score,
    mp.team_color,
    mp.profession
FROM matches m
JOIN match_participants mp ON m.match_id = mp.match_id
WHERE mp.char_name = ?
ORDER BY m.timestamp DESC
LIMIT 10;
```

#### Get Most Played Profession for Player
```sql
SELECT
    profession,
    COUNT(*) as times_played
FROM match_participants
WHERE char_name = ?
GROUP BY profession
ORDER BY times_played DESC
LIMIT 1;
```

#### Get Win Rate vs Specific Player
```sql
-- Matches where both user and target player participated
WITH shared_matches AS (
    SELECT m.match_id, m.match_result
    FROM matches m
    JOIN match_participants mp1 ON m.match_id = mp1.match_id
    JOIN match_participants mp2 ON m.match_id = mp2.match_id
    WHERE mp1.char_name = ? -- user
    AND mp2.char_name = ?   -- target player
    AND mp1.team_color != mp2.team_color -- opponents
)
SELECT
    COUNT(*) as total_matches,
    SUM(CASE WHEN match_result = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(CAST(SUM(CASE WHEN match_result = 'win' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) as win_rate
FROM shared_matches;
```

---

## Appendix D: Screenshot Processing Examples

### Example 1: Full Workflow
```
Input: match_start_full.png (1920x1080)
↓
Step 1: Detect Scoreboard Anchor
  - Template match → (x=760, y=120)
↓
Step 2: Crop Roster Region
  - Extract roster → (760, 150, 600, 400)
↓
Step 3: Detect User Row
  - Yellow pixel density → Row 3 (Blue Team)
↓
Step 4: Extract Names
  - OCR 10 rows → ["Player1", "Player2", ...]
↓
Step 5: Detect Professions
  - Icon matching → ["Guardian", "Thief", ...]
↓
Step 6: Query Database
  - Lookup each player → Win rates, history
↓
Output: Tactical Briefing (Console)
```

### Example 2: Score Extraction
```
Input: match_end_roster.png
↓
Step 1: Locate Score Boxes
  - Blue box: (anchor_x - 100, anchor_y - 50, 80, 40)
  - Red box: (anchor_x + 20, anchor_y - 50, 80, 40)
↓
Step 2: Preprocess
  - Grayscale → Resize 2x → Threshold
↓
Step 3: OCR (Digits Only)
  - Blue: "500"
  - Red: "487"
↓
Step 4: Validate
  - Check 0 ≤ score ≤ 500
  - Determine winner: Blue
↓
Output: (500, 487, 'blue')
```

---

## Appendix E: Testing Checklist

### Pre-Release Testing

#### Functional Tests
- [ ] Match start capture works
- [ ] Match end capture works
- [ ] Scoreboard detection at UI scale 0.8
- [ ] Scoreboard detection at UI scale 1.0
- [ ] Scoreboard detection at UI scale 1.2
- [ ] User row detection (Blue team)
- [ ] User row detection (Red team)
- [ ] Name OCR accuracy > 95%
- [ ] Score OCR accuracy > 98%
- [ ] Profession detection accuracy > 90%
- [ ] Database insert/update works
- [ ] Win rate calculation correct
- [ ] Tactical briefing formatting

#### Edge Case Tests
- [ ] Screenshot captured without scoreboard open
- [ ] Screenshot captured mid-animation
- [ ] OCR with special characters in names
- [ ] Match with 2+ same professions
- [ ] Database with 1000+ players
- [ ] Database with 10,000+ matches
- [ ] Hotkey conflict with other apps
- [ ] Application crash recovery

#### Performance Tests
- [ ] Match start processing < 3s
- [ ] Match end processing < 2s
- [ ] Database query < 50ms
- [ ] Memory usage < 200MB

#### Usability Tests
- [ ] First-time setup wizard
- [ ] Error messages are clear
- [ ] Console output readable
- [ ] Configuration file validation

---

**END OF IMPLEMENTATION PLAN**

*Last Updated: 2026-01-09*
*Document Version: 1.0.0*
