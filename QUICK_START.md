# Quick Start - Live Match Capture

## Step 1: Start Live Capture

**IMPORTANT:** On Windows, you need to run as Administrator for hotkeys to work.

1. Right-click on **Command Prompt** or **PowerShell**
2. Select **"Run as administrator"**
3. Navigate to the project folder:
   ```bash
   cd C:\Users\nzorn\Documents\gw2-pvp-tracker
   ```
4. Activate virtual environment:
   ```bash
   venv\Scripts\activate
   ```
5. Run live capture:
   ```bash
   python live_capture.py
   ```

Press Enter to start listening for hotkeys.

## Step 2: Play GW2 PvP

Keep the live_capture.py terminal running in the background.

When your match starts:
1. Open the scoreboard (press Tab in GW2)
2. Press **F8** to capture the start screenshot

When your match ends:
1. Open the scoreboard again
2. Press **F9** to capture the end screenshot

## Step 3: Log the Match

After capturing both screenshots, run:

```bash
python log_latest_match.py
```

This will:
- Find your most recent match screenshots
- Prompt you to enter the scores, team, and player names
- Log the match to the database
- Update all player statistics

**Note:** You'll need to manually type in player names and professions for now.
(OCR auto-extraction coming in a future update!)

## Step 4: View Your Stats

At any time, check your stats with:

```bash
python view_stats.py
```

## Files Created

Screenshots are saved to the `screenshots/` folder with timestamps:
- `match_start_YYYYMMDD_HHMMSS_full.png`
- `match_end_YYYYMMDD_HHMMSS_full.png`

## Tips

- Keep the live capture terminal visible on a second monitor
- You'll see confirmation messages when F8/F9 are pressed
- Screenshots are saved immediately - you don't need to wait
- Press ESC in the live capture terminal to exit

## Example Workflow

```bash
# Terminal 1: Start live capture
python live_capture.py

# [Play GW2, press F8 at start, F9 at end]

# Terminal 2: After match, log it
python manual_match_entry.py

# Check your stats
python view_stats.py
```

That's it! You're tracking your PvP matches!
