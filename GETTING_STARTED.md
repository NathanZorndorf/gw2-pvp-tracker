# Getting Started with GW2 PvP Tracker

## ✅ What's Working Now

You have a fully functional database and screenshot capture system! Here's what you can do right now:

### 1. Manual Match Entry

Enter match data from your screenshots manually:

```bash
python manual_match_entry.py
```

This will:
- Log the match to the database
- Update all 12 players' win/loss records
- Show you the results

### 2. View Statistics

View player stats and match history:

```bash
python view_stats.py
```

Options:
- View all players sorted by win rate
- View match history
- Search for specific players

### 3. Component Testing

Test all the core components:

```bash
python test_components.py
```

This tests:
- Configuration loading
- Database operations
- Screenshot capture
- OCR engine

## 📊 Your First Match

You've already logged your first match! Here's what's in the database:

**Match #1** (2026-01-09)
- **Score**: Red 308 - Blue 500
- **Winner**: Blue Team
- **12 Players tracked** with their win/loss records

## 🗄️ Database Location

Your database is at: `data/pvp_tracker.db`

You can open it with any SQLite browser like:
- [DB Browser for SQLite](https://sqlitebrowser.org/)
- [SQLiteStudio](https://sqlitestudio.pl/)

## 📂 Project Structure

```
gw2-pvp-tracker/
├── data/
│   └── pvp_tracker.db          # Your match database
├── screenshots/                # Auto-captured screenshots
├── src/
│   ├── database/
│   │   └── models.py           # Database operations
│   ├── vision/
│   │   ├── capture.py          # Screenshot capture
│   │   └── ocr_engine.py       # OCR text extraction
│   └── config.py               # Configuration
├── manual_match_entry.py       # Enter matches manually
├── view_stats.py               # View statistics
├── test_components.py          # Test all components
└── config.yaml                 # Settings
```

## 🎯 Current Status

### ✅ Completed
- [x] Database schema and operations
- [x] Screenshot capture (full screen and regions)
- [x] OCR engine with Tesseract
- [x] Configuration system
- [x] Manual match entry tool
- [x] Statistics viewer
- [x] Test suite

### 🚧 In Progress
- [ ] OCR tuning for GW2 fonts (needs better preprocessing)
- [ ] Automatic player name extraction
- [ ] Score extraction from screenshots

### 📋 Next Steps
1. **Fine-tune OCR** for GW2's specific font and UI
2. **Template Matching** - Detect scoreboard automatically
3. **Highlight Detection** - Find user's row (yellow highlight)
4. **Profession Icons** - Recognize class icons
5. **Hotkey Integration** - F8/F9 for automatic capture
6. **Tactical Briefing** - Pre-match player analysis

## 🧪 Testing OCR

The OCR is installed but needs tuning. Current tests show:
- ✅ Tesseract is working
- ⚠️ Text extraction needs better preprocessing for GW2 fonts
- ⚠️ Need to fine-tune region coordinates

Test OCR on your screenshots:

```bash
python test_ocr_on_roster.py
```

This will create test images in `screenshots/` showing what Tesseract sees.

## 💡 Tips

### Adding More Matches

Run `manual_match_entry.py` again to add more matches. Each match:
- Updates player statistics
- Tracks win/loss records
- Records profession usage
- Maintains match history

### Viewing Specific Player Stats

```bash
python view_stats.py
# Choose option 3
# Enter player name (e.g., "Terminal Gearning")
```

### Database Queries

You can query the database directly:

```python
from src.database.models import Database

db = Database("data/pvp_tracker.db")

# Get player stats
player = db.get_player("Terminal Gearning")
print(f"{player['global_wins']}W - {player['global_losses']}L")

# Get win rate
win_rate, total = db.get_player_winrate("Terminal Gearning")
print(f"Win rate: {win_rate}% over {total} matches")

db.close()
```

## 🐛 Troubleshooting

### OCR Not Working
- Make sure Tesseract is installed at `C:\Program Files\Tesseract-OCR\`
- Or update `config.yaml` with the correct path

### Database Errors
- Check that `data/` directory exists
- Delete `data/pvp_tracker.db` to start fresh if needed

### Screenshot Issues
- Make sure you have permission to capture screen
- Try running as Administrator if captures fail

## 📚 Documentation

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Complete technical specification
- [README.md](README.md) - Project overview and usage examples
- [CLAUDE.md](CLAUDE.md) - Development guidelines

## 🎮 Workflow Once Complete

The end goal is:

1. **Match Start**: Press F8 in GW2 with scoreboard open
   - Captures screenshot
   - Extracts 10 player names
   - Detects your highlighted row
   - Shows tactical briefing in console

2. **Match End**: Press F9 when match ends
   - Captures final screenshot
   - Extracts scores
   - Logs match to database
   - Updates all player stats
   - Shows match result

3. **View Stats**: Run `view_stats.py` anytime
   - See your win/loss record
   - Track opponent win rates
   - Analyze match history

## 🚀 Quick Start Commands

```bash
# Enter a match manually
python manual_match_entry.py

# View your stats
python view_stats.py

# Test components
python test_components.py

# Test OCR on screenshots
python test_ocr_on_roster.py
```

## 📧 Support

If you encounter issues, check:
1. Is Tesseract installed? Run: `"C:/Program Files/Tesseract-OCR/tesseract.exe" --version`
2. Is the virtual environment activated? Run: `venv\Scripts\activate`
3. Are dependencies installed? Run: `pip list` and check for opencv-python, pytesseract, etc.

---

**Current Version**: v0.1.0 (Core Components)
**Last Updated**: 2026-01-09
