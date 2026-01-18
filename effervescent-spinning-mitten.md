# GW2 PvP Tracker: Python to Blish HUD C# Module Conversion Plan

## Executive Summary

This plan outlines the conversion of the standalone Python GW2 PvP Tracker into a C# Blish HUD module. The key challenge is replicating OCR-based opponent detection while adapting to Blish HUD's lifecycle and .NET ecosystem.

**Critical Finding**: GW2 API cannot provide real-time opponent data (only returns last 10 completed matches, no opponent names/professions). Therefore, **OCR is mandatory** for the core feature (real-time overlay showing opponent win rates).

**User Requirements**:
- Priority features: Real-time overlay + automatic match logging
- Data-only tracking (no screenshot archival)
- OCR-based approach (API insufficient)
- C# guidance needed (user is new to C#)

---

## Technology Stack

### Required NuGet Packages

| Package | Purpose | Rationale |
|---------|---------|-----------|
| **Tesseract** (4.x+) | OCR text extraction | Python app uses EasyOCR (no C# equivalent) + Tesseract fallback. Tesseract.NET provides character whitelisting & PSM modes |
| **OpenCvSharp4** + runtime.win | Image processing | Near 1:1 API mapping to OpenCV (easier Python→C# migration than EmguCV) |
| **Microsoft.Data.Sqlite** | Database | Modern, maintained by Microsoft. Compatible with Python app's SQLite schema |
| **FuzzySharp** | Fuzzy string matching | C# port of Python's `thefuzz` library (same Levenshtein algorithm) |

### Configuration Approach

- **Blish HUD Settings**: User-facing options (keybinds, thresholds, auto-close overlay)
- **Embedded Constants**: Fixed OCR region coordinates (from config.yaml) in C# classes
- **Resolution Scaling**: Detect monitor resolution, scale 4K coordinates proportionally

---

## Module Architecture

### Namespace Structure

```
Gw2PvpTracker/
├── Gw2PvpTrackerModule.cs           # Main module (lifecycle, keybinds)
├── Services/
│   ├── DatabaseService.cs            # SQLite operations
│   ├── OcrService.cs                 # OCR with preprocessing
│   ├── ProfessionDetectionService.cs # Template matching
│   ├── MatchProcessorService.cs      # Orchestrates extraction
│   └── MumbleLinkService.cs          # Map detection wrapper
├── Models/
│   ├── Player.cs, Match.cs, MatchParticipant.cs
│   ├── RegionConfig.cs               # OCR coordinates
│   └── MatchData.cs                  # Extracted data DTO
├── UI/
│   ├── OverlayPanel.cs              # Main overlay container
│   ├── PlayerCardControl.cs         # Individual player card
│   └── TeamPanel.cs                 # Red/Blue sections
└── Utils/
    ├── BoldTextDetector.cs          # HSV user detection
    ├── ImagePreprocessor.cs         # CLAHE, resize, threshold
    └── ScreenshotHelper.cs          # Screen capture
```

### Python → C# Class Mapping

| Python Class | C# Equivalent | File Path (Python) |
|-------------|---------------|-------------------|
| `Database` | `DatabaseService` | `src/database/models.py` |
| `OCREngine` | `OcrService` | `src/vision/ocr_engine.py` |
| `ProfessionDetector` | `ProfessionDetectionService` | `src/vision/profession_detector.py` |
| `MatchProcessor` | `MatchProcessorService` | `src/automation/match_processor.py` |
| `WinRateOverlay` | `OverlayPanel` | `src/ui/overlay.py` |

---

## Critical Algorithm Migrations

### 1. Bold Text Detection (User Identification)

**Source**: `src/vision/ocr_engine.py:548-634`

**Algorithm**: HSV color detection to identify highlighted user name (gold/bright vs gray)

**Python Implementation**:
```python
def _calculate_bold_score(self, region: np.ndarray) -> float:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Gold/Yellow Detection (Hue: 20-35, Sat: 50-255, Val: 150-255)
    gold_lower = np.array([20, 50, 150])
    gold_upper = np.array([35, 255, 255])
    gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)
    gold_ratio = np.sum(gold_mask > 0) / gold_mask.size

    # Brightness Detection (V channel max ~255 for user vs ~187 for others)
    _, _, v = cv2.split(hsv)
    max_v = np.max(v)

    brightness_score = 5.0 if max_v > 240 else (0.5 if max_v > 200 else 0.0)
    gold_score = gold_ratio * 10

    return max(gold_score, brightness_score)
```

**C# Equivalent** (`Utils/BoldTextDetector.cs`):
```csharp
using OpenCvSharp;

public class BoldTextDetector
{
    public float CalculateBoldScore(Mat region)
    {
        // Convert BGR to HSV
        using var hsv = new Mat();
        Cv2.CvtColor(region, hsv, ColorConversionCodes.BGR2HSV);

        // Gold/Yellow Detection
        var goldLower = new Scalar(20, 50, 150);
        var goldUpper = new Scalar(35, 255, 255);
        using var goldMask = new Mat();
        Cv2.InRange(hsv, goldLower, goldUpper, goldMask);

        int goldPixels = Cv2.CountNonZero(goldMask);
        float goldRatio = (float)goldPixels / goldMask.Total();

        // Brightness Detection (V channel)
        var channels = Cv2.Split(hsv);
        var vChannel = channels[2]; // V is index 2 in HSV

        Cv2.MinMaxLoc(vChannel, out double minV, out double maxV);
        vChannel.Dispose();
        foreach (var ch in channels) ch.Dispose();

        float brightnessScore = maxV > 240 ? 5.0f : (maxV > 200 ? 0.5f : 0.0f);
        float goldScore = goldRatio * 10;

        return Math.Max(goldScore, brightnessScore);
    }

    public (int BoldIndex, float Confidence) DetectBoldText(List<Mat> nameRegions)
    {
        var scores = nameRegions.Select(r => CalculateBoldScore(r)).ToList();

        int boldIndex = scores.IndexOf(scores.Max());
        float confidence = scores.OrderByDescending(s => s).First() -
                          scores.OrderByDescending(s => s).Skip(1).First();

        return (boldIndex, confidence);
    }
}
```

**C# Pattern Notes**:
- `np.array([20, 50, 150])` → `new Scalar(20, 50, 150)`
- `cv2.inRange(hsv, lower, upper)` → `Cv2.InRange(hsv, lower, upper, outputMask)`
- `np.sum(mask > 0)` → `Cv2.CountNonZero(mask)`
- `np.max(v)` → `Cv2.MinMaxLoc(v, out double min, out double max)`
- Always dispose OpenCvSharp Mats: `using var mat = ...` or `mat.Dispose()`

### 2. OCR Preprocessing Pipeline

**Source**: `src/vision/ocr_engine.py:101-141`

**Algorithm**: CLAHE → Resize (2x) → Adaptive Threshold → Denoise

**C# Implementation** (`Utils/ImagePreprocessor.cs`):
```csharp
using OpenCvSharp;

public class ImagePreprocessor
{
    public Mat PreprocessForOcr(Mat image, float resizeFactor = 2.0f)
    {
        // Convert to grayscale
        var gray = image.Channels() == 3
            ? image.CvtColor(ColorConversionCodes.BGR2GRAY)
            : image.Clone();

        // Resize for better OCR accuracy
        if (Math.Abs(resizeFactor - 1.0f) > 0.01f)
        {
            int newWidth = (int)(gray.Width * resizeFactor);
            int newHeight = (int)(gray.Height * resizeFactor);
            gray = gray.Resize(new Size(newWidth, newHeight), 0, 0, InterpolationFlags.Cubic);
        }

        // Adaptive threshold (white text on dark background)
        var thresh = new Mat();
        Cv2.AdaptiveThreshold(gray, thresh, 255, AdaptiveThresholdTypes.GaussianC,
                             AdaptiveThresholdTypes.BinaryInv, 11, 2);

        // Denoise
        var denoised = new Mat();
        Cv2.FastNlMeansDenoising(thresh, denoised, 10, 7, 21);

        gray.Dispose();
        thresh.Dispose();

        return denoised;
    }
}
```

**C# Pattern Notes**:
- `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` → `img.CvtColor(ColorConversionCodes.BGR2GRAY)`
- `cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)` → `img.Resize(new Size(w, h), 0, 0, InterpolationFlags.Cubic)`
- `cv2.adaptiveThreshold(...)` → `Cv2.AdaptiveThreshold(...)`
- `len(image.shape) == 3` → `image.Channels() == 3`

### 3. Profession Detection (Template Matching)

**Source**: `src/vision/profession_detector.py:126-172`

**Algorithm**: Grayscale → Letterbox → CLAHE → Canny → Circular Mask → Template Match

**C# Implementation** (`Services/ProfessionDetectionService.cs`):
```csharp
private Mat PreprocessIcon(Mat icon)
{
    // Grayscale
    var gray = icon.Channels() == 3
        ? icon.CvtColor(ColorConversionCodes.BGR2GRAY)
        : icon.Clone();

    // Letterbox resize to 32x32 (maintains aspect ratio, black padding)
    gray = LetterboxResize(gray, new Size(32, 32));

    // CLAHE enhancement
    using var clahe = Cv2.CreateCLAHE(clipLimit: 1.0, tileGridSize: new Size(2, 2));
    var enhanced = new Mat();
    clahe.Apply(gray, enhanced);

    // Canny edge detection
    var edges = new Mat();
    Cv2.Canny(enhanced, edges, 50, 150);

    // Circular mask (remove corner noise)
    var mask = Mat.Zeros(edges.Size(), MatType.CV_8UC1);
    var center = new Point(edges.Width / 2, edges.Height / 2);
    int radius = Math.Min(edges.Width, edges.Height) / 2;
    Cv2.Circle(mask, center, radius, Scalar.White, -1);

    var masked = new Mat();
    Cv2.BitwiseAnd(edges, edges, masked, mask);

    // Convert to RGB for template matching
    var result = masked.CvtColor(ColorConversionCodes.Gray2BGR);

    // Cleanup
    gray.Dispose(); enhanced.Dispose(); edges.Dispose();
    mask.Dispose(); masked.Dispose();

    return result;
}

public string DetectProfession(Mat iconRegion, out float confidence)
{
    var preprocessed = PreprocessIcon(iconRegion);

    string bestProfession = "Unknown";
    float bestScore = 0.0f;

    foreach (var (profession, refIcon) in _referenceIcons)
    {
        var result = new Mat();
        Cv2.MatchTemplate(preprocessed, refIcon, result, TemplateMatchModes.CCoeffNormed);

        Cv2.MinMaxLoc(result, out _, out double maxVal);
        result.Dispose();

        if (maxVal > bestScore)
        {
            bestScore = (float)maxVal;
            bestProfession = profession;
        }
    }

    confidence = bestScore;
    preprocessed.Dispose();

    return bestScore >= 0.2f ? bestProfession : "Unknown"; // Low threshold from config
}
```

**C# Pattern Notes**:
- `cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))` → `Cv2.CreateCLAHE(clipLimit: 1.0, tileGridSize: new Size(2, 2))`
- `cv2.Canny(img, 50, 150)` → `Cv2.Canny(img, output, 50, 150)`
- `cv2.matchTemplate(target, ref, cv2.TM_CCOEFF_NORMED)` → `Cv2.MatchTemplate(target, ref, result, TemplateMatchModes.CCoeffNormed)`

### 4. OCR Text Extraction

**C# Implementation** (`Services/OcrService.cs`):
```csharp
using Tesseract;
using FuzzySharp;

public class OcrService : IDisposable
{
    private TesseractEngine _engine;
    private ImagePreprocessor _preprocessor;

    public OcrService(string tessdataPath)
    {
        _engine = new TesseractEngine(tessdataPath, "eng", EngineMode.Default);
        _preprocessor = new ImagePreprocessor();
    }

    public string ExtractPlayerName(Mat region, List<string> knownNames = null)
    {
        var preprocessed = _preprocessor.PreprocessForOcr(region, resizeFactor: 2.0f);

        // Convert OpenCvSharp Mat to Bitmap for Tesseract
        var bitmap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(preprocessed);

        // Character whitelist (supports accented characters)
        _engine.SetVariable("tessedit_char_whitelist",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
            "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØŒÙÚÛÜÝŸß" +
            "àáâãäåæçèéêëìíîïñòóôõöøœùúûüýÿ. ");

        using var page = _engine.Process(bitmap, PageSegMode.SingleLine);
        string rawText = page.GetText().Trim();

        preprocessed.Dispose();
        bitmap.Dispose();

        // Fuzzy match against known names if available
        if (knownNames?.Any() == true && !string.IsNullOrEmpty(rawText))
        {
            var match = Process.ExtractOne(rawText, knownNames);
            if (match.Score >= 80) // Threshold from config.yaml
            {
                return match.Value;
            }
        }

        return rawText;
    }

    public void Dispose() => _engine?.Dispose();
}
```

**C# Pattern Notes**:
- Python `thefuzz.process.extractOne(query, choices)` → C# `FuzzySharp.Process.ExtractOne(query, choices)`
- Python `if knownNames and rawText:` → C# `if (knownNames?.Any() == true && !string.IsNullOrEmpty(rawText))`
- Python `with engine:` → C# `using var engine = ...` or `engine.Dispose()`

---

## Database Migration Strategy

### Schema Compatibility

**Key Principle**: Use **identical SQLite schema** as Python app for seamless data migration.

**Tables** (from `src/database/models.py:44-128`):
```sql
players (
  char_name TEXT PRIMARY KEY,
  global_wins INTEGER DEFAULT 0,
  global_losses INTEGER DEFAULT 0,
  total_matches INTEGER DEFAULT 0,
  most_played_profession TEXT,
  first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CHECK (total_matches = global_wins + global_losses)
)

matches (
  match_id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  red_score INTEGER NOT NULL,
  blue_score INTEGER NOT NULL,
  arena_type TEXT,
  map_name TEXT,
  winning_team TEXT NOT NULL CHECK(winning_team IN ('red', 'blue')),
  user_team TEXT NOT NULL CHECK(user_team IN ('red', 'blue')),
  user_char_name TEXT NOT NULL,
  screenshot_start_path TEXT,  -- Set to NULL (data-only tracking)
  screenshot_end_path TEXT,    -- Set to NULL
  FOREIGN KEY (user_char_name) REFERENCES players(char_name)
)

match_participants (
  participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  char_name TEXT NOT NULL,
  profession TEXT NOT NULL,
  team_color TEXT NOT NULL CHECK(team_color IN ('red', 'blue')),
  is_user BOOLEAN DEFAULT 0,
  FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
  FOREIGN KEY (char_name) REFERENCES players(char_name),
  UNIQUE(match_id, char_name)
)

-- View for win rate calculation
CREATE VIEW player_winrates AS
SELECT char_name, global_wins, global_losses, total_matches,
       ROUND(CAST(global_wins AS FLOAT) / NULLIF(total_matches, 0) * 100, 1) as win_rate_pct,
       most_played_profession
FROM players WHERE total_matches > 0
```

**C# Implementation** (`Services/DatabaseService.cs`):
```csharp
using Microsoft.Data.Sqlite;

public class DatabaseService : IDisposable
{
    private SqliteConnection _connection;

    public DatabaseService(string dbPath)
    {
        _connection = new SqliteConnection($"Data Source={dbPath}");
        _connection.Open();
        InitializeSchema(); // Creates tables if not exist
    }

    public void LogMatch(MatchData match)
    {
        using var transaction = _connection.BeginTransaction();

        try
        {
            // Insert match
            var matchCmd = _connection.CreateCommand();
            matchCmd.CommandText = @"
                INSERT INTO matches
                (red_score, blue_score, arena_type, map_name, winning_team,
                 user_team, user_char_name, screenshot_start_path, screenshot_end_path)
                VALUES (@red, @blue, @arena, @map, @winner, @userTeam, @userName, NULL, NULL)
            ";

            matchCmd.Parameters.AddWithValue("@red", match.RedScore);
            matchCmd.Parameters.AddWithValue("@blue", match.BlueScore);
            // ... other parameters

            matchCmd.ExecuteNonQuery();
            long matchId = _connection.LastInsertRowId;

            // Insert participants + update player stats
            foreach (var player in match.Players)
            {
                AddOrUpdatePlayer(player.Name);

                var partCmd = _connection.CreateCommand();
                partCmd.CommandText = @"
                    INSERT INTO match_participants
                    (match_id, char_name, profession, team_color, is_user)
                    VALUES (@matchId, @name, @prof, @team, @isUser)
                ";

                partCmd.Parameters.AddWithValue("@matchId", matchId);
                partCmd.Parameters.AddWithValue("@name", player.Name);
                partCmd.Parameters.AddWithValue("@prof", player.Profession);
                partCmd.Parameters.AddWithValue("@team", player.Team);
                partCmd.Parameters.AddWithValue("@isUser", player.IsUser ? 1 : 0);
                partCmd.ExecuteNonQuery();

                // Update player stats (wins/losses)
                bool won = player.Team == (match.BlueScore > match.RedScore ? "blue" : "red");
                UpdatePlayerStats(player.Name, won, player.Profession);
            }

            transaction.Commit();
        }
        catch
        {
            transaction.Rollback();
            throw;
        }
    }

    public (float WinRate, int TotalMatches) GetPlayerWinRate(string charName)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT win_rate_pct, total_matches FROM player_winrates WHERE char_name = @name";
        cmd.Parameters.AddWithValue("@name", charName);

        using var reader = cmd.ExecuteReader();
        if (reader.Read())
        {
            return (reader.IsDBNull(0) ? 0f : (float)reader.GetDouble(0), reader.GetInt32(1));
        }
        return (0f, 0);
    }

    public void Dispose() => _connection?.Dispose();
}
```

**Migration Path**:
1. User copies `pvp_tracker.db` from Python app's `data/` folder to Blish module's data directory
2. C# module reads existing data seamlessly (schema identical)
3. New matches logged with `screenshot_start_path` and `screenshot_end_path` set to NULL

---

## Blish HUD Module Integration

### Module Lifecycle

**Source**: [Anatomy of a Module | Blish HUD](https://blishhud.com/docs/modules/overview/anatomy/)

**C# Implementation** (`Gw2PvpTrackerModule.cs`):
```csharp
using Blish_HUD;
using Blish_HUD.Modules;
using Blish_HUD.Settings;

[Export(typeof(Module))]
public class Gw2PvpTrackerModule : Module
{
    private DatabaseService _databaseService;
    private OcrService _ocrService;
    private ProfessionDetectionService _professionService;
    private MatchProcessorService _matchProcessor;
    private OverlayPanel _overlayPanel;

    private SettingEntry<KeyBinding> _matchStartKeybind;
    private SettingEntry<KeyBinding> _matchEndKeybind;
    private SettingEntry<bool> _overlayAutoClose;

    [ImportingConstructor]
    public Gw2PvpTrackerModule([Import("ModuleParameters")] ModuleParameters moduleParameters)
        : base(moduleParameters)
    {
        // Constructor - minimal initialization only
    }

    protected override void DefineSettings(SettingCollection settings)
    {
        // Define user-configurable settings
        _matchStartKeybind = settings.DefineSetting("MatchStartKey",
            new KeyBinding(Keys.F8),
            () => "Match Start",
            () => "Capture match start (scoreboard appears)");

        _matchEndKeybind = settings.DefineSetting("MatchEndKey",
            new KeyBinding(Keys.F9),
            () => "Match End",
            () => "Capture match end (log results)");

        _overlayAutoClose = settings.DefineSetting("OverlayAutoClose",
            true,
            () => "Auto-close Overlay",
            () => "Hide overlay when F9 pressed");
    }

    protected override async Task LoadAsync()
    {
        // Initialize services
        var dbPath = Path.Combine(DirectoriesManager.GetFullDirectoryPath("data"), "pvp_tracker.db");
        _databaseService = new DatabaseService(dbPath);

        var tessdataPath = Path.Combine(ModuleDirectory, "tessdata");
        _ocrService = new OcrService(tessdataPath);

        _professionService = new ProfessionDetectionService();
        await _professionService.LoadReferenceIconsAsync(ContentsManager);

        _matchProcessor = new MatchProcessorService(
            _databaseService, _ocrService, _professionService,
            GameService.Gw2Mumble);

        // Create overlay UI
        _overlayPanel = new OverlayPanel();

        // Register keybinds
        _matchStartKeybind.Value.Activated += OnMatchStart;
        _matchEndKeybind.Value.Activated += OnMatchEnd;
    }

    protected override void Update(GameTime gameTime)
    {
        // Called every frame - use sparingly
        // Could poll MumbleLink for map changes here if needed
    }

    protected override void Unload()
    {
        // Cleanup
        _matchStartKeybind.Value.Activated -= OnMatchStart;
        _matchEndKeybind.Value.Activated -= OnMatchEnd;

        _databaseService?.Dispose();
        _ocrService?.Dispose();
        _overlayPanel?.Dispose();
    }

    private async void OnMatchStart(object sender, EventArgs e)
    {
        if (!GameService.GameIntegration.Gw2Instance.Gw2HasFocus) return;

        // Capture screenshot
        await Task.Delay(300); // 0.3s delay from config
        var screenshot = CaptureGameWindow();

        // Process match start
        var players = await _matchProcessor.ProcessMatchStartAsync(screenshot);

        // Show overlay with win rates
        _overlayPanel.ShowPlayers(players);

        screenshot.Dispose();
    }

    private async void OnMatchEnd(object sender, EventArgs e)
    {
        if (!GameService.GameIntegration.Gw2Instance.Gw2HasFocus) return;

        // Hide overlay if enabled
        if (_overlayAutoClose.Value)
        {
            _overlayPanel.Hide();
        }

        // Capture end screenshot
        await Task.Delay(300);
        var endScreenshot = CaptureGameWindow();

        // Process and log match
        await _matchProcessor.ProcessMatchEndAsync(endScreenshot);

        endScreenshot.Dispose();
    }

    private Mat CaptureGameWindow()
    {
        // Use GDI+ to capture GW2 window
        var gw2Handle = GameService.GameIntegration.Gw2Instance.Gw2WindowHandle;

        // Get window bounds
        GetWindowRect(gw2Handle, out var rect);
        int width = rect.Right - rect.Left;
        int height = rect.Bottom - rect.Top;

        // Capture via BitBlt
        var bitmap = new Bitmap(width, height);
        using (var g = Graphics.FromImage(bitmap))
        {
            g.CopyFromScreen(rect.Left, rect.Top, 0, 0, new Size(width, height));
        }

        // Convert Bitmap to OpenCvSharp Mat
        return OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap);
    }

    [DllImport("user32.dll")]
    private static extern bool GetWindowRect(IntPtr hwnd, out RECT lpRect);

    [StructLayout(LayoutKind.Sequential)]
    private struct RECT { public int Left, Top, Right, Bottom; }
}
```

**Lifecycle Flow**:
1. **Constructor**: Minimal initialization (assign module parameters)
2. **DefineSettings**: Define user settings (keybinds, auto-close)
3. **LoadAsync**: Initialize services, load resources, register keybinds
4. **Update**: Called every frame (use for MumbleLink polling if needed)
5. **Unload**: Dispose services, unregister events

---

## UI Implementation (Blish HUD Overlay)

### Overlay Panel Structure

**C# Implementation** (`UI/OverlayPanel.cs`):
```csharp
using Blish_HUD.Controls;

public class OverlayPanel : Container
{
    private FlowPanel _redTeamPanel;
    private FlowPanel _blueTeamPanel;
    private List<PlayerCardControl> _playerCards = new();

    public OverlayPanel()
    {
        Parent = GameService.Graphics.SpriteScreen;
        Size = new Point(420, 600);
        Location = new Point(
            (GameService.Graphics.SpriteScreen.Width - 420) / 2,
            (GameService.Graphics.SpriteScreen.Height - 600) / 2
        );

        BackgroundColor = Color.FromArgb(230, 20, 20, 30); // Semi-transparent dark
        Visible = false;

        BuildUI();
    }

    private void BuildUI()
    {
        // Title
        new Label
        {
            Text = "Match Analysis",
            Font = GameService.Content.DefaultFont32,
            Parent = this,
            Location = new Point(20, 20)
        };

        // Red team panel
        _redTeamPanel = new FlowPanel
        {
            Parent = this,
            Location = new Point(20, 60),
            Size = new Point(180, 500),
            FlowDirection = ControlFlowDirection.SingleTopToBottom
        };

        // Blue team panel
        _blueTeamPanel = new FlowPanel
        {
            Parent = this,
            Location = new Point(220, 60),
            Size = new Point(180, 500),
            FlowDirection = ControlFlowDirection.SingleTopToBottom
        };
    }

    public void ShowPlayers(List<PlayerStats> players)
    {
        // Clear existing cards
        _playerCards.ForEach(c => c.Dispose());
        _playerCards.Clear();

        // Separate by team
        var redPlayers = players.Where(p => p.Team == "red").ToList();
        var bluePlayers = players.Where(p => p.Team == "blue").ToList();

        // Create cards
        foreach (var player in redPlayers)
        {
            _playerCards.Add(new PlayerCardControl(player) { Parent = _redTeamPanel });
        }

        foreach (var player in bluePlayers)
        {
            _playerCards.Add(new PlayerCardControl(player) { Parent = _blueTeamPanel });
        }

        Show();
    }
}
```

**Player Card Control** (`UI/PlayerCardControl.cs`):
```csharp
public class PlayerCardControl : Panel
{
    public PlayerCardControl(PlayerStats player)
    {
        Size = new Point(180, 80);
        BackgroundColor = Color.FromArgb(150, 40, 40, 50);

        // Profession icon
        new Image
        {
            Parent = this,
            Location = new Point(5, 5),
            Size = new Point(32, 32),
            Texture = GameService.Content.GetTexture($"profession_icons/{player.Profession}.png")
        };

        // Player name
        new Label
        {
            Parent = this,
            Location = new Point(45, 5),
            Text = player.Name,
            Font = GameService.Content.DefaultFont14
        };

        // Win rate with stars
        string stars = new string('★', (int)(player.WinRate / 20)) +
                      new string('☆', 5 - (int)(player.WinRate / 20));
        new Label
        {
            Parent = this,
            Location = new Point(45, 28),
            Text = $"{stars} {player.WinRate:F1}%",
            TextColor = GetWinRateColor(player.WinRate)
        };

        // Match count
        new Label
        {
            Parent = this,
            Location = new Point(45, 48),
            Text = $"({player.TotalMatches} games)",
            Font = GameService.Content.DefaultFont12
        };
    }

    private Color GetWinRateColor(float winRate)
    {
        if (winRate >= 60) return Color.FromArgb(100, 255, 100); // Green
        if (winRate >= 50) return Color.FromArgb(255, 255, 100); // Yellow
        if (winRate >= 40) return Color.FromArgb(255, 180, 100); // Orange
        return Color.FromArgb(255, 100, 100); // Red
    }
}
```

---

## Configuration Management

### OCR Region Coordinates

**Source**: `config.yaml:76-174` (4K resolution coordinates)

**C# Implementation** (`Models/RegionConfig.cs`):
```csharp
public static class RegionConfig
{
    // Base coordinates for 4K (3840x2160)
    public static readonly Dictionary<string, ArenaRegions> BaseRegions = new()
    {
        ["ranked"] = new ArenaRegions
        {
            RedTeamNames = new NameRegion
            {
                XStart = 1192, XEnd = 1799,
                YStart = 682, RowHeight = 58
            },
            BlueTeamNames = new NameRegion
            {
                XStart = 2055, XEnd = 2670,
                YStart = 678, RowHeight = 58
            },
            RedScoreBox = new Rect(1760, 555, 152, 111),
            BlueScoreBox = new Rect(1944, 555, 144, 111)
        },
        ["unranked"] = new ArenaRegions
        {
            RedTeamNames = new NameRegion
            {
                XStart = 1200, XEnd = 1803,
                YStart = 712, RowHeight = 58
            },
            BlueTeamNames = new NameRegion
            {
                XStart = 2038, XEnd = 2656,
                YStart = 710, RowHeight = 58
            },
            RedScoreBox = new Rect(1702, 588, 207, 109),
            BlueScoreBox = new Rect(1939, 588, 211, 109)
        }
    };

    // Scale coordinates based on actual resolution
    public static ArenaRegions GetScaledRegions(string arenaType, int screenHeight)
    {
        float scale = (float)screenHeight / 2160f; // 2160 = 4K height
        var baseRegions = BaseRegions[arenaType];

        return new ArenaRegions
        {
            RedTeamNames = Scale(baseRegions.RedTeamNames, scale),
            BlueTeamNames = Scale(baseRegions.BlueTeamNames, scale),
            RedScoreBox = Scale(baseRegions.RedScoreBox, scale),
            BlueScoreBox = Scale(baseRegions.BlueScoreBox, scale)
        };
    }

    private static NameRegion Scale(NameRegion region, float scale) => new()
    {
        XStart = (int)(region.XStart * scale),
        XEnd = (int)(region.XEnd * scale),
        YStart = (int)(region.YStart * scale),
        RowHeight = (int)(region.RowHeight * scale)
    };
}

public class ArenaRegions
{
    public NameRegion RedTeamNames { get; set; }
    public NameRegion BlueTeamNames { get; set; }
    public Rect RedScoreBox { get; set; }
    public Rect BlueScoreBox { get; set; }
}

public class NameRegion
{
    public int XStart { get; set; }
    public int XEnd { get; set; }
    public int YStart { get; set; }
    public int RowHeight { get; set; }
}
```

---

## Development Phases

### Phase 1: Foundation & Database (Week 1)
**Goal**: Establish C# project and database layer

**Tasks**:
1. Create Blish HUD module project from template
2. Add NuGet packages
3. Implement `DatabaseService` with schema initialization
4. Test database compatibility with existing Python db

**Deliverables**: Working database service that can read/write Python app's SQLite db

### Phase 2: Image Processing & OCR (Week 2)
**Goal**: Port OCR extraction pipeline

**Tasks**:
1. Implement `ImagePreprocessor` (CLAHE, resize, threshold)
2. Implement `OcrService` with Tesseract.NET
3. Port bold text detection (`BoldTextDetector`)
4. Test OCR accuracy against Python test screenshots

**Deliverables**: OCR service achieving 90%+ accuracy on player names

**Challenge**: Tesseract may be less accurate than Python's EasyOCR (97%+). Mitigation:
- Use aggressive fuzzy matching (threshold 70-80)
- Multi-pass OCR with varied preprocessing
- Consider Windows.Media.OCR as fallback

### Phase 3: Profession Detection (Week 3)
**Goal**: Port template matching

**Tasks**:
1. Implement `ProfessionDetectionService` with preprocessing
2. Port circular masking and letterbox resize
3. Load reference icons from module contents
4. Test accuracy (target: 97.5% from Python benchmarks)

**Deliverables**: Profession detection with 95%+ accuracy

### Phase 4: Match Processing (Week 4)
**Goal**: Orchestrate full extraction pipeline

**Tasks**:
1. Implement `MatchProcessorService` (coordinates OCR + profession + database)
2. Implement arena type detection (ranked vs unranked)
3. Implement screen capture utility
4. Test end-to-end flow with manual screenshots

**Deliverables**: Screenshot → database logging working

### Phase 5: Blish HUD Integration (Week 5)
**Goal**: Integrate with Blish lifecycle

**Tasks**:
1. Implement module lifecycle methods
2. Register keybinds (F8/F9)
3. Test in-game with Blish HUD
4. Handle edge cases (GW2 not focused, screenshot fails)

**Deliverables**: Module loads in Blish HUD, keybinds work

### Phase 6: UI & Overlay (Week 6)
**Goal**: Build real-time overlay

**Tasks**:
1. Implement `OverlayPanel` with team layout
2. Implement `PlayerCardControl` with win rates
3. Load profession icon textures
4. Integrate with match processor

**Deliverables**: Functional overlay matching Python app's UI

### Phase 7: Polish & Testing (Week 7)
**Goal**: Settings, optimization, testing

**Tasks**:
1. Settings panel (keybinds, thresholds, auto-close)
2. Logging and error handling
3. Performance optimization (async OCR, parallel processing)
4. Beta testing with real matches

**Deliverables**: Production-ready module

### Phase 8: Release (Week 8)
**Goal**: Documentation and publishing

**Tasks**:
1. User documentation (README, settings guide)
2. Package module (.bhm file)
3. Submit to Blish HUD repo

**Deliverables**: Published module

---

## C# Learning Guide (Python → C#)

### Basic Syntax

| Python | C# |
|--------|-----|
| `def method(self, arg):` | `public void Method(Type arg) { }` |
| `self.field = value` | `this.field = value;` or `_field = value;` |
| `if not value:` | `if (!value)` or `if (value == null)` |
| `with open(file) as f:` | `using (var stream = File.Open(path))` |
| `try...except Exception as e:` | `try { } catch (Exception ex)` |
| `for item in items:` | `foreach (var item in items)` |

### Collections & LINQ

| Python | C# (LINQ) |
|--------|-----------|
| `[x for x in items if x > 0]` | `items.Where(x => x > 0).ToList()` |
| `sorted(items, key=lambda x: x.field)` | `items.OrderBy(x => x.field).ToList()` |
| `max(items)` | `items.Max()` |
| `sum(items)` | `items.Sum()` |
| `any(items)` | `items.Any()` |
| `items[0]` | `items.First()` or `items[0]` |

### Async/Await

| Python | C# |
|--------|-----|
| `async def func():` | `async Task FuncAsync()` |
| `await asyncio.gather(...)` | `await Task.WhenAll(...)` |
| `await func()` | `await FuncAsync()` |

### NumPy → OpenCvSharp

| NumPy/OpenCV (Python) | OpenCvSharp (C#) |
|----------------------|------------------|
| `np.array([1, 2, 3])` | `new Scalar(1, 2, 3)` |
| `np.zeros((h, w))` | `Mat.Zeros(new Size(w, h), MatType.CV_8UC1)` |
| `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | `img.CvtColor(ColorConversionCodes.BGR2HSV)` |
| `cv2.inRange(hsv, lower, upper)` | `Cv2.InRange(hsv, lower, upper, mask)` |
| `np.sum(mask > 0)` | `Cv2.CountNonZero(mask)` |
| `img.shape` | `img.Size()` (Width/Height properties) |

### Memory Management

**Python**: Automatic garbage collection
**C#**: Explicit disposal for unmanaged resources (Mats, database connections)

```csharp
// Option 1: using statement (preferred)
using var mat = new Mat();
// Mat automatically disposed when out of scope

// Option 2: Manual disposal
var mat = new Mat();
try
{
    // Use mat
}
finally
{
    mat.Dispose();
}
```

---

## Potential Challenges & Solutions

### Challenge 1: Tesseract Accuracy vs EasyOCR
**Problem**: Tesseract may achieve lower accuracy than Python's EasyOCR (97%+)

**Solutions**:
1. Heavy fuzzy matching (threshold 70-80 vs 80)
2. Multi-pass OCR with varied preprocessing
3. Windows.Media.OCR as fallback
4. User correction in overlay (allow editing names)

### Challenge 2: Screen Capture in Overlay Context
**Problem**: Blish HUD is an overlay - capturing GW2 content may be tricky

**Solutions**:
1. Use GDI+ BitBlt to capture entire GW2 window
2. Add 100ms delay before capture (let Blish overlay hide)
3. Blish overlay is semi-transparent, shouldn't interfere with OCR regions

### Challenge 3: Cross-Resolution Support
**Problem**: Hardcoded 4K coordinates - users have 1080p/1440p monitors

**Solutions**:
1. Auto-scale coordinates: `scaleFactor = screenHeight / 2160f`
2. Test on multiple resolutions (1080p, 1440p, 4K)
3. Add calibration UI for manual adjustment (future enhancement)

### Challenge 4: Performance (OCR is Expensive)
**Problem**: Processing 10 names + 2 scores + 10 professions = 300-500ms

**Solutions**:
1. Async processing on background thread
2. Parallel OCR (process 10 names simultaneously)
3. Cache preprocessed profession icons
4. GPU Tesseract if available

---

## Critical Files Reference

### Python Source Files (for migration)

| File | Lines | Purpose |
|------|-------|---------|
| `src/database/models.py` | 44-128 | Database schema (replicate exactly in C#) |
| `src/vision/ocr_engine.py` | 548-634 | Bold text detection algorithm |
| `src/vision/ocr_engine.py` | 101-141 | OCR preprocessing pipeline |
| `src/vision/profession_detector.py` | 126-172 | Profession detection preprocessing |
| `src/automation/match_processor.py` | Full file | Match extraction orchestration |
| `config.yaml` | 76-174 | OCR region coordinates (4K) |
| `config.yaml` | 175-186 | Profession detection parameters |

### C# Files to Create

| File | Purpose |
|------|---------|
| `Gw2PvpTrackerModule.cs` | Main module entry point |
| `Services/DatabaseService.cs` | SQLite operations |
| `Services/OcrService.cs` | Tesseract OCR wrapper |
| `Services/ProfessionDetectionService.cs` | Template matching |
| `Services/MatchProcessorService.cs` | Orchestration |
| `Utils/BoldTextDetector.cs` | HSV user detection |
| `Utils/ImagePreprocessor.cs` | CLAHE, resize, threshold |
| `Models/RegionConfig.cs` | OCR coordinates |
| `UI/OverlayPanel.cs` | Main overlay UI |
| `UI/PlayerCardControl.cs` | Player card widget |

---

## Verification Plan

### Unit Testing
1. **DatabaseService**: CRUD operations, schema creation, transaction rollback
2. **BoldTextDetector**: HSV detection on synthetic test images
3. **ImagePreprocessor**: Preprocessing output validation
4. **ProfessionDetectionService**: Template matching on reference dataset

### Integration Testing
1. **MatchProcessorService**: Full pipeline with real match screenshots
2. **Database Compatibility**: Load Python app's db, verify data integrity

### Manual In-Game Testing
1. Run 10+ matches (ranked + unranked)
2. Verify user detection accuracy (bold text)
3. Verify OCR accuracy (player names, scores)
4. Verify profession detection accuracy
5. Verify overlay display correctness
6. Verify database logging correctness

### Performance Benchmarks
- OCR Speed: <200ms for 10 player names
- Profession Detection: <100ms for 10 icons
- Total Match Processing: <500ms for F8 or F9 press

---

## Resources

### Documentation
- [Blish HUD Module Getting Started](https://blishhud.com/docs/modules/overview/getting-started/)
- [Blish HUD Module Anatomy](https://blishhud.com/docs/modules/overview/anatomy/)
- [GW2 Web API Guide | Blish HUD](https://blishhud.com/docs/modules/guides/gw2api/)
- [Example Blish HUD Module (GitHub)](https://github.com/blish-hud/Example-Blish-HUD-Module)
- [Guild Wars 2 API: PvP Games](https://wiki.guildwars2.com/wiki/API:2/pvp/games)

### Libraries
- [OpenCvSharp Documentation](https://github.com/shimat/opencvsharp)
- [Tesseract.NET (NuGet)](https://www.nuget.org/packages/Tesseract/)
- [FuzzySharp (NuGet)](https://www.nuget.org/packages/FuzzySharp/)
- [Microsoft.Data.Sqlite (NuGet)](https://www.nuget.org/packages/Microsoft.Data.Sqlite/)

---

## Summary

This plan converts the Python GW2 PvP Tracker into a C# Blish HUD module by:

1. **Replicating core algorithms** in C# using OpenCvSharp (bold detection, OCR preprocessing, profession matching)
2. **Maintaining database compatibility** with identical SQLite schema
3. **Integrating with Blish HUD** lifecycle (LoadAsync, keybinds, overlay UI)
4. **Using OCR** for real-time opponent detection (GW2 API insufficient)
5. **Providing C# guidance** for Python developers (syntax, LINQ, async/await, memory management)

**Next Steps**: Review plan, then begin Phase 1 (Foundation & Database).
