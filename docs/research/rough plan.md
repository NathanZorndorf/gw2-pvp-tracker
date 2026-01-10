1. Vision & Core Logic
A local Python application that automates PvP performance tracking. It uses computer vision to identify players and scores, removing manual data entry.

The Golden Rule: The app never asks who the user is or what team they are on. It detects the "Highlight" row to find the user and compares "Red vs. Blue" scores to find the winner.

2. Database Schema (SQLite)
players: char_name (PK), global_wins, global_losses

matches: match_id (PK), timestamp, red_score, blue_score, winning_team, user_team

match_participants: match_id (FK), char_name (FK), profession, team_color

3. Computer Vision Pipeline
A. Anchor & Scaling
Target: Use cv2.matchTemplate to find the "Scoreboard" header.

Scalability: Search at scales 0.8 to 1.2 to account for different "Interface Size" settings in GW2.

B. User Detection (No-Name Dependency)
Logic: Scan the 10 roster rows for the unique golden-yellow highlight pixels.

Team Assignment:

Rows 1-5 (Blue) | Rows 6-10 (Red)

If the highlight is in Row 3, the user is on the Blue Team.

C. Automated Scoring
OCR Targets: Two specific boxes at the top-center of the scoreboard.

Config: Use Tesseract digits only mode.

Winner: If blue_score > red_score, Blue wins.

4. User Workflow
Step 1: Match Start (F8)
Capture: Takes a screenshot and finds the scoreboard anchor.

Identify: OCRs all 10 names and identifies the User's Row via color.

Lookup: Prints a "Tactical Briefing" to the console:

Blue Team (Your Team):

Player X: 65% WR (12 games) - Usually plays Thief.

Player Y: 40% WR (5 games).

Red Team (Opponents):

Player Z: 80% WR (20 games) - High Threat!

Step 2: Match End (F9)
Final Scan: Captures the final scores and confirms the winner.

Auto-Log: If the user was Blue and Blue score was 500, the app logs a Win.

Update: Increments global_wins/losses for all 10 players in the database.

5. Coding Instructions for AI Assistant
Markdown

Role: Senior Python Developer (Computer Vision & SQLite)

Task: Build a GW2 PvP Tracker using the following constraints:
1. Detection: Use cv2.matchTemplate for a user-provided 'template.png' of the scoreboard header.
2. Highlight: Identify the user's row by finding the row with the highest density of yellow pixels (HSV range: H:20-40, S:50+, V:50+).
3. OCR Cleanup: 
   - Grayscale -> Resize 2x -> Inverted Threshold.
   - Use 'thefuzz' library to correct common OCR typos (e.g., '1' instead of 'l').
4. Database: Implement the 3-table schema provided.
5. Automation: Use 'keyboard' library for F8/F9 triggers.