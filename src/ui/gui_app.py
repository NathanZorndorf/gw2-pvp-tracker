from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTabWidget,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QGridLayout,
    QComboBox,
    QInputDialog,
    QSizePolicy,
)

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from database.models import Database
from .qt_panels import WinRatePanel, RankingsWidget, ClickableLabel
from .analytics_tab import AnalyticsWidget
from .styles import COLORS, get_stylesheet


class DatabaseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = Database()
        self.layout = QVBoxLayout(self)

        # Players Table
        self.players_table = QTableWidget()
        self.layout.addWidget(QLabel("Players"))
        self.layout.addWidget(self.players_table)

        # Matches Table
        self.matches_table = QTableWidget()
        self.layout.addWidget(QLabel("Recent Matches"))
        self.layout.addWidget(self.matches_table)

        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        btn_layout.addWidget(refresh_btn)

        del_btn = QPushButton("Delete Selected Match")
        del_btn.clicked.connect(self.delete_selected_match)
        btn_layout.addWidget(del_btn)

        self.layout.addLayout(btn_layout)

        self.refresh()

    def refresh(self):
        # Load players
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT char_name, global_wins, global_losses, total_matches, most_played_profession FROM players ORDER BY total_matches DESC LIMIT 200")
        rows = cursor.fetchall()

        self.players_table.setColumnCount(5)
        self.players_table.setHorizontalHeaderLabels(["Name", "Wins", "Losses", "Matches", "Top Profession"])
        self.players_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                item = QTableWidgetItem(str(v) if v is not None else "")
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.players_table.setItem(r, c, item)

        # Load recent matches
        cursor.execute("SELECT match_id, timestamp, red_score, blue_score, map_name, winning_team, user_char_name FROM matches ORDER BY timestamp DESC LIMIT 200")
        rows = cursor.fetchall()
        self.matches_table.setColumnCount(7)
        self.matches_table.setHorizontalHeaderLabels(["ID", "Timestamp", "Red", "Blue", "Map", "Winner", "User"])
        self.matches_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                item = QTableWidgetItem(str(v) if v is not None else "")
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.matches_table.setItem(r, c, item)

    def delete_selected_match(self):
        selected = self.matches_table.currentRow()
        if selected < 0:
            QMessageBox.information(self, "Delete Match", "No match selected")
            return
        match_id_item = self.matches_table.item(selected, 0)
        if not match_id_item:
            QMessageBox.information(self, "Delete Match", "Could not read match id")
            return
        match_id = int(match_id_item.text())

        ok = QMessageBox.question(self, "Confirm Delete", f"Delete match {match_id}? This will adjust player stats.")
        if ok == QMessageBox.StandardButton.Yes:
            self.db.delete_match(match_id)
            self.refresh()


class MatchAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.db = Database()
        self.current_match_id = None
        self.current_red_score = 0
        self.current_blue_score = 0

        # Header: Selector + Refresh
        self.header_layout = QHBoxLayout()
        
        self.match_selector = QComboBox()
        self.match_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.match_selector.currentIndexChanged.connect(self.on_match_selected)
        self.header_layout.addWidget(self.match_selector)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.header_layout.addWidget(self.refresh_btn)
        
        self.layout.addLayout(self.header_layout)

        # Score Section
        self.score_layout = QHBoxLayout()
        self.score_layout.setAlignment(Qt.AlignCenter)
        
        # Display scores as plain labels (read-only)
        self.red_score_btn = QLabel("Red: 0")
        self.red_score_btn.setStyleSheet(f"color: {COLORS['team_red']}; font-weight: bold; font-size: 14px;")
        
        self.blue_score_btn = QLabel("Blue: 0")
        self.blue_score_btn.setStyleSheet(f"color: {COLORS['team_blue']}; font-weight: bold; font-size: 14px;")
        
        self.score_layout.addWidget(self.red_score_btn)
        self.score_layout.addWidget(QLabel("-", alignment=Qt.AlignCenter)) 
        self.score_layout.addWidget(self.blue_score_btn)
        
        self.layout.addLayout(self.score_layout)

        self.analysis_panel = WinRatePanel(
            on_profession_change=self.update_player_profession,
            on_name_change=self.update_player_name
        )
        self.layout.addWidget(self.analysis_panel)

        self.refresh_data()

    def refresh_data(self):
        current_id = self.match_selector.currentData()
        
        self.match_selector.blockSignals(True)
        self.match_selector.clear()
        
        matches = self.db.get_recent_matches(50)
        
        index_to_set = 0
        for i, m in enumerate(matches):
            map_name = m.get('map_name', 'Unknown')
            # Determine result from winning_team and user_team
            # We need to fetch user_team if it's not in the dict returned by get_recent_matches
            # Based on previous code: result = "Win" if (match['winning_team'] == match.get('user_team'))
            user_team = m.get('user_team')
            if not user_team:
                # Fallback if query didn't return user_team (check query in db class later if needed)
                # Assuming get_recent_matches returns user_team (it wasn't in the original select I saw earlier?)
                # Wait, original select: SELECT match_id, timestamp, red_score, blue_score, map_name, winning_team, user_char_name
                # It does NOT verify user_team column usage in get_recent_matches.
                # I'll check if user_team is available. If not, can't determine win/loss easily without query.
                # For UI list, I'll just show Match ID and Map.
                result_str = ""
            else:
                 result_str = "[W]" if m['winning_team'] == user_team else "[L]"
            
            text = f"#{m['match_id']} {result_str} {map_name} ({m['timestamp']})"
            self.match_selector.addItem(text, m['match_id'])
            
            if current_id and m['match_id'] == current_id:
                index_to_set = i
        
        self.match_selector.setCurrentIndex(index_to_set)
        self.match_selector.blockSignals(False)
        
        if self.match_selector.count() > 0:
            self.on_match_selected(index_to_set)
        else:
             self.score_layout.setEnabled(False)

    def on_match_selected(self, index):
        match_id = self.match_selector.itemData(index)
        if match_id is not None:
             self.load_match(match_id)
             self.score_layout.setEnabled(True)

    def load_match(self, match_id):
        self.current_match_id = match_id
        
        cursor = self.db.connection.cursor()
        
        # Get scores
        cursor.execute("SELECT red_score, blue_score FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()
        if row:
            self.current_red_score = row[0]
            self.current_blue_score = row[1]
            self.red_score_btn.setText(f"Red: {row[0]}")
            self.blue_score_btn.setText(f"Blue: {row[1]}")
            
        # Get participants
        cursor.execute("SELECT char_name, profession, team_color, is_user FROM match_participants WHERE match_id = ?", (match_id,))
        parts = cursor.fetchall()

        players = []
        for p in parts:
            name = p[0]
            profession = p[1]
            team = p[2]
            is_user = bool(p[3])
            win_rate, total = self.db.get_player_winrate(name)
            players.append({
                'name': name,
                'profession': profession,
                'team': team,
                'win_rate': win_rate,
                'total_matches': total,
                'is_user': is_user,
            })

        self.analysis_panel.show_players(players)

    def update_player_name(self, old_name, new_name):
        if not self.current_match_id: return
        if not new_name.strip(): return 

        try:
            cursor = self.db.connection.cursor()
            
            # Ensure new player exists in players table to satisfy FK
            cursor.execute("SELECT 1 FROM players WHERE char_name = ?", (new_name,))
            if not cursor.fetchone():
                 cursor.execute("INSERT INTO players (char_name) VALUES (?)", (new_name,))

            # Update match_participants
            cursor.execute("""
                UPDATE match_participants 
                SET char_name = ? 
                WHERE match_id = ? AND char_name = ?
            """, (new_name, self.current_match_id, old_name))
            
            self.db.connection.commit()
            self.load_match(self.current_match_id)
            
        except Exception as e:
            QMessageBox.warning(self, "Update Failed", f"Failed to update name: {e}")

    def update_player_profession(self, player_name, new_profession):
        if not self.current_match_id: return
            
        try:
            cursor = self.db.connection.cursor()
            cursor.execute("""
                UPDATE match_participants 
                SET profession = ? 
                WHERE match_id = ? AND char_name = ?
            """, (new_profession, self.current_match_id, player_name))
            
            self.db.connection.commit()
            self.load_match(self.current_match_id)
            
        except Exception as e:
            QMessageBox.warning(self, "Update Failed", f"Failed to update profession: {e}")

    def load_latest_match(self):
        # Legacy stub if called from elsewhere, redirect to refresh
        self.refresh_data()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GW2 PvP Tracker")
        # Apply dark background similar to overlay
        self.setStyleSheet(f"background: {COLORS['bg_main']};")

        self.capture_process: Optional[subprocess.Popen] = None
        self.start_live_capture()

        tabs = QTabWidget()
        tabs.addTab(MatchAnalysisWidget(), "Match Analysis")
        tabs.addTab(AnalyticsWidget(), "Analytics")
        tabs.addTab(DatabaseWidget(), "Database")
        tabs.addTab(RankingsWidget(), "Rankings")

        self.setCentralWidget(tabs)

    def start_live_capture(self):
        # Launch live_capture.py as a subprocess
        def _find_live_capture_script() -> str:
            """Search upward from this file for src/automation/live_capture.py and return its path."""
            here = Path(__file__).resolve()
            # Try to resolve relative to repository root
            # gui_app is in src/ui. Repo root is parent.parent.
            repo_root = here.parent.parent.parent
            candidate = repo_root / 'src' / 'automation' / 'live_capture.py'
            
            if candidate.exists():
                return str(candidate)
            
            # Fallback for different CWD
            return 'src/automation/live_capture.py'

        script = _find_live_capture_script()
        if not os.path.exists(script):
            QMessageBox.warning(self, "Warning", f"live_capture.py not found at {script}. Auto-capture not started.")
            return

        try:
            args = [sys.executable, script]
            # No map arg, map selection happens in the capture process now
            self.capture_process = subprocess.Popen(args)
            print(f"Live capture started (pid {self.capture_process.pid})")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to start live capture: {e}")

    def closeEvent(self, event):
        if self.capture_process:
            print("Terminating live capture process...")
            try:
                self.capture_process.terminate()
                self.capture_process.wait(timeout=2)
            except Exception:
                try:
                    self.capture_process.kill()
                except Exception:
                    pass
        event.accept()


def run_gui():
    app = QApplication(sys.argv)
    # Apply centralized stylesheet for consistent dark theme and contrast
    app.setStyleSheet(get_stylesheet())
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
