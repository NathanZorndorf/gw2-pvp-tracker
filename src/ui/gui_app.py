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
)

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from database.models import Database
from .qt_panels import WinRatePanel, RankingsWidget
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

        self.header_layout = QHBoxLayout()
        self.status_label = QLabel("Last Match Analysis")
        self.header_layout.addWidget(self.status_label)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_latest_match)
        self.header_layout.addWidget(self.refresh_btn)
        
        self.layout.addLayout(self.header_layout)

        self.analysis_panel = WinRatePanel()
        self.layout.addWidget(self.analysis_panel)

        self.load_latest_match()

    def load_latest_match(self):
        db = Database()
        recent = db.get_recent_matches(1)
        if not recent:
            self.status_label.setText("No matches found in database.")
            return
            
        match = recent[0]
        match_id = match['match_id']
        
        # Update status label with match info
        map_name = match.get('map_name', 'Unknown Map')
        result = "Win" if (match['winning_team'] == match.get('user_team')) else "Loss" # Note: check DB schema for user_team logic or infer
        # Simplified for now:
        self.status_label.setText(f"Match #{match_id} - {map_name} - {match['winning_team']} Won")

        cursor = db.connection.cursor()
        cursor.execute("SELECT char_name, profession, team_color, is_user FROM match_participants WHERE match_id = ?", (match_id,))
        parts = cursor.fetchall()

        players = []
        for p in parts:
            name = p[0]
            team = p[2]
            is_user = bool(p[3])
            # For analysis panel we might want current winrates
            win_rate, total = db.get_player_winrate(name)
            players.append({
                'name': name,
                'team': team,
                'win_rate': win_rate,
                'total_matches': total,
                'is_user': is_user,
            })

        self.analysis_panel.show_players(players)
        db.close()


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
