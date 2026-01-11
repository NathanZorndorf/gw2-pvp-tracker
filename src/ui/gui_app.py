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


class DashboardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = Database()
        self.layout = QVBoxLayout(self)

        self.summary_label = QLabel("Loading stats...")
        self.layout.addWidget(self.summary_label)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        self.layout.addWidget(refresh_btn)

        self.refresh()

    def refresh(self):
        # Aggregate wins/losses
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT SUM(global_wins) as wins, SUM(global_losses) as losses FROM players")
        row = cursor.fetchone()
        wins = row[0] or 0
        losses = row[1] or 0

        total = wins + losses
        win_pct = (wins / total * 100) if total > 0 else 0.0

        self.summary_label.setText(f"Total matches: {total}  Wins: {wins}  Losses: {losses}  Win%: {win_pct:.1f}")

        # Plot pie chart
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pie([wins, losses], labels=[f"Wins ({wins})", f"Losses ({losses})"], autopct="%1.1f%%", colors=["#2ca02c", "#d62728"])
        ax.set_title("Overall Win/Loss")
        self.canvas.draw()


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
        cursor.execute("SELECT match_id, timestamp, red_score, blue_score, winning_team, user_char_name FROM matches ORDER BY timestamp DESC LIMIT 200")
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


class LiveCaptureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: Optional[subprocess.Popen] = None
        self.layout = QVBoxLayout(self)

        self.status_label = QLabel("Live capture: stopped")
        self.layout.addWidget(self.status_label)

        self.start_btn = QPushButton("Start Live Capture")
        self.start_btn.clicked.connect(self.start_capture)
        self.layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Live Capture")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)
        self.layout.addWidget(self.stop_btn)

        # Quick last-match analysis using DB
        self.analysis_panel = WinRatePanel()
        self.layout.addWidget(self.analysis_panel)

        btn = QPushButton("Show Last Match Analysis")
        btn.clicked.connect(self.show_last_match_analysis)
        self.layout.addWidget(btn)

        # Map selection grid (loads images from docs/maps)
        self.selected_map: Optional[str] = None
        self.map_buttons = {}
        maps_area = QScrollArea()
        maps_area.setWidgetResizable(True)
        maps_container = QWidget()
        self.maps_layout = QGridLayout(maps_container)
        maps_area.setWidget(maps_container)
        self.layout.addWidget(QLabel("Select Map (optional):"))
        self.layout.addWidget(maps_area)

        self._load_map_thumbnails()

    def start_capture(self):
        # Launch live_capture.py as a subprocess
        def _find_live_capture_script() -> str:
            """Search upward from this file for live_capture.py and return its path.

            This is more robust when the package layout or working dir changes.
            """
            here = Path(__file__).resolve()
            # Check current and parent directories up to repository root
            for parent in here.parents:
                candidate = parent / 'live_capture.py'
                if candidate.exists():
                    return str(candidate)
            # Fallback: assume repo root is two levels up from src/ui
            fallback = here.parents[2] / 'live_capture.py'
            return str(fallback)

        script = _find_live_capture_script()
        if not os.path.exists(script):
            QMessageBox.critical(self, "Error", f"live_capture.py not found at {script}")
            return

        if self.process:
            QMessageBox.information(self, "Live Capture", "Already running")
            return

        try:
            args = [sys.executable, script]
            if getattr(self, 'selected_map', None):
                args += ['--map', self.selected_map]
            # Also set env var for subprocess as fallback
            env = os.environ.copy()
            if getattr(self, 'selected_map', None):
                env['LIVE_CAPTURE_MAP'] = self.selected_map

            self.process = subprocess.Popen(args, env=env)
            self.status_label.setText(f"Live capture: running (pid {self.process.pid})")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Start Failed", str(e))

    def stop_capture(self):
        if not self.process:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None
        self.status_label.setText("Live capture: stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def show_last_match_analysis(self):
        db = Database()
        recent = db.get_recent_matches(1)
        if not recent:
            QMessageBox.information(self, "No Matches", "No matches in database")
            return
        match = recent[0]
        match_id = match['match_id']
        cursor = db.connection.cursor()
        cursor.execute("SELECT char_name, profession, team_color, is_user FROM match_participants WHERE match_id = ?", (match_id,))
        parts = cursor.fetchall()

        players = []
        for p in parts:
            name = p[0]
            team = p[2]
            is_user = bool(p[3])
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

    def _load_map_thumbnails(self):
        # Look for images under project/docs/maps
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        maps_dir = os.path.join(root, 'docs', 'maps')
        if not os.path.isdir(maps_dir):
            return

        exts = ('.png', '.jpg', '.jpeg', '.webp')
        files = [f for f in os.listdir(maps_dir) if f.lower().endswith(exts)]
        files.sort()

        col_count = 4
        row = col = 0
        for fname in files:
            path = os.path.join(maps_dir, fname)
            pix = QPixmap(path)
            if pix.isNull():
                continue
            icon = QIcon(pix.scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            btn = QPushButton()
            btn.setIcon(icon)
            btn.setIconSize(QSize(160, 90))
            btn.setToolTip(fname)
            btn.setCheckable(True)

            def make_cb(name, button):
                def cb():
                    # Uncheck others
                    for b in self.map_buttons.values():
                        if b is not button:
                            b.setChecked(False)
                    if button.isChecked():
                        self.selected_map = name
                    else:
                        self.selected_map = None
                return cb

            cb = make_cb(fname, btn)
            btn.clicked.connect(cb)
            self.map_buttons[fname] = btn

            self.maps_layout.addWidget(btn, row, col)
            col += 1
            if col >= col_count:
                col = 0
                row += 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GW2 PvP Tracker — Dashboard")
        # Apply dark background similar to overlay
        self.setStyleSheet(f"background: {COLORS['bg_main']};")

        tabs = QTabWidget()
        tabs.addTab(DashboardWidget(), "Dashboard")
        tabs.addTab(DatabaseWidget(), "Database")
        tabs.addTab(LiveCaptureWidget(), "Live Capture")
        tabs.addTab(RankingsWidget(), "Rankings")

        self.setCentralWidget(tabs)


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
