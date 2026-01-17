from __future__ import annotations

from typing import List

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QScrollArea,
    QPushButton,
    QDialog,
    QGridLayout,
    QInputDialog,
    QLineEdit,
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QSize, Signal
import os
import json
from pathlib import Path

# Re-import styles to ensure COLORS is defined
from .styles import COLORS, get_star_count, SYMBOLS, get_winrate_color, FONTS
from .confidence import format_winrate
from database.models import Database
from config import Config
from analysis.win_rate_utils import get_display_win_rate

class ClickableLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

class ProfessionSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Profession")
        self.setStyleSheet(f"background: {COLORS['bg_main']}; color: {COLORS['text_primary']};")
        self.layout = QGridLayout(self)
        self.selected_profession = None
        
        # Calculate paths relative to this file
        root_path = Path(__file__).parents[2]
        icons_path = root_path / "data/reference-icons/icons-raw"
        json_path = root_path / "data/professions.json"
        
        # Load professions structure
        ordered_professions = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                ordered_professions = data.get('professions', [])
        except Exception as e:
            print(f"Error loading professions.json: {e}")

        # Map available icons
        available_icons = {}
        if icons_path.exists():
            for icon_file in icons_path.glob("*.png"):
                available_icons[icon_file.stem] = icon_file

        placed_icons = set()
        current_row = 0

        def create_btn(name, icon_file, r, c):
            btn = QPushButton()
            btn.setIcon(QIcon(str(icon_file)))
            btn.setIconSize(QSize(32, 32))
            btn.setToolTip(name)
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, p=name: self.select_profession(p))
            self.layout.addWidget(btn, r, c)

        # 1. Place organized professions from JSON
        for prof_entry in ordered_professions:
            col = 0
            
            # Core profession
            base_name = prof_entry['name']
            if base_name in available_icons:
                create_btn(base_name, available_icons[base_name], current_row, col)
                placed_icons.add(base_name)
            
            col += 1
            
            # Elite specializations
            for spec_name in prof_entry.get('elite_specializations', []):
                if spec_name in available_icons:
                    create_btn(spec_name, available_icons[spec_name], current_row, col)
                    placed_icons.add(spec_name)
                col += 1
            
            current_row += 1

        # 2. Place any remaining icons that weren't in the JSON
        remaining_keys = sorted([k for k in available_icons.keys() if k not in placed_icons and k != 'Unknown'])
        
        if remaining_keys:
            if placed_icons:
                current_row += 1 # Gap if needed, or just next row

            col = 0
            max_cols = 6
            for name in remaining_keys:
                create_btn(name, available_icons[name], current_row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    current_row += 1

    def select_profession(self, profession):
        self.selected_profession = profession
        self.accept()


def stars_display(win_rate: float, total_matches: int) -> str:
    if total_matches == 0:
        return "-"
    count = get_star_count(win_rate)
    filled = SYMBOLS['star_filled'] * count
    empty = SYMBOLS['star_empty'] * (5 - count)
    # Use colored spans so stars pop on dark background
    return f"<span style='color:{COLORS['star_filled']}'>{filled}</span>" + f"<span style='color:{COLORS['star_empty']}; opacity:0.7'>{empty}</span>"


class PlayerCardWidget(QWidget):
    def __init__(self, name: str, team: str, win_rate: float, total_matches: int, is_user: bool = False, profession: str = None, on_profession_change=None, on_name_change=None, parent=None):
        super().__init__(parent)
        self.name = name
        self.team = team
        self.win_rate = win_rate
        self.total_matches = total_matches
        self.is_user = is_user
        self.profession = profession
        self.on_profession_change = on_profession_change
        self.on_name_change = on_name_change

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)

        team_label = QLabel(f"[{self.team[0].upper()}]")
        team_label.setFixedWidth(28)
        team_label.setStyleSheet(f"color: {COLORS['team_red'] if self.team=='red' else COLORS['team_blue']}; font-weight:600;")
        layout.addWidget(team_label)

        stars_label = QLabel()
        stars_label.setTextFormat(Qt.RichText)
        stars_label.setText(stars_display(self.win_rate, self.total_matches))
        stars_label.setFixedWidth(100)
        layout.addWidget(stars_label)

        winrate_text = "NEW" if self.total_matches == 0 else format_winrate(self.win_rate, self.total_matches)
        winrate_label = QLabel(winrate_text)
        winrate_label.setFixedWidth(70)
        # color winrate appropriately
        wr_color = COLORS['winrate_new'] if self.total_matches == 0 else get_winrate_color(self.win_rate)
        winrate_label.setStyleSheet(f"color: {wr_color}; font-family: {FONTS['stats'][0]}; font-size: {FONTS['stats'][1]}px;")
        layout.addWidget(winrate_label)

        matches = QLabel(f"({self.total_matches})")
        matches.setFixedWidth(50)
        matches.setStyleSheet(f"color: {COLORS['text_secondary']}; font-family: {FONTS['small'][0]}; font-size: {FONTS['small'][1]}px;")
        layout.addWidget(matches)

        # Icon (Clickable Button)
        icon_btn = QPushButton()
        icon_btn.setFixedSize(32, 32)
        icon_btn.setFlat(True)
        icon_btn.setCursor(Qt.PointingHandCursor)
        
        # Determine icon path
        icon_path = None
        if self.profession:
            # Try raw icons first (colored)
            p = Path(f"data/reference-icons/icons-raw/{self.profession}.png")
            if p.exists():
                icon_path = str(p)
            else:
                # Fallback to white or specific mapped name if needed
                p = Path(f"data/reference-icons/icons-white/{self.profession}.png") 
                if p.exists():
                    icon_path = str(p)
        
        if icon_path:
            icon_btn.setIcon(QIcon(icon_path))
            icon_btn.setIconSize(QSize(24, 24))
        else:
            # Placeholder or Unknown
            pass
            
        if self.on_profession_change:
            icon_btn.clicked.connect(self.open_profession_selector)
            
        layout.addWidget(icon_btn)

        name = ClickableLabel(self.name)
        name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if self.is_user:
            name.setStyleSheet(f"font-weight: bold; color: {COLORS['text_header']}; font-family: {FONTS['player_name_user'][0]}; font-size: {FONTS['player_name_user'][1]}px;")
        else:
            name.setStyleSheet(f"color: {COLORS['text_primary']}; font-family: {FONTS['player_name'][0]}; font-size: {FONTS['player_name'][1]}px;")
        
        if self.on_name_change:
            name.setToolTip("Double-click to edit name")
            name.setCursor(Qt.PointingHandCursor)
            name.doubleClicked.connect(self.edit_name)

        layout.addWidget(name)

        # Background color
        if self.is_user:
            bg = COLORS['bg_row_user']
        else:
            bg = COLORS['bg_row_red'] if self.team == 'red' else COLORS['bg_row_blue']
        # Slightly transparent darker rows, border subtle
        self.setStyleSheet(
            f"background: {bg}; color: {COLORS['text_primary']}; border-radius:4px; border: 1px solid {COLORS['border']}; padding:4px;"
        )

    def open_profession_selector(self):
        dlg = ProfessionSelectorDialog(self)
        if dlg.exec():
            if dlg.selected_profession and self.on_profession_change:
                self.on_profession_change(self.name, dlg.selected_profession)

    def edit_name(self):
        new_name, ok = QInputDialog.getText(self, "Edit Player Name", "New Name:", text=self.name)
        if ok and new_name and new_name != self.name:
            if self.on_name_change:
                self.on_name_change(self.name, new_name)


class WinRatePanel(QWidget):
    """Panel that mimics the overlay: side-by-side teams with PlayerCardWidget rows."""

    def __init__(self, parent=None, on_profession_change=None, on_name_change=None):
        super().__init__(parent)
        self.on_profession_change = on_profession_change
        self.on_name_change = on_name_change
        self.layout = QVBoxLayout(self)
        self.title = QLabel("Match Analysis")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(f"font-weight: bold; font-size: 16px; color: {COLORS['text_header']}; margin-bottom:10px;")
        self.layout.addWidget(self.title)

        self.teams_layout = QHBoxLayout()
        self.layout.addLayout(self.teams_layout)

        self.red_column = QVBoxLayout()
        self.blue_column = QVBoxLayout()

        self.teams_layout.addLayout(self.red_column)
        # Divider
        divider = QWidget()
        divider.setFixedWidth(2)
        divider.setStyleSheet(f"background:{COLORS['divider']}; margin-left:12px; margin-right:12px;")
        self.teams_layout.addWidget(divider)
        self.teams_layout.addLayout(self.blue_column)

        # Legend (Added once)
        self.legend = QLabel("Stars = Win Rate | (n) = Games Seen")
        self.legend.setAlignment(Qt.AlignCenter)
        self.legend.setStyleSheet(f"color:{COLORS['text_secondary']}; margin-top:10px; font-size:11px;")
        self.layout.addWidget(self.legend)

        self.setStyleSheet(f"background: {COLORS['bg_main']};")

    def show_players(self, players: List[dict]):
        # Clear columns
        def clear_layout(l):
            while l.count():
                item = l.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

        clear_layout(self.red_column)
        clear_layout(self.blue_column)

        red_players = [p for p in players if p.get('team') == 'red']
        blue_players = [p for p in players if p.get('team') == 'blue']

        # Team headers
        def add_team_header(layout, text, color):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{color}; font-weight:700; font-size:13px; margin-bottom:6px;")
            layout.addWidget(lbl)

        add_team_header(self.red_column, "RED TEAM", COLORS['team_red'])
        add_team_header(self.blue_column, "BLUE TEAM", COLORS['team_blue'])

        for p in red_players:
            card = PlayerCardWidget(
                p.get('name', 'Unknown'), 
                'red', 
                p.get('win_rate', 0.0), 
                p.get('total_matches', 0), 
                p.get('is_user', False), 
                profession=p.get('profession'),
                on_profession_change=self.on_profession_change,
                on_name_change=self.on_name_change
            )
            self.red_column.addWidget(card)

        for p in blue_players:
            card = PlayerCardWidget(
                p.get('name', 'Unknown'), 
                'blue', 
                p.get('win_rate', 0.0), 
                p.get('total_matches', 0), 
                p.get('is_user', False), 
                profession=p.get('profession'),
                on_profession_change=self.on_profession_change,
                on_name_change=self.on_name_change
            )
            self.blue_column.addWidget(card)


class RankingsWidget(QWidget):
    """Displays all players in the database with the same card UI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = Database()
        self.layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        controls.addWidget(refresh)
        self.layout.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self.scroll.setWidget(self.inner)
        self.layout.addWidget(self.scroll)

        self.refresh()

    def refresh(self):
        # Clear
        while self.inner_layout.count():
            item = self.inner_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        cursor = self.db.connection.cursor()
        cursor.execute("SELECT char_name, global_wins, global_losses, total_matches, most_played_profession FROM players ORDER BY total_matches DESC")
        rows = cursor.fetchall()
        
        config = Config()

        for row in rows:
            name = row[0]
            wins = row[1] or 0
            losses = row[2] or 0
            total = row[3] or 0
            
            # Apply Bayesian correction if enabled
            win_rate = get_display_win_rate(wins, total, config.data)

            card = PlayerCardWidget(name, 'red', win_rate, total, False)
            self.inner_layout.addWidget(card)

        self.inner_layout.addStretch(1)
