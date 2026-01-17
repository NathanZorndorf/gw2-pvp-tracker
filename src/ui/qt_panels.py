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
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QSize
import os
from pathlib import Path

from .styles import COLORS, get_star_count, SYMBOLS, get_winrate_color, FONTS
from .confidence import format_winrate
from database.models import Database


class ProfessionSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Profession")
        self.setStyleSheet(f"background: {COLORS['bg_main']}; color: {COLORS['text_primary']};")
        self.layout = QGridLayout(self)
        self.selected_profession = None
        
        icons_path = Path("data/reference-icons/icons-raw")
        row, col = 0, 0
        max_cols = 6
        
        if icons_path.exists():
            for icon_file in sorted(icons_path.glob("*.png")):
                prof_name = icon_file.stem
                btn = QPushButton()
                btn.setIcon(QIcon(str(icon_file)))
                btn.setIconSize(QSize(32, 32))
                btn.setToolTip(prof_name)
                btn.setFlat(True)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda checked=False, p=prof_name: self.select_profession(p))
                
                self.layout.addWidget(btn, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

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
    def __init__(self, name: str, team: str, win_rate: float, total_matches: int, is_user: bool = False, profession: str = None, on_profession_change=None, parent=None):
        super().__init__(parent)
        self.name = name
        self.team = team
        self.win_rate = win_rate
        self.total_matches = total_matches
        self.is_user = is_user
        self.profession = profession
        self.on_profession_change = on_profession_change

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

        name = QLabel(self.name)
        name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if self.is_user:
            name.setStyleSheet(f"font-weight: bold; color: {COLORS['text_header']}; font-family: {FONTS['player_name_user'][0]}; font-size: {FONTS['player_name_user'][1]}px;")
        else:
            name.setStyleSheet(f"color: {COLORS['text_primary']}; font-family: {FONTS['player_name'][0]}; font-size: {FONTS['player_name'][1]}px;")
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


class WinRatePanel(QWidget):
    """Panel that mimics the overlay: side-by-side teams with PlayerCardWidget rows."""

    def __init__(self, parent=None, on_profession_change=None):
        super().__init__(parent)
        self.on_profession_change = on_profession_change
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
                on_profession_change=self.on_profession_change
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
                on_profession_change=self.on_profession_change
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

        for row in rows:
            name = row[0]
            wins = row[1] or 0
            losses = row[2] or 0
            total = row[3] or 0
            win_rate = (wins / total * 100) if total > 0 else 0.0

            card = PlayerCardWidget(name, 'red', win_rate, total, False)
            self.inner_layout.addWidget(card)

        self.inner_layout.addStretch(1)
