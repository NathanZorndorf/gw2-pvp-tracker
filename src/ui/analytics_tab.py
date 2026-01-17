import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QSpinBox, QPushButton, QTableWidget, 
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
import pandas as pd

from analysis.analytics_engine import AnalyticsEngine

class AnalyticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = AnalyticsEngine()
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout(self)
        
        # --- Controls Area ---
        controls_layout = QHBoxLayout()
        
        # Label: My Profession
        controls_layout.addWidget(QLabel("My Profession:"))
        
        # ComboBox: User Professions
        self.prof_combo = QComboBox()
        self.prof_combo.addItems(["All"])  # Default option
        controls_layout.addWidget(self.prof_combo)
        
        # Label: Match Type
        controls_layout.addWidget(QLabel("Match Type:"))
        
        # ComboBox: Match Type
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["All", "Ranked", "Unranked"])
        controls_layout.addWidget(self.mode_combo)

        # Label: Top N
        controls_layout.addWidget(QLabel("Show Top N:"))
        
        # SpinBox: N
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50)
        self.n_spin.setValue(5)
        controls_layout.addWidget(self.n_spin)
        
        # Refresh Button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(self.refresh_btn)
        
        # Spacer to push controls to left
        controls_layout.addStretch()
        
        self.layout.addLayout(controls_layout)
        
        # --- Main Visualization Area ---
        tables_layout = QHBoxLayout()
        
        # Left: Best Matchups
        best_container = QWidget()
        best_layout = QVBoxLayout(best_container)
        best_header = QLabel("Best Matchups (Highest Win Rate)")
        best_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;") # Greenish
        best_layout.addWidget(best_header)
        
        self.best_table = self.create_table()
        best_layout.addWidget(self.best_table)
        
        tables_layout.addWidget(best_container)
        
        # Right: Worst Matchups
        worst_container = QWidget()
        worst_layout = QVBoxLayout(worst_container)
        worst_header = QLabel("Worst Matchups (Lowest Win Rate)")
        worst_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #F44336;") # Reddish
        worst_layout.addWidget(worst_header)
        
        self.worst_table = self.create_table()
        worst_layout.addWidget(self.worst_table)
        
        tables_layout.addWidget(worst_container)
        
        self.layout.addLayout(tables_layout)
        
        # Initial Population
        self.populate_professions()
        self.refresh_data()
        
    def create_table(self):
        """Helper to create a configured QTableWidget."""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["", "Profession", "Win Rate", "Games"])
        
        header = table.horizontalHeader()
        
        # Icon Column
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(0, 40)
        
        # Name Column
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        # Stats Columns
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        # Hide Vertical Header (Row Numbers)
        table.verticalHeader().setVisible(False)
        
        # Read Only
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        
        return table

    def populate_professions(self):
        """Populate the combo box with available user professions."""
        profs = self.engine.get_user_professions()
        # Keep "All" as first item
        existing = [self.prof_combo.itemText(i) for i in range(self.prof_combo.count())]
        for p in profs:
            if p not in existing:
                self.prof_combo.addItem(p)

    def refresh_data(self):
        """Fetch data and update tables."""
        prof = self.prof_combo.currentText()
        mode = self.mode_combo.currentText()
        n = self.n_spin.value()
        
        stats = self.engine.get_matchup_stats(prof, match_type=mode)
        
        if stats.empty:
            self.best_table.setRowCount(0)
            self.worst_table.setRowCount(0)
            return

        # --- Best Matchups ---
        # Sort by Win Rate DESC, then Total Games DESC
        best_df = stats.sort_values(by=['win_rate', 'total'], ascending=[False, False]).head(n)
        self.update_table_rows(self.best_table, best_df)
        
        # --- Worst Matchups ---
        # Sort by Win Rate ASC, then Total Games DESC
        worst_df = stats.sort_values(by=['win_rate', 'total'], ascending=[True, False]).head(n)
        self.update_table_rows(self.worst_table, worst_df)

    def update_table_rows(self, table, df):
        """Update rows of a specific table."""
        table.setRowCount(len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            prof_name = row['enemy_profession']
            win_rate = row['win_rate']
            total = row['total']
            
            # 1. Icon
            icon_item = QTableWidgetItem()
            # Path logic: data/reference-icons/icons-raw/{Name}.png
            # Assuming CWD is project root
            icon_path = f"data/reference-icons/icons-raw/{prof_name}.png"
            
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    icon_item.setData(Qt.DecorationRole, scaled)
            
            table.setItem(i, 0, icon_item)
            
            # 2. Profession Name
            table.setItem(i, 1, QTableWidgetItem(prof_name))
            
            # 3. Win Rate
            wr_str = f"{win_rate:.1f}%"
            table.setItem(i, 2, QTableWidgetItem(wr_str))
            
            # 4. Games
            table.setItem(i, 3, QTableWidgetItem(str(total)))

