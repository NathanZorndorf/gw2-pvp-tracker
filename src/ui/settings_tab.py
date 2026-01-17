from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QDoubleSpinBox, QGroupBox, QPushButton,
    QFormLayout, QMessageBox
)
from PySide6.QtCore import Qt, Signal
import logging

from config import Config

logger = logging.getLogger(__name__)

class SettingsWidget(QWidget):
    # Signal to notify other tabs/windows when settings change
    settingsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = Config()
        self._init_ui()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        # --- Analytics Settings Group ---
        analytics_group = QGroupBox("Analytics Settings")
        analytics_layout = QFormLayout(analytics_group)

        # Win Rate Mode
        self.win_rate_mode = QComboBox()
        self.win_rate_mode.addItems(["Raw (wins / matches)", "Bayesian Averaged"])
        analytics_mode = self.config.get('analytics.win_rate_mode', 'raw')
        self.win_rate_mode.setCurrentIndex(1 if analytics_mode == 'bayesian' else 0)
        self.win_rate_mode.currentIndexChanged.connect(self._toggle_bayesian_params)
        analytics_layout.addRow("Win Rate Metric:", self.win_rate_mode)

        # Bayesian Confidence (C)
        self.bayesian_confidence = QDoubleSpinBox()
        self.bayesian_confidence.setRange(0.1, 100.0)
        self.bayesian_confidence.setSingleStep(1.0)
        self.bayesian_confidence.setValue(self.config.get('analytics.bayesian_confidence', 10.0))
        self.bayesian_confidence.setToolTip("Number of matches required to 'trust' a trend (C)")
        analytics_layout.addRow("Bayesian Confidence (C):", self.bayesian_confidence)

        # Bayesian Mean (M)
        self.bayesian_mean = QDoubleSpinBox()
        self.bayesian_mean.setRange(0.0, 100.0)
        self.bayesian_mean.setSingleStep(5.0)
        self.bayesian_mean.setValue(self.config.get('analytics.bayesian_mean', 50.0))
        self.bayesian_mean.setSuffix("%")
        self.bayesian_mean.setToolTip("The global mean win rate to pull towards (M)")
        analytics_layout.addRow("Global Mean (M):", self.bayesian_mean)

        self.layout.addWidget(analytics_group)

        # --- Save Button ---
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setFixedHeight(40)
        self.save_btn.setStyleSheet("font-weight: bold; background-color: #2c3e50;")
        self.save_btn.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_btn)

        # Bottom stretch
        self.layout.addStretch()

        # Initial parameter visibility
        self._toggle_bayesian_params()

    def _toggle_bayesian_params(self):
        is_bayesian = self.win_rate_mode.currentIndex() == 1
        self.bayesian_confidence.setEnabled(is_bayesian)
        self.bayesian_mean.setEnabled(is_bayesian)

    def save_settings(self):
        """Save settings to config and notify others."""
        try:
            # Use strict setter to ensure clean persistence
            mode = 'bayesian' if self.win_rate_mode.currentIndex() == 1 else 'raw'
            self.config.set_setting('analytics', 'win_rate_mode', mode)
            self.config.set_setting('analytics', 'bayesian_confidence', self.bayesian_confidence.value())
            self.config.set_setting('analytics', 'bayesian_mean', self.bayesian_mean.value())

            # Persist to disk
            self.config.save()
            
            # Emit signal
            self.settingsChanged.emit()
            
            QMessageBox.information(self, "Settings Saved", "Analytics settings have been updated and saved.")
            logger.info("Settings saved and notification emitted.")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
