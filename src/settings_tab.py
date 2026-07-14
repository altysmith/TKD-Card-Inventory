from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsTab(QWidget):
    """Persistent scanner and export settings."""

    settings_changed = Signal()

    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.settings = settings
        self._build_ui()
        self.load_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        heading = QLabel("Settings")
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(heading)

        note = QLabel(
            "Camera source, resolution, orientation, scanner mode, confidence, and export location "
            "are stored separately on each computer. Saving camera changes stops the active camera."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        root.addWidget(note)

        form = QFormLayout()

        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(260)
        self.detect_button = QPushButton("Detect Cameras")
        self.detect_button.clicked.connect(self.detect_cameras)
        camera_row.addWidget(self.camera_combo)
        camera_row.addWidget(self.detect_button)
        camera_row.addStretch()
        form.addRow("Camera:", camera_row)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("Default", "")
        self.resolution_combo.addItem("1920 × 1080", "1920x1080")
        self.resolution_combo.addItem("1280 × 720", "1280x720")
        self.resolution_combo.addItem("640 × 480", "640x480")
        form.addRow("Camera resolution:", self.resolution_combo)

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("Automatic", "auto")
        self.orientation_combo.addItem("Landscape", "landscape")
        self.orientation_combo.addItem("Portrait — rotate clockwise", "cw")
        self.orientation_combo.addItem("Portrait — rotate counterclockwise", "ccw")
        self.orientation_combo.addItem("Upside down", "180")
        form.addRow("Camera orientation:", self.orientation_combo)

        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItem("Full card", "full")
        self.scan_mode_combo.addItem("Identifier close-up", "identifier")
        self.scan_mode_combo.setToolTip(
            "Identifier close-up enlarges the lower card area so set codes and collector numbers use more pixels."
        )
        form.addRow("Scanner mode:", self.scan_mode_combo)

        export_row = QHBoxLayout()
        self.export_input = QLineEdit()
        self.export_input.setPlaceholderText("Ask each time")
        browse = QPushButton("Choose Folder")
        browse.clicked.connect(self.choose_export_folder)
        clear = QPushButton("Ask Each Time")
        clear.clicked.connect(self.export_input.clear)
        export_row.addWidget(self.export_input)
        export_row.addWidget(browse)
        export_row.addWidget(clear)
        form.addRow("Default export folder:", export_row)

        self.confidence_input = QSpinBox()
        self.confidence_input.setRange(50, 100)
        self.confidence_input.setSuffix("%")
        self.confidence_input.setValue(90)
        self.confidence_input.setToolTip(
            "Automatic scans require at least this recognition confidence before being accepted."
        )
        form.addRow("Auto-scan confidence:", self.confidence_input)

        self.beep_checkbox = QCheckBox("Play a confirmation sound after a successful scan")
        self.beep_checkbox.setChecked(True)
        form.addRow("Scanner sound:", self.beep_checkbox)

        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        save = QPushButton("Save Settings")
        save.clicked.connect(self.save_settings)
        buttons.addWidget(save)
        root.addLayout(buttons)
        root.addStretch()

    @staticmethod
    def _open_camera(index: int) -> cv2.VideoCapture:
        if sys.platform == "darwin":
            return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if sys.platform.startswith("win"):
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(index)

    def detect_cameras(self) -> None:
        current = int(self.settings.value("scanner/camera_index", 0))
        found: list[int] = []
        self.detect_button.setEnabled(False)
        max_indexes = 2 if sys.platform == "darwin" else 6
        try:
            for index in range(max_indexes):
                camera = self._open_camera(index)
                try:
                    if camera.isOpened():
                        ok, _ = camera.read()
                        if ok:
                            found.append(index)
                finally:
                    camera.release()
        finally:
            self.detect_button.setEnabled(True)

        self.camera_combo.clear()
        if not found:
            self.camera_combo.addItem("Camera 0 (not currently available)", 0)
            QMessageBox.warning(
                self,
                "No cameras detected",
                "No available cameras were found. Check permissions and make sure another app is not using the camera.",
            )
            return

        for index in found:
            self.camera_combo.addItem(f"Camera {index}", index)
        selected = self.camera_combo.findData(current)
        self.camera_combo.setCurrentIndex(selected if selected >= 0 else 0)

    def choose_export_folder(self) -> None:
        start = self.export_input.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose default export folder", start)
        if folder:
            self.export_input.setText(folder)

    def load_settings(self) -> None:
        camera_index = int(self.settings.value("scanner/camera_index", 0))
        self.camera_combo.clear()
        self.camera_combo.addItem(f"Camera {camera_index}", camera_index)

        resolution = str(self.settings.value("scanner/resolution", ""))
        index = self.resolution_combo.findData(resolution)
        self.resolution_combo.setCurrentIndex(index if index >= 0 else 0)

        orientation = str(self.settings.value("scanner/orientation", "auto"))
        index = self.orientation_combo.findData(orientation)
        self.orientation_combo.setCurrentIndex(index if index >= 0 else 0)

        scan_mode = str(self.settings.value("scanner/mode", "full"))
        index = self.scan_mode_combo.findData(scan_mode)
        self.scan_mode_combo.setCurrentIndex(index if index >= 0 else 0)

        self.export_input.setText(str(self.settings.value("exports/default_folder", "")))
        self.confidence_input.setValue(int(self.settings.value("scanner/confidence", 90)))
        self.beep_checkbox.setChecked(
            str(self.settings.value("scanner/beep", "true")).casefold() in {"1", "true", "yes"}
        )

    def save_settings(self) -> None:
        self.settings.setValue("scanner/camera_index", int(self.camera_combo.currentData() or 0))
        self.settings.setValue("scanner/resolution", self.resolution_combo.currentData() or "")
        self.settings.setValue("scanner/orientation", self.orientation_combo.currentData() or "auto")
        self.settings.setValue("scanner/mode", self.scan_mode_combo.currentData() or "full")
        self.settings.setValue("exports/default_folder", self.export_input.text().strip())
        self.settings.setValue("scanner/confidence", self.confidence_input.value())
        self.settings.setValue("scanner/beep", self.beep_checkbox.isChecked())
        self.settings.sync()
        self.settings_changed.emit()
        QMessageBox.information(
            self,
            "Settings saved",
            "Settings were saved. Any active camera was stopped so the new camera configuration can be applied safely.",
        )
