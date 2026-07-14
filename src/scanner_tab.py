from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .catalog import PokemonCatalog
from .database import InventoryDatabase


class ScannerTab(QWidget):
    """First scanner milestone: webcam capture, local matching, and a review queue.

    OCR is intentionally the next layer. This version proves camera access and gives the
    user a fast set/name/number fallback against the local catalog.
    """

    def __init__(
        self,
        catalog: PokemonCatalog,
        database: InventoryDatabase,
        inventory_changed: Callable[[], None],
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.database = database
        self.inventory_changed = inventory_changed
        self.camera: cv2.VideoCapture | None = None
        self.current_frame = None
        self.captured_frame = None
        self.matches: list[dict] = []
        self.queue: list[dict] = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_camera_frame)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        heading = QLabel("Scan cards")
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(heading)

        note = QLabel(
            "Scanner MVP: capture a card, identify it from the local catalog, choose quantity, "
            "then add it to the scan queue. Automatic OCR comes next."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        root.addWidget(note)

        body = QHBoxLayout()
        root.addLayout(body, 2)

        camera_column = QVBoxLayout()
        self.preview = QLabel("Camera is off")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(520, 360)
        self.preview.setStyleSheet("border: 1px solid #555; background: #111;")
        camera_column.addWidget(self.preview)

        camera_buttons = QHBoxLayout()
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.capture_button = QPushButton("Capture Card")
        self.capture_button.clicked.connect(self.capture_card)
        self.capture_button.setEnabled(False)
        self.resume_button = QPushButton("Resume Live View")
        self.resume_button.clicked.connect(self.resume_live_view)
        self.resume_button.setEnabled(False)
        camera_buttons.addWidget(self.camera_button)
        camera_buttons.addWidget(self.capture_button)
        camera_buttons.addWidget(self.resume_button)
        camera_column.addLayout(camera_buttons)
        body.addLayout(camera_column, 3)

        identify_column = QVBoxLayout()
        identify_heading = QLabel("Identify captured card")
        identify_heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        identify_column.addWidget(identify_heading)

        self.card_name_input = QLineEdit()
        self.card_name_input.setPlaceholderText("Card name, optional")
        self.set_input = QLineEdit()
        self.set_input.setPlaceholderText("Set name or set code, e.g. Temporal Forces or TEF")
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("Collector number, e.g. 114")
        identify_column.addWidget(self.card_name_input)
        identify_column.addWidget(self.set_input)
        identify_column.addWidget(self.number_input)

        find_button = QPushButton("Find Matching Card")
        find_button.clicked.connect(self.find_matches)
        identify_column.addWidget(find_button)

        self.matches_table = QTableWidget(0, 5)
        self.matches_table.setHorizontalHeaderLabels(
            ["Card", "Set", "Set Code", "Number", "Rarity"]
        )
        self.matches_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.matches_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.matches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.matches_table.horizontalHeader().setStretchLastSection(True)
        identify_column.addWidget(self.matches_table)

        quantity_row = QHBoxLayout()
        quantity_row.addWidget(QLabel("Quantity for this scan:"))
        self.scan_quantity = QSpinBox()
        self.scan_quantity.setRange(1, 9999)
        self.scan_quantity.setValue(1)
        quantity_row.addWidget(self.scan_quantity)
        quantity_row.addStretch()
        identify_column.addLayout(quantity_row)

        queue_button = QPushButton("Add Selected Match to Scan Queue")
        queue_button.clicked.connect(self.add_to_queue)
        identify_column.addWidget(queue_button)
        body.addLayout(identify_column, 2)

        queue_heading = QLabel("Scan queue")
        queue_heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(queue_heading)

        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(
            ["Card", "Set", "Number", "Quantity", "Card ID"]
        )
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.queue_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.queue_table, 1)

        queue_controls = QHBoxLayout()
        remove_button = QPushButton("Remove Selected Queue Item")
        remove_button.clicked.connect(self.remove_queue_item)
        clear_button = QPushButton("Clear Queue")
        clear_button.clicked.connect(self.clear_queue)
        commit_button = QPushButton("Commit Queue to Inventory")
        commit_button.clicked.connect(self.commit_queue)
        queue_controls.addWidget(remove_button)
        queue_controls.addWidget(clear_button)
        queue_controls.addStretch()
        queue_controls.addWidget(commit_button)
        root.addLayout(queue_controls)

    def toggle_camera(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera.release()
            QMessageBox.warning(
                self,
                "Camera unavailable",
                "The default webcam could not be opened. Check camera permissions and make "
                "sure another application is not using it.",
            )
            return
        self.camera = camera
        self.timer.start(30)
        self.camera_button.setText("Stop Camera")
        self.capture_button.setEnabled(True)
        self.resume_button.setEnabled(False)

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.current_frame = None
        self.camera_button.setText("Start Camera")
        self.capture_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.preview.setText("Camera is off")
        self.preview.setPixmap(QPixmap())

    def _read_camera_frame(self) -> None:
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            return
        self.current_frame = frame
        self._display_frame(frame)

    def _display_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def capture_card(self) -> None:
        if self.current_frame is None:
            QMessageBox.information(self, "Capture", "Start the camera and place a card in view first.")
            return
        self.captured_frame = self.current_frame.copy()
        self.timer.stop()
        self._display_frame(self.captured_frame)
        self.capture_button.setEnabled(False)
        self.resume_button.setEnabled(True)

    def resume_live_view(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self.timer.start(30)
            self.capture_button.setEnabled(True)
            self.resume_button.setEnabled(False)

    def find_matches(self) -> None:
        name = self.card_name_input.text().strip()
        set_query = self.set_input.text().strip()
        number = self.number_input.text().strip()
        if not name and not set_query and not number:
            QMessageBox.information(
                self,
                "Identify card",
                "Enter at least a card name, set name/code, or collector number.",
            )
            return
        if not self.catalog.is_ready():
            QMessageBox.warning(self, "Catalog required", "The local card catalog is not ready.")
            return

        self.matches = self.catalog.search_cards(
            name=name,
            number=number,
            set_query=set_query,
            limit=100,
        )
        self.matches_table.setRowCount(len(self.matches))
        for row_index, card in enumerate(self.matches):
            values = [
                card["name"],
                card["set_name"],
                card.get("set_code", ""),
                card["number"],
                card.get("rarity", ""),
            ]
            for column, value in enumerate(values):
                self.matches_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.matches_table.resizeColumnsToContents()
        if self.matches:
            self.matches_table.selectRow(0)

    def add_to_queue(self) -> None:
        row = self.matches_table.currentRow()
        if row < 0 or row >= len(self.matches):
            QMessageBox.information(self, "Scan queue", "Select a matching card first.")
            return

        card = self.matches[row]
        quantity = self.scan_quantity.value()
        existing = next((item for item in self.queue if item["card"]["id"] == card["id"]), None)
        if existing:
            existing["quantity"] += quantity
        else:
            self.queue.append({"card": card, "quantity": quantity})
        self._refresh_queue()

        self.card_name_input.clear()
        self.set_input.clear()
        self.number_input.clear()
        self.matches = []
        self.matches_table.setRowCount(0)
        self.scan_quantity.setValue(1)
        self.resume_live_view()

    def _refresh_queue(self) -> None:
        self.queue_table.setRowCount(len(self.queue))
        for row_index, item in enumerate(self.queue):
            card = item["card"]
            values = [
                card["name"],
                card["set_name"],
                card["number"],
                item["quantity"],
                card["id"],
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                if column == 3:
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.queue_table.setItem(row_index, column, table_item)
        self.queue_table.resizeColumnsToContents()

    def remove_queue_item(self) -> None:
        row = self.queue_table.currentRow()
        if 0 <= row < len(self.queue):
            self.queue.pop(row)
            self._refresh_queue()

    def clear_queue(self) -> None:
        if not self.queue:
            return
        answer = QMessageBox.question(self, "Clear queue", "Remove every card from the scan queue?")
        if answer == QMessageBox.StandardButton.Yes:
            self.queue.clear()
            self._refresh_queue()

    def commit_queue(self) -> None:
        if not self.queue:
            QMessageBox.information(self, "Scan queue", "The scan queue is empty.")
            return
        total_cards = sum(item["quantity"] for item in self.queue)
        unique_cards = len(self.queue)
        for item in self.queue:
            self.database.add_card(item["card"], item["quantity"])
        self.queue.clear()
        self._refresh_queue()
        self.inventory_changed()
        QMessageBox.information(
            self,
            "Inventory updated",
            f"Added {total_cards} cards across {unique_cards} unique printings to inventory.",
        )

    def shutdown(self) -> None:
        self.stop_camera()
