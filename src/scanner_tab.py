from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .catalog import PokemonCatalog
from .database import InventoryDatabase
from .ocr_engine import CardOCREngine


class ScannerTab(QWidget):
    def __init__(
        self,
        catalog: PokemonCatalog,
        database: InventoryDatabase,
        inventory_changed: Callable[[], None],
        settings: QSettings,
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.database = database
        self.inventory_changed = inventory_changed
        self.settings = settings
        self.ocr = CardOCREngine()
        self.camera: cv2.VideoCapture | None = None
        self.current_frame = None
        self.captured_frame = None
        self.matches: list[dict] = []
        self.queue: list[dict] = []

        self._previous_guide_gray = None
        self._stable_frames = 0
        self._capture_pending = False
        self._auto_capture_cooldown = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_camera_frame)
        self._build_ui()
        self.refresh_settings_display()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        heading = QLabel("Scan cards")
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(heading)
        note = QLabel(
            "Place a card fully inside the white guide and hold it still. The app will capture a "
            "temporary still automatically, read it, and show possible matches. The capture button "
            "remains available as a manual fallback."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        root.addWidget(note)

        self.settings_status = QLabel()
        self.settings_status.setStyleSheet("color: #888888;")
        root.addWidget(self.settings_status)

        self.placement_status = QLabel("Start the camera, then place a card in the guide.")
        self.placement_status.setWordWrap(True)
        self.placement_status.setStyleSheet("color: #aaaaaa;")
        root.addWidget(self.placement_status)

        body = QHBoxLayout()
        root.addLayout(body, 2)
        camera_column = QVBoxLayout()
        self.preview = QLabel("Camera is off")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(520, 360)
        self.preview.setStyleSheet("border: 1px solid #555; background: #111;")
        camera_column.addWidget(self.preview)
        buttons = QHBoxLayout()
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.capture_button = QPushButton("Capture & Read Card")
        self.capture_button.clicked.connect(self.capture_card)
        self.capture_button.setEnabled(False)
        self.resume_button = QPushButton("Resume Live View")
        self.resume_button.clicked.connect(self.resume_live_view)
        self.resume_button.setEnabled(False)
        for button in (self.camera_button, self.capture_button, self.resume_button):
            buttons.addWidget(button)
        camera_column.addLayout(buttons)
        body.addLayout(camera_column, 3)

        identify = QVBoxLayout()
        title = QLabel("Identify captured card")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        identify.addWidget(title)
        self.ocr_status = QLabel("OCR has not run yet.")
        self.ocr_status.setWordWrap(True)
        self.ocr_status.setStyleSheet("color: #888888;")
        identify.addWidget(self.ocr_status)
        self.card_name_input = QLineEdit()
        self.card_name_input.setPlaceholderText("Card name, optional")
        self.set_input = QLineEdit()
        self.set_input.setPlaceholderText("Set name or code")
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("Collector number")
        for widget in (self.card_name_input, self.set_input, self.number_input):
            identify.addWidget(widget)
        find_button = QPushButton("Find Matching Card")
        find_button.clicked.connect(self.find_matches)
        identify.addWidget(find_button)

        self.matches_table = QTableWidget(0, 6)
        self.matches_table.setHorizontalHeaderLabels(
            ["Card", "Type", "Set", "Set Code", "Number", "Rarity"]
        )
        self.matches_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.matches_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.matches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.matches_table.horizontalHeader().setStretchLastSection(True)
        identify.addWidget(self.matches_table)

        quantity_row = QHBoxLayout()
        quantity_row.addWidget(QLabel("Quantity for this scan:"))
        self.scan_quantity = QSpinBox()
        self.scan_quantity.setRange(1, 9999)
        self.scan_quantity.setValue(1)
        quantity_row.addWidget(self.scan_quantity)
        quantity_row.addStretch()
        identify.addLayout(quantity_row)
        queue_button = QPushButton("Add Selected Match to Scan Queue")
        queue_button.clicked.connect(self.add_to_queue)
        identify.addWidget(queue_button)
        body.addLayout(identify, 2)

        queue_heading = QLabel("Scan queue")
        queue_heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(queue_heading)
        self.queue_table = QTableWidget(0, 6)
        self.queue_table.setHorizontalHeaderLabels(
            ["Card", "Type", "Set", "Number", "Quantity", "Card ID"]
        )
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.queue_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.queue_table, 1)

        controls = QHBoxLayout()
        remove = QPushButton("Remove Selected Queue Item")
        remove.clicked.connect(self.remove_queue_item)
        clear = QPushButton("Clear Queue")
        clear.clicked.connect(self.clear_queue)
        commit = QPushButton("Commit Queue to Inventory")
        commit.clicked.connect(self.commit_queue)
        controls.addWidget(remove)
        controls.addWidget(clear)
        controls.addStretch()
        controls.addWidget(commit)
        root.addLayout(controls)

    def refresh_settings_display(self) -> None:
        camera_index = int(self.settings.value("scanner/camera_index", 0))
        confidence = int(self.settings.value("scanner/confidence", 90))
        resolution = str(self.settings.value("scanner/resolution", "")) or "camera default"
        self.settings_status.setText(
            f"Camera {camera_index} • Resolution: {resolution} • Recognition threshold: {confidence}%"
        )

    def toggle_camera(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        camera_index = int(self.settings.value("scanner/camera_index", 0))
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            QMessageBox.warning(
                self,
                "Camera unavailable",
                f"Camera {camera_index} could not be opened. Choose another camera in Settings or check camera permissions.",
            )
            return
        resolution = str(self.settings.value("scanner/resolution", ""))
        if "x" in resolution:
            width_text, height_text = resolution.split("x", 1)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width_text))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height_text))
        self.camera = camera
        self._reset_auto_capture()
        self.timer.start(30)
        self.camera_button.setText("Stop Camera")
        self.capture_button.setEnabled(True)
        self.placement_status.setText("Waiting for a card inside the guide…")
        self.refresh_settings_display()

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.current_frame = None
        self.captured_frame = None
        self._reset_auto_capture()
        self.camera_button.setText("Start Camera")
        self.capture_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.preview.setText("Camera is off")
        self.preview.setPixmap(QPixmap())
        self.placement_status.setText("Start the camera, then place a card in the guide.")

    @staticmethod
    def _guide_crop(frame):
        height, width = frame.shape[:2]
        guide_height = int(height * 0.82)
        guide_width = min(width, int(guide_height * 0.714))
        left = max(0, (width - guide_width) // 2)
        top = max(0, (height - guide_height) // 2)
        return frame[top : top + guide_height, left : left + guide_width]

    def _reset_auto_capture(self) -> None:
        self._previous_guide_gray = None
        self._stable_frames = 0
        self._capture_pending = False
        self._auto_capture_cooldown = 18

    def _analyze_placement(self, frame) -> tuple[bool, float]:
        guide = self._guide_crop(frame)
        gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (180, 252), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)

        edges = cv2.Canny(blurred, 45, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        guide_area = float(small.shape[0] * small.shape[1])
        largest_ratio = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < guide_area * 0.20:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if rw <= 0 or rh <= 0:
                continue
            ratio = min(rw, rh) / max(rw, rh)
            if 0.55 <= ratio <= 0.82:
                largest_ratio = max(largest_ratio, area / guide_area)

        detail = cv2.Laplacian(small, cv2.CV_64F).var()
        present = largest_ratio >= 0.34 and detail >= 35

        motion = 999.0
        if self._previous_guide_gray is not None:
            motion = float(cv2.absdiff(blurred, self._previous_guide_gray).mean())
        self._previous_guide_gray = blurred
        return present, motion

    def _read_camera_frame(self) -> None:
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            return

        self.current_frame = frame
        self._display_frame(frame)

        if self._auto_capture_cooldown > 0:
            self._auto_capture_cooldown -= 1
            return
        if self._capture_pending or not self.timer.isActive():
            return

        present, motion = self._analyze_placement(frame)
        if not present:
            self._stable_frames = 0
            self.placement_status.setText("Move the whole card inside the white guide.")
            return

        if motion <= 3.8:
            self._stable_frames += 1
        else:
            self._stable_frames = 0

        required = 12
        progress = min(100, int((self._stable_frames / required) * 100))
        if self._stable_frames < required:
            self.placement_status.setText(f"Card detected. Hold still… {progress}%")
            return

        self._capture_pending = True
        self.placement_status.setText("Card steady — capturing still image…")
        QTimer.singleShot(0, self.capture_card)

    def _display_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        guide_height = int(pixmap.height() * 0.82)
        guide_width = int(guide_height * 0.714)
        left = (pixmap.width() - guide_width) // 2
        top = (pixmap.height() - guide_height) // 2
        painter.drawRoundedRect(left, top, guide_width, guide_height, 12, 12)
        strip_top = top + int(guide_height * 0.84)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
        painter.drawRect(left, strip_top, guide_width, max(1, int(guide_height * 0.12)))
        painter.end()
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
        self.placement_status.setText("Still image captured. Reading card…")
        self.run_ocr()

    def run_ocr(self) -> None:
        if self.captured_frame is None:
            return
        self.ocr_status.setText("Reading card text…")
        try:
            result = self.ocr.read_card(self.captured_frame)
        except RuntimeError as exc:
            self.ocr_status.setText("OCR unavailable. Manual identification still works.")
            QMessageBox.warning(self, "OCR unavailable", str(exc))
            return
        except Exception as exc:
            self.ocr_status.setText("OCR could not read this capture. Try better lighting or manual entry.")
            QMessageBox.warning(self, "OCR failed", str(exc))
            return
        finally:
            self.captured_frame = None

        self.card_name_input.clear()
        self.set_input.clear()
        self.number_input.clear()
        self.matches = []
        self.matches_table.setRowCount(0)

        # Low-confidence title OCR is usually distracting garbage. Prefer collector data first.
        if result.collector_number:
            self.number_input.setText(result.collector_number)
        if result.set_code:
            self.set_input.setText(result.set_code)
        if result.card_name and result.confidence >= 55:
            self.card_name_input.setText(result.card_name)

        threshold = int(self.settings.value("scanner/confidence", 90))
        pieces = [f"OCR confidence: {result.confidence}%"]
        if result.collector_number:
            pieces.append(f"number: {result.collector_number}")
        if result.set_code:
            pieces.append(f"set: {result.set_code}")
        if result.card_name and result.confidence >= 55:
            pieces.append(f"name: {result.card_name}")
        self.ocr_status.setText(" • ".join(pieces))

        if result.collector_number or result.set_code or (result.card_name and result.confidence >= 55):
            self.find_matches()
            if result.confidence >= threshold and len(self.matches) == 1:
                self.matches_table.selectRow(0)
                self.ocr_status.setText(self.ocr_status.text() + " • High-confidence single match selected")
            elif result.confidence < threshold:
                self.ocr_status.setText(self.ocr_status.text() + " • Below threshold; review before adding")
        else:
            self.ocr_status.setText(
                f"OCR confidence: {result.confidence}% • No reliable collector information found. "
                "Try again with the card flatter and closer, or identify it manually."
            )
        self.placement_status.setText("Review the result, then add it or resume for another card.")

    def resume_live_view(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self._reset_auto_capture()
            self.timer.start(30)
            self.capture_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.placement_status.setText("Waiting for a card inside the guide…")

    def find_matches(self) -> None:
        name = self.card_name_input.text().strip()
        set_query = self.set_input.text().strip()
        number = self.number_input.text().strip()
        if not name and not set_query and not number:
            QMessageBox.information(self, "Identify card", "Enter a name, set, or collector number.")
            return
        self.matches = self.catalog.search_cards(name=name, set_query=set_query, number=number, limit=100)
        self.matches_table.setRowCount(len(self.matches))
        for row, card in enumerate(self.matches):
            values = [
                card["name"], card.get("card_category", ""), card["set_name"],
                card.get("set_code", ""), card["number"], card.get("rarity", ""),
            ]
            for column, value in enumerate(values):
                self.matches_table.setItem(row, column, QTableWidgetItem(str(value)))
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
        self.ocr_status.setText("Ready for the next card.")
        self.resume_live_view()

    def _refresh_queue(self) -> None:
        self.queue_table.setRowCount(len(self.queue))
        for row, item in enumerate(self.queue):
            card = item["card"]
            values = [
                card["name"], card.get("card_category", ""), card["set_name"],
                card["number"], item["quantity"], card["id"],
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                if column == 4:
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.queue_table.setItem(row, column, table_item)
        self.queue_table.resizeColumnsToContents()

    def remove_queue_item(self) -> None:
        row = self.queue_table.currentRow()
        if 0 <= row < len(self.queue):
            self.queue.pop(row)
            self._refresh_queue()

    def clear_queue(self) -> None:
        if self.queue and QMessageBox.question(
            self, "Clear queue", "Remove every card from the scan queue?"
        ) == QMessageBox.StandardButton.Yes:
            self.queue.clear()
            self._refresh_queue()

    def commit_queue(self) -> None:
        if not self.queue:
            QMessageBox.information(self, "Scan queue", "The scan queue is empty.")
            return
        total = sum(item["quantity"] for item in self.queue)
        unique = len(self.queue)
        for item in self.queue:
            self.database.add_card(item["card"], item["quantity"])
        self.queue.clear()
        self._refresh_queue()
        self.inventory_changed()
        QMessageBox.information(
            self, "Inventory updated", f"Added {total} cards across {unique} unique printings."
        )

    def shutdown(self) -> None:
        self.stop_camera()
