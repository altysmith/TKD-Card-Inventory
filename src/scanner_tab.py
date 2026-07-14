from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .catalog import PokemonCatalog
from .database import InventoryDatabase
from .ocr_engine import CardOCREngine, OCRResult


class ScannerTab(QWidget):
    """Camera scanner with temporary OCR debug previews."""

    CARD_ASPECT_RATIO = 0.714

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
        self._active_signature: tuple[int, str, str, str] | None = None

        self._baseline_gray = None
        self._previous_gray = None
        self._baseline_frames = 0
        self._stable_frames = 0
        self._capture_pending = False
        self._armed = False

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
            "The fast OCR pass now uses two targeted reads. Enable OCR Debug Mode in Settings to see "
            "the exact crop, processed image, raw text, sharpness, and processing time."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        root.addWidget(note)

        self.settings_status = QLabel()
        self.settings_status.setStyleSheet("color: #888888;")
        root.addWidget(self.settings_status)
        self.placement_status = QLabel("Start the camera, then calibrate with the guide empty.")
        self.placement_status.setWordWrap(True)
        self.placement_status.setStyleSheet("color: #aaaaaa;")
        root.addWidget(self.placement_status)

        body = QHBoxLayout()
        root.addLayout(body, 2)

        camera_column = QVBoxLayout()
        self.preview = QLabel("Camera is off")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(680, 460)
        self.preview.setStyleSheet("border: 1px solid #555; background: #111;")
        camera_column.addWidget(self.preview)

        buttons = QHBoxLayout()
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.capture_button = QPushButton("Capture & Read Card")
        self.capture_button.clicked.connect(self.capture_card)
        self.capture_button.setEnabled(False)
        self.reset_button = QPushButton("Reset Auto Capture")
        self.reset_button.clicked.connect(self.reset_auto_capture)
        self.reset_button.setEnabled(False)
        self.resume_button = QPushButton("Resume Live View")
        self.resume_button.clicked.connect(self.resume_live_view)
        self.resume_button.setEnabled(False)
        for button in (self.camera_button, self.capture_button, self.reset_button, self.resume_button):
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
        self.set_input.setPlaceholderText("Set acronym or set name")
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

        self.debug_group = QGroupBox("OCR Debug — what the app actually read")
        debug_layout = QVBoxLayout(self.debug_group)
        debug_images = QHBoxLayout()
        self.debug_crop = QLabel("OCR crop")
        self.debug_processed = QLabel("Processed OCR image")
        for label in (self.debug_crop, self.debug_processed):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(300, 150)
            label.setStyleSheet("border: 1px solid #555; background: #111;")
            debug_images.addWidget(label)
        debug_layout.addLayout(debug_images)
        self.debug_metrics = QLabel("No debug data yet.")
        self.debug_metrics.setWordWrap(True)
        debug_layout.addWidget(self.debug_metrics)
        self.debug_text = QPlainTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setPlaceholderText("Raw OCR output will appear here.")
        self.debug_text.setMaximumHeight(100)
        debug_layout.addWidget(self.debug_text)
        self.enhanced_button = QPushButton("Retry Enhanced OCR on This Capture")
        self.enhanced_button.clicked.connect(self.retry_enhanced_ocr)
        self.enhanced_button.setEnabled(False)
        debug_layout.addWidget(self.enhanced_button, alignment=Qt.AlignmentFlag.AlignRight)
        root.addWidget(self.debug_group)

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

    def _settings_signature(self) -> tuple[int, str, str, str]:
        return (
            int(self.settings.value("scanner/camera_index", 0)),
            str(self.settings.value("scanner/resolution", "")),
            str(self.settings.value("scanner/orientation", "landscape")),
            str(self.settings.value("scanner/mode", "full")),
        )

    def refresh_settings_display(self) -> None:
        signature = self._settings_signature()
        if self._active_signature is not None and signature != self._active_signature:
            self.stop_camera()
            self.placement_status.setText("Camera settings changed. Click Start Camera to apply them.")
        camera_index, requested_resolution, orientation, mode = signature
        confidence = int(self.settings.value("scanner/confidence", 90))
        resolution = requested_resolution or "camera default"
        actual = ""
        if self.camera is not None and self.camera.isOpened():
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual = f" • Actual: {actual_width}×{actual_height}"
        self.settings_status.setText(
            f"Camera {camera_index} • Requested: {resolution}{actual} • Orientation: {orientation} "
            f"• Mode: {mode} • Threshold: {confidence}%"
        )
        debug_enabled = str(self.settings.value("scanner/debug", "true")).casefold() in {"1", "true", "yes"}
        self.debug_group.setVisible(debug_enabled)

    def toggle_camera(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self.stop_camera()
        else:
            self.start_camera()

    @staticmethod
    def _rotate_frame(frame, orientation: str):
        if orientation == "cw":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if orientation == "ccw":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation == "180":
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def start_camera(self) -> None:
        camera_index, resolution, _, _ = self._settings_signature()
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            QMessageBox.warning(
                self, "Camera unavailable",
                f"Camera {camera_index} could not be opened. Choose another camera in Settings or check permissions.",
            )
            return
        if "x" in resolution:
            width_text, height_text = resolution.split("x", 1)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width_text))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height_text))
        self.camera = camera
        self._active_signature = self._settings_signature()
        self.timer.start(30)
        self.camera_button.setText("Stop Camera")
        self.capture_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.reset_auto_capture()
        self.refresh_settings_display()

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.current_frame = None
        self.captured_frame = None
        self._active_signature = None
        self.camera_button.setText("Start Camera")
        self.capture_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.enhanced_button.setEnabled(False)
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Camera is off")

    def _guide_rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        mode = str(self.settings.value("scanner/mode", "full"))
        if mode == "identifier":
            guide_width = int(width * 0.88)
            guide_height = int(height * 0.38)
        else:
            guide_height = int(height * 0.94)
            guide_width = int(guide_height * self.CARD_ASPECT_RATIO)
            if guide_width > int(width * 0.92):
                guide_width = int(width * 0.92)
                guide_height = int(guide_width / self.CARD_ASPECT_RATIO)
        return max(0, (width - guide_width) // 2), max(0, (height - guide_height) // 2), guide_width, guide_height

    def _guide_crop(self, frame):
        height, width = frame.shape[:2]
        left, top, guide_width, guide_height = self._guide_rect(width, height)
        return frame[top : top + guide_height, left : left + guide_width]

    def _prepare_detection_frame(self, frame):
        guide = self._guide_crop(frame)
        gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (220, 130), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0)

    def reset_auto_capture(self) -> None:
        self._baseline_gray = None
        self._previous_gray = None
        self._baseline_frames = 0
        self._stable_frames = 0
        self._capture_pending = False
        self._armed = False
        self.captured_frame = None
        self.enhanced_button.setEnabled(False)
        if self.camera is not None and self.camera.isOpened():
            if not self.timer.isActive():
                self.timer.start(30)
            self.capture_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.placement_status.setText("Keep the guide empty for about one second while auto capture resets.")

    def _read_camera_frame(self) -> None:
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            return
        orientation = str(self.settings.value("scanner/orientation", "landscape"))
        frame = self._rotate_frame(frame, orientation)
        self.current_frame = frame
        self._display_frame(frame)
        if self._capture_pending or not self.timer.isActive():
            return

        current = self._prepare_detection_frame(frame)
        if self._baseline_gray is None:
            if self._previous_gray is None:
                self._previous_gray = current
                return
            motion = float(cv2.absdiff(current, self._previous_gray).mean())
            self._previous_gray = current
            self._baseline_frames = self._baseline_frames + 1 if motion <= 3.2 else 0
            self.placement_status.setText(f"Calibrating empty guide… {min(100, self._baseline_frames * 12)}%")
            if self._baseline_frames >= 8:
                self._baseline_gray = current.copy()
                self._armed = True
                self.placement_status.setText("Ready — place the card identifier area inside the guide.")
            return

        change = cv2.absdiff(current, self._baseline_gray)
        changed_ratio = float((change > 16).mean())
        mean_change = float(change.mean())
        motion = float(cv2.absdiff(current, self._previous_gray).mean()) if self._previous_gray is not None else 999.0
        self._previous_gray = current
        scene_changed = changed_ratio >= 0.14 or mean_change >= 6.5
        if not scene_changed:
            self._stable_frames = 0
            self._armed = True
            self.placement_status.setText("Ready — place the next card inside the guide.")
            return
        if not self._armed:
            self.placement_status.setText("Remove the current card, or click Reset Auto Capture.")
            return

        self._stable_frames = self._stable_frames + 1 if motion <= 4.2 else 0
        progress = min(100, int((self._stable_frames / 8) * 100))
        self.placement_status.setText(f"Card detected. Hold still… {progress}%")
        if self._stable_frames >= 8:
            self._capture_pending = True
            self._armed = False
            self.placement_status.setText("Card steady — capturing a temporary still…")
            QTimer.singleShot(250, self.capture_card)

    def _display_frame(self, frame) -> None:
        pixmap = self._frame_to_pixmap(frame, self.preview.size())
        painter = QPainter(pixmap)
        left, top, guide_width, guide_height = self._guide_rect(pixmap.width(), pixmap.height())
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawRoundedRect(left, top, guide_width, guide_height, 12, 12)
        painter.setPen(QPen(Qt.GlobalColor.yellow, 3))
        if str(self.settings.value("scanner/mode", "full")) == "identifier":
            painter.drawRect(left, top, guide_width, guide_height)
        else:
            strip_top = top + int(guide_height * 0.64)
            painter.drawRect(left, strip_top, guide_width, max(1, int(guide_height * 0.35)))
        painter.end()
        self.preview.setPixmap(pixmap)

    @staticmethod
    def _frame_to_pixmap(frame, size) -> QPixmap:
        if len(frame.shape) == 2:
            image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1], QImage.Format.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[2] * rgb.shape[1], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy()).scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    @staticmethod
    def _sharpness(frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def capture_card(self) -> None:
        if self.current_frame is None or self.camera is None:
            QMessageBox.information(self, "Capture", "Start the camera and place a card in view first.")
            return
        orientation = str(self.settings.value("scanner/orientation", "landscape"))
        candidates = [self.current_frame.copy()]
        for _ in range(2):
            ok, frame = self.camera.read()
            if ok:
                candidates.append(self._rotate_frame(frame, orientation))
        self.captured_frame = max(candidates, key=self._sharpness)
        candidates.clear()
        self.timer.stop()
        self._display_frame(self.captured_frame)
        self.capture_button.setEnabled(False)
        self.resume_button.setEnabled(True)
        self.enhanced_button.setEnabled(True)
        self.placement_status.setText("Temporary still captured. Running fast OCR…")
        self.run_ocr(enhanced=False)

    def retry_enhanced_ocr(self) -> None:
        if self.captured_frame is None:
            QMessageBox.information(self, "Enhanced OCR", "There is no current capture to retry.")
            return
        self.run_ocr(enhanced=True)

    def run_ocr(self, enhanced: bool = False) -> None:
        if self.captured_frame is None:
            return
        self.ocr_status.setText("Running enhanced OCR…" if enhanced else "Running fast OCR…")
        mode = str(self.settings.value("scanner/mode", "full"))
        try:
            result = self.ocr.read_card(self.captured_frame, mode=mode, enhanced=enhanced)
        except RuntimeError as exc:
            self.ocr_status.setText("OCR unavailable. Manual identification still works.")
            QMessageBox.warning(self, "OCR unavailable", str(exc))
            self._resume_after_failed_read()
            return
        except Exception as exc:
            self.ocr_status.setText("OCR could not read this capture.")
            QMessageBox.warning(self, "OCR failed", str(exc))
            self._resume_after_failed_read()
            return

        self._show_debug_result(result, enhanced)
        self.card_name_input.clear()
        self.set_input.clear()
        self.number_input.clear()
        self.matches = []
        self.matches_table.setRowCount(0)
        if result.set_code:
            self.set_input.setText(result.set_code)
        if result.collector_number:
            number = result.collector_number
            if result.printed_total:
                number += f"/{result.printed_total}"
            self.number_input.setText(number)
        if result.card_name and result.confidence >= 60:
            self.card_name_input.setText(result.card_name)

        details = [f"OCR confidence: {result.confidence}%", f"time: {result.processing_ms} ms"]
        if result.set_code:
            details.append(f"set: {result.set_code}")
        if result.collector_number:
            number = result.collector_number + (f"/{result.printed_total}" if result.printed_total else "")
            details.append(f"number: {number}")
        self.ocr_status.setText(" • ".join(details))

        if result.set_code or result.collector_number:
            self.find_matches(printed_total=result.printed_total)
        threshold = int(self.settings.value("scanner/confidence", 90))
        if result.confidence >= threshold and len(self.matches) == 1:
            self.matches_table.selectRow(0)
            self.placement_status.setText("Match ready. Choose quantity and add it to the queue.")
            return
        self.ocr_status.setText(self.ocr_status.text() + " • Review debug output or retry enhanced OCR")
        self.placement_status.setText("Review the debug panel. Remove the card and resume when finished.")

    def _show_debug_result(self, result: OCRResult, enhanced: bool) -> None:
        if result.crop_image is not None:
            self.debug_crop.setPixmap(self._frame_to_pixmap(result.crop_image, self.debug_crop.size()))
        if result.processed_image is not None:
            self.debug_processed.setPixmap(self._frame_to_pixmap(result.processed_image, self.debug_processed.size()))
        self.debug_metrics.setText(
            f"Pass: {'Enhanced' if enhanced else 'Fast'} • Processing: {result.processing_ms} ms • "
            f"Crop sharpness: {result.sharpness:.1f} • Confidence: {result.confidence}%"
        )
        self.debug_text.setPlainText(result.raw_text or "No text returned by Tesseract.")

    def _resume_after_failed_read(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self.captured_frame = None
            self.enhanced_button.setEnabled(False)
            self.timer.start(30)
            self.capture_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self._capture_pending = False
            self._stable_frames = 0
            self._armed = False
            self.placement_status.setText("Remove this card to rearm. Click Reset Auto Capture if needed.")

    def resume_live_view(self) -> None:
        self._resume_after_failed_read()

    def find_matches(self, printed_total: int | None = None) -> None:
        name = self.card_name_input.text().strip()
        set_query = self.set_input.text().strip()
        number_text = self.number_input.text().strip()
        number = number_text.split("/")[0] if number_text else ""
        if printed_total is None and "/" in number_text:
            total_text = number_text.split("/", 1)[1].strip()
            if total_text.isdigit():
                printed_total = int(total_text)
        if not name and not set_query and not number:
            QMessageBox.information(self, "Identify card", "Enter a set acronym, number, or card name.")
            return
        self.matches = self.catalog.search_cards(
            name=name, set_query=set_query, number=number, printed_total=printed_total, limit=100,
        )
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
