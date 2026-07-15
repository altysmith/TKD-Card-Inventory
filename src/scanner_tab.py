from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .catalog import PokemonCatalog
from .database import InventoryDatabase
from .ocr_engine import CardOCREngine, OCRResult


class ScannerTab(QWidget):
    """Simple manual camera scanner with temporary OCR debug previews."""

    CARD_ASPECT_RATIO = 0.714
    BURST_FRAME_COUNT = 6

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
            "Place the card inside the fixed guide, then click Capture & Read Card or press Space. "
            "The app captures a short burst, keeps the sharpest still, reads the identifier, and checks the local catalog."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        root.addWidget(note)

        self.settings_status = QLabel()
        self.settings_status.setStyleSheet("color: #888888;")
        root.addWidget(self.settings_status)

        self.placement_status = QLabel("Start the camera and position a card inside the guide.")
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

        self.resume_button = QPushButton("Return to Live View")
        self.resume_button.clicked.connect(self.resume_live_view)
        self.resume_button.setEnabled(False)

        for button in (self.camera_button, self.capture_button, self.resume_button):
            buttons.addWidget(button)
        camera_column.addLayout(buttons)
        body.addLayout(camera_column, 3)

        self.capture_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.capture_shortcut.activated.connect(self.capture_card)

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
        self.debug_text.setMaximumHeight(120)
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
            f"• Mode: {mode} • Threshold: {confidence}% • Manual capture"
        )
        debug_enabled = str(self.settings.value("scanner/debug", "true")).casefold() in {
            "1",
            "true",
            "yes",
        }
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
                self,
                "Camera unavailable",
                f"Camera {camera_index} could not be opened. Choose another camera in Settings or check permissions.",
            )
            return

        if "x" in resolution:
            width_text, height_text = resolution.split("x", 1)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width_text))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height_text))
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.camera = camera
        self._active_signature = self._settings_signature()
        self.timer.start(30)
        self.camera_button.setText("Stop Camera")
        self.capture_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.enhanced_button.setEnabled(False)
        self.captured_frame = None
        self.placement_status.setText(
            "Live view ready. Position the card inside the white guide, then click Capture & Read Card or press Space."
        )
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
        self.resume_button.setEnabled(False)
        self.enhanced_button.setEnabled(False)
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Camera is off")
        self.placement_status.setText("Start the camera and position a card inside the guide.")

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
        return (
            max(0, (width - guide_width) // 2),
            max(0, (height - guide_height) // 2),
            guide_width,
            guide_height,
        )

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

    def _display_frame(self, frame) -> None:
        pixmap = self._frame_to_pixmap(frame, self.preview.size())
        painter = QPainter(pixmap)
        left, top, guide_width, guide_height = self._guide_rect(
            pixmap.width(), pixmap.height()
        )
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawRoundedRect(left, top, guide_width, guide_height, 12, 12)

        mode = str(self.settings.value("scanner/mode", "full"))
        painter.setPen(QPen(Qt.GlobalColor.green, 3))
        if mode == "identifier":
            identifier_left = left + int(guide_width * 0.02)
            identifier_top = top + int(guide_height * 0.58)
            identifier_width = int(guide_width * 0.60)
            identifier_height = int(guide_height * 0.36)
        else:
            identifier_left = left + int(guide_width * 0.035)
            identifier_top = top + int(guide_height * 0.900)
            identifier_width = int(guide_width * 0.435)
            identifier_height = int(guide_height * 0.072)
        painter.drawRect(
            identifier_left,
            identifier_top,
            max(1, identifier_width),
            max(1, identifier_height),
        )
        painter.end()
        self.preview.setPixmap(pixmap)

    @staticmethod
    def _frame_to_pixmap(frame, size) -> QPixmap:
        if len(frame.shape) == 2:
            image = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1],
                QImage.Format.Format_Grayscale8,
            )
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.shape[2] * rgb.shape[1],
                QImage.Format.Format_RGB888,
            )
        return QPixmap.fromImage(image.copy()).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    @staticmethod
    def _sharpness(frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def capture_card(self) -> None:
        if self.current_frame is None or self.camera is None or not self.camera.isOpened():
            return
        if not self.timer.isActive():
            return

        self.capture_button.setEnabled(False)
        self.placement_status.setText("Capturing a short burst and selecting the sharpest frame…")

        orientation = str(self.settings.value("scanner/orientation", "landscape"))
        candidates = [self.current_frame.copy()]
        for _ in range(self.BURST_FRAME_COUNT - 1):
            ok, frame = self.camera.read()
            if ok:
                candidates.append(self._rotate_frame(frame, orientation))

        self.captured_frame = max(candidates, key=self._sharpness)
        candidates.clear()
        self.timer.stop()
        self._display_frame(self.captured_frame)
        self.resume_button.setEnabled(True)
        self.enhanced_button.setEnabled(True)
        self.placement_status.setText("Sharpest temporary still selected. Running OCR…")
        self.run_ocr(enhanced=False)

    def retry_enhanced_ocr(self) -> None:
        if self.captured_frame is None:
            QMessageBox.information(
                self, "Enhanced OCR", "There is no current capture to retry."
            )
            return
        self.run_ocr(enhanced=True)

    def run_ocr(self, enhanced: bool = False) -> None:
        if self.captured_frame is None:
            return
        self.ocr_status.setText("Running enhanced OCR…" if enhanced else "Running OCR…")
        mode = str(self.settings.value("scanner/mode", "full"))
        try:
            result = self.ocr.read_card(
                self.captured_frame, mode=mode, enhanced=enhanced
            )
        except RuntimeError as exc:
            self.ocr_status.setText("OCR unavailable. Manual identification still works.")
            QMessageBox.warning(self, "OCR unavailable", str(exc))
            return
        except Exception as exc:
            self.ocr_status.setText("OCR could not read this capture.")
            QMessageBox.warning(self, "OCR failed", str(exc))
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

        details = [
            f"Recognition confidence: {result.confidence}%",
            f"time: {result.processing_ms} ms",
        ]
        if result.set_code:
            details.append(f"set: {result.set_code}")
        if result.collector_number:
            number = result.collector_number + (
                f"/{result.printed_total}" if result.printed_total else ""
            )
            details.append(f"number: {number}")
        self.ocr_status.setText(" • ".join(details))

        if result.set_code or result.collector_number:
            self.find_matches(printed_total=result.printed_total)

        threshold = int(self.settings.value("scanner/confidence", 90))
        if result.confidence >= threshold and len(self.matches) == 1:
            self.matches_table.selectRow(0)
            self.placement_status.setText(
                "Match ready. Choose quantity and add it to the queue, or return to live view."
            )
            return

        self.ocr_status.setText(
            self.ocr_status.text() + " • Review results or retry enhanced OCR"
        )
        self.placement_status.setText(
            "Review the result. Use Enhanced Retry, edit the fields manually, or return to live view."
        )

    def _show_debug_result(self, result: OCRResult, enhanced: bool) -> None:
        if result.crop_image is not None:
            self.debug_crop.setPixmap(
                self._frame_to_pixmap(result.crop_image, self.debug_crop.size())
            )
        if result.processed_image is not None:
            self.debug_processed.setPixmap(
                self._frame_to_pixmap(result.processed_image, self.debug_processed.size())
            )
        self.debug_metrics.setText(
            f"Pass: {'Enhanced' if enhanced else 'Standard'} • Processing: {result.processing_ms} ms • "
            f"Crop sharpness: {result.sharpness:.1f} • Confidence: {result.confidence}%"
        )
        self.debug_text.setPlainText(result.raw_text or "No OCR text was returned.")

    def resume_live_view(self) -> None:
        if self.camera is None or not self.camera.isOpened():
            return
        self.captured_frame = None
        self.enhanced_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.capture_button.setEnabled(True)
        if not self.timer.isActive():
            self.timer.start(30)
        self.placement_status.setText(
            "Live view ready. Position the next card and click Capture & Read Card or press Space."
        )

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
            QMessageBox.information(
                self, "Identify card", "Enter a set acronym, number, or card name."
            )
            return

        self.matches = self.catalog.search_cards(
            name=name,
            set_query=set_query,
            number=number,
            printed_total=printed_total,
            limit=100,
        )
        used_relaxed_identifier_match = False
        if not self.matches and set_query and number:
            # Keep the reliable collector fraction exact while relaxing only
            # the noisy set badge. For 67/86 this narrows the catalog to three
            # candidates before BLN is compared with the real BLK code.
            self.matches = self.catalog.search_cards(
                name=name,
                number=number,
                printed_total=printed_total,
                limit=100,
            )
            self.matches = self.ocr.rank_catalog_candidates(
                self.matches, set_query
            )
            if (
                self.matches
                and self.ocr.set_code_similarity(
                    set_query, str(self.matches[0].get("set_code", ""))
                )
                < 0.60
            ):
                # The collector total was probably the noisy field. Do not let
                # unrelated exact-total matches prevent the number-only retry.
                self.matches = []
            used_relaxed_identifier_match = bool(self.matches)

        if not self.matches and number:
            # OCR commonly gets one character of a small set badge or printed
            # total wrong (for example BLK -> BLN or 086 -> 066). The collector
            # number is usually more reliable, so retrieve its candidates and
            # let the catalog rank/correct the noisy fields.
            self.matches = self.catalog.search_cards(
                name=name,
                number=number,
                limit=100,
            )
            if set_query:
                self.matches = self.ocr.rank_catalog_candidates(
                    self.matches, set_query
                )
            used_relaxed_identifier_match = bool(self.matches)

        if self.matches and len(self.matches) == 1 and number and not set_query:
            best_code = str(self.matches[0].get("set_code", ""))
            if best_code:
                self.set_input.setText(best_code)
                self.number_input.setText(str(self.matches[0].get("number", number)))
                self.ocr_status.setText(
                    self.ocr_status.text()
                    + f" | Catalog identified set {best_code}"
                )
        elif used_relaxed_identifier_match and set_query:
            best_code = str(self.matches[0].get("set_code", ""))
            best_score = self.ocr.set_code_similarity(set_query, best_code)
            second_score = (
                self.ocr.set_code_similarity(
                    set_query, str(self.matches[1].get("set_code", ""))
                )
                if len(self.matches) > 1
                else 0.0
            )
            if best_code and best_score >= 0.60 and best_score - second_score >= 0.15:
                self.matches = [
                    card
                    for card in self.matches
                    if str(card.get("set_code", "")).casefold()
                    == best_code.casefold()
                ]
                self.set_input.setText(best_code)
                self.number_input.setText(str(self.matches[0].get("number", number)))
                self.ocr_status.setText(
                    self.ocr_status.text()
                    + f" | Catalog resolved {best_code} {self.matches[0].get('number', number)}"
                )
        self.matches_table.setRowCount(len(self.matches))
        for row, card in enumerate(self.matches):
            values = [
                card["name"],
                card.get("card_category", ""),
                card["set_name"],
                card.get("set_code", ""),
                card["number"],
                card.get("rarity", ""),
            ]
            for column, value in enumerate(values):
                self.matches_table.setItem(
                    row, column, QTableWidgetItem(str(value))
                )
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
        existing = next(
            (item for item in self.queue if item["card"]["id"] == card["id"]),
            None,
        )
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
        self.ocr_status.setText("Added to queue. Ready for the next card.")
        self.resume_live_view()

    def _refresh_queue(self) -> None:
        self.queue_table.setRowCount(len(self.queue))
        for row, item in enumerate(self.queue):
            card = item["card"]
            values = [
                card["name"],
                card.get("card_category", ""),
                card["set_name"],
                card["number"],
                item["quantity"],
                card["id"],
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
            self,
            "Inventory updated",
            f"Added {total} cards across {unique} unique printings.",
        )

    def shutdown(self) -> None:
        self.stop_camera()
