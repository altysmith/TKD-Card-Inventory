from __future__ import annotations

from collections.abc import Callable

import cv2
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
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

        self._baseline_gray = None
        self._previous_gray = None
        self._baseline_frames = 0
        self._stable_frames = 0
        self._clear_frames = 0
        self._capture_pending = False
        self._needs_clear = False

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
            "Start with the guide empty for a moment. Then place a card inside the white guide "
            "and hold it still. The app captures a temporary still and prioritizes the collector "
            "number and printed set total; a printed set acronym is not required."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888;")
        root.addWidget(note)

        self.settings_status = QLabel()
        self.settings_status.setStyleSheet("color: #888888;")
        root.addWidget(self.settings_status)

        self.placement_status = QLabel("Start the camera with the guide empty.")
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
        self.set_input.setPlaceholderText("Set name or code, optional")
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
        self._reset_auto_capture(learn_baseline=True)
        self.timer.start(30)
        self.camera_button.setText("Stop Camera")
        self.capture_button.setEnabled(True)
        self.placement_status.setText("Keep the guide empty while the scanner calibrates…")
        self.refresh_settings_display()

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.current_frame = None
        self.captured_frame = None
        self._reset_auto_capture(learn_baseline=True)
        self.camera_button.setText("Start Camera")
        self.capture_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.preview.setText("Camera is off")
        self.preview.setPixmap(QPixmap())
        self.placement_status.setText("Start the camera with the guide empty.")

    @staticmethod
    def _guide_crop(frame):
        height, width = frame.shape[:2]
        guide_height = int(height * 0.82)
        guide_width = min(width, int(guide_height * 0.714))
        left = max(0, (width - guide_width) // 2)
        top = max(0, (height - guide_height) // 2)
        return frame[top : top + guide_height, left : left + guide_width]

    @staticmethod
    def _prepare_detection_frame(frame):
        guide = ScannerTab._guide_crop(frame)
        gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 224), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0)

    def _reset_auto_capture(self, learn_baseline: bool) -> None:
        self._previous_gray = None
        self._stable_frames = 0
        self._clear_frames = 0
        self._capture_pending = False
        self._needs_clear = not learn_baseline
        if learn_baseline:
            self._baseline_gray = None
            self._baseline_frames = 0

    def _read_camera_frame(self) -> None:
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            return

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
            self._baseline_frames = self._baseline_frames + 1 if motion <= 2.5 else 0
            self.placement_status.setText(
                f"Calibrating empty guide… {min(100, self._baseline_frames * 10)}%"
            )
            if self._baseline_frames >= 10:
                self._baseline_gray = current.copy()
                self._previous_gray = current
                self.placement_status.setText("Ready — place a card inside the white guide.")
            return

        change = cv2.absdiff(current, self._baseline_gray)
        changed_ratio = float((change > 18).mean())
        mean_change = float(change.mean())
        motion = 999.0
        if self._previous_gray is not None:
            motion = float(cv2.absdiff(current, self._previous_gray).mean())
        self._previous_gray = current
        scene_changed = changed_ratio >= 0.18 or mean_change >= 8.0

        if self._needs_clear:
            self._clear_frames = self._clear_frames + 1 if not scene_changed else 0
            self.placement_status.setText("Remove the scanned card to arm the next capture.")
            if self._clear_frames >= 5:
                self._needs_clear = False
                self._stable_frames = 0
                self.placement_status.setText("Ready — place the next card inside the guide.")
            return

        if not scene_changed:
            self._stable_frames = 0
            self.placement_status.setText("Ready — place a card inside the white guide.")
            return

        self._stable_frames = self._stable_frames + 1 if motion <= 3.5 else 0
        required = 8
        progress = min(100, int((self._stable_frames / required) * 100))
        self.placement_status.setText(
            f"Card detected. Hold still… {progress}% (scene change {int(changed_ratio * 100)}%)"
        )

        if self._stable_frames >= required:
            self._capture_pending = True
            self.placement_status.setText("Card steady — capturing still image…")
            QTimer.singleShot(0, self.capture_card)

    def _display_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
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
            self._resume_after_failed_read()
            return
        except Exception as exc:
            self.ocr_status.setText("OCR could not read this capture. Try better lighting or manual entry.")
            QMessageBox.warning(self, "OCR failed", str(exc))
            self._resume_after_failed_read()
            return
        finally:
            self.captured_frame = None

        self.card_name_input.clear()
        self.set_input.clear()
        self.number_input.clear()
        self.matches = []
        self.matches_table.setRowCount(0)

        if result.collector_number:
            display_number = result.collector_number
            if result.printed_total:
                display_number = f"{display_number}/{result.printed_total}"
            self.number_input.setText(display_number)
        if result.set_code:
            self.set_input.setText(result.set_code)
        if result.card_name and result.confidence >= 55:
            self.card_name_input.setText(result.card_name)

        threshold = int(self.settings.value("scanner/confidence", 90))
        pieces = [f"OCR confidence: {result.confidence}%"]
        if result.collector_number:
            number_text = result.collector_number
            if result.printed_total:
                number_text += f"/{result.printed_total}"
            pieces.append(f"number: {number_text}")
        if result.set_code:
            pieces.append(f"set: {result.set_code}")
        if result.card_name and result.confidence >= 55:
            pieces.append(f"name: {result.card_name}")
        self.ocr_status.setText(" • ".join(pieces))

        has_identifier = bool(
            result.collector_number
            or result.set_code
            or (result.card_name and result.confidence >= 55)
        )
        if has_identifier:
            self.find_matches(printed_total=result.printed_total)

        accepted = result.confidence >= threshold and len(self.matches) == 1
        if accepted:
            self.matches_table.selectRow(0)
            self.ocr_status.setText(
                self.ocr_status.text() + " • High-confidence single match selected"
            )
            self.placement_status.setText("Match ready. Choose quantity and add it to the queue.")
            return

        if result.confidence < threshold:
            self.ocr_status.setText(
                self.ocr_status.text() + " • Below threshold; scanner rearmed for the next card"
            )
        elif len(self.matches) != 1:
            self.ocr_status.setText(
                self.ocr_status.text() + " • No unique match; scanner rearmed for the next card"
            )
        else:
            self.ocr_status.setText(
                f"OCR confidence: {result.confidence}% • No reliable collector information found."
            )

        self._resume_after_failed_read()

    def _resume_after_failed_read(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self._reset_auto_capture(learn_baseline=False)
            self.timer.start(30)
            self.capture_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.placement_status.setText("Remove this card to arm the next capture.")

    def resume_live_view(self) -> None:
        if self.camera is not None and self.camera.isOpened():
            self._reset_auto_capture(learn_baseline=False)
            self.timer.start(30)
            self.capture_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.placement_status.setText("Remove the scanned card to arm the next capture.")

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
            QMessageBox.information(self, "Identify card", "Enter a name, set, or collector number.")
            return

        self.matches = self.catalog.search_cards(
            name=name,
            set_query=set_query,
            number=number,
            printed_total=printed_total,
            limit=100,
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
            self,
            "Clear queue",
            "Remove every card from the scan queue?",
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
