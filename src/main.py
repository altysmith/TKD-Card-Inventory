from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QThread, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .catalog import PokemonCatalog
from .catalog_worker import CatalogDownloadWorker
from .database import InventoryDatabase
from .pokemon_api import PokemonTCGClient


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TKD Card Inventory")
        self.resize(1100, 700)

        self.database = InventoryDatabase()
        self.catalog = PokemonCatalog()
        self.api = PokemonTCGClient()
        self.search_results: list[dict] = []
        self.inventory_rows: list[dict] = []
        self.catalog_thread: QThread | None = None
        self.catalog_worker: CatalogDownloadWorker | None = None

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_search_tab(), "Find & Add Cards")
        self.tabs.addTab(self._build_inventory_tab(), "Inventory")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self.refresh_inventory()
        self._refresh_catalog_state()
        if not self.catalog.is_ready():
            QTimer.singleShot(250, self._offer_catalog_download)

    def _build_search_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        heading_row = QHBoxLayout()
        heading = QLabel("Search the local Pokémon card catalog")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        heading_row.addWidget(heading)
        heading_row.addStretch()
        self.catalog_button = QPushButton("Download Card Catalog")
        self.catalog_button.clicked.connect(self.download_catalog)
        heading_row.addWidget(self.catalog_button)
        layout.addLayout(heading_row)

        self.catalog_status = QLabel()
        layout.addWidget(self.catalog_status)
        self.catalog_progress = QProgressBar()
        self.catalog_progress.setVisible(False)
        layout.addWidget(self.catalog_progress)

        controls = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Card name, e.g. Metagross")
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("Collector number, e.g. 125")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_cards)
        self.name_input.returnPressed.connect(self.search_cards)
        self.number_input.returnPressed.connect(self.search_cards)
        controls.addWidget(self.name_input, 2)
        controls.addWidget(self.number_input, 1)
        controls.addWidget(self.search_button)
        layout.addLayout(controls)

        help_label = QLabel("Select multiple rows with Ctrl+Click, or a range with Shift+Click.")
        help_label.setStyleSheet("color: #888888;")
        layout.addWidget(help_label)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Card", "Set", "Number", "Rarity", "Card ID"])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.doubleClicked.connect(self.add_current_card)
        layout.addWidget(self.results_table)

        self.add_cards_button = QPushButton("Add Selected Cards to Inventory")
        self.add_cards_button.clicked.connect(self.add_selected_cards)
        layout.addWidget(self.add_cards_button, alignment=Qt.AlignmentFlag.AlignRight)
        return page

    def _build_inventory_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        heading = QLabel("Current inventory")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        top.addWidget(heading)
        top.addStretch()
        csv_button = QPushButton("Export CSV")
        csv_button.clicked.connect(lambda: self.export_inventory("csv"))
        excel_button = QPushButton("Export Excel")
        excel_button.clicked.connect(lambda: self.export_inventory("xlsx"))
        top.addWidget(csv_button)
        top.addWidget(excel_button)
        layout.addLayout(top)

        quantity_controls = QHBoxLayout()
        quantity_controls.addWidget(QLabel("Selected card quantity:"))
        minus_button = QPushButton("−1")
        minus_button.clicked.connect(lambda: self.change_selected_quantity(-1))
        quantity_controls.addWidget(minus_button)
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(0, 999999)
        self.quantity_input.setValue(1)
        self.quantity_input.setMinimumWidth(100)
        quantity_controls.addWidget(self.quantity_input)
        plus_button = QPushButton("+1")
        plus_button.clicked.connect(lambda: self.change_selected_quantity(1))
        quantity_controls.addWidget(plus_button)
        set_button = QPushButton("Set Quantity")
        set_button.clicked.connect(self.set_selected_quantity)
        quantity_controls.addWidget(set_button)
        quantity_controls.addStretch()
        quantity_controls.addWidget(QLabel("Setting quantity to 0 removes the card."))
        layout.addLayout(quantity_controls)

        self.inventory_table = QTableWidget(0, 7)
        self.inventory_table.setHorizontalHeaderLabels(
            ["Card", "Set", "Set Code", "Number", "Rarity", "Quantity", "Card ID"]
        )
        self.inventory_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.inventory_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.inventory_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.inventory_table.horizontalHeader().setStretchLastSection(True)
        self.inventory_table.itemSelectionChanged.connect(self._sync_quantity_control)
        layout.addWidget(self.inventory_table)
        return page

    def _offer_catalog_download(self) -> None:
        answer = QMessageBox.question(
            self,
            "Download Pokémon Catalog",
            "TKD Card Inventory needs a local card catalog for fast offline searches.\n\n"
            "Download card metadata now? Images are not downloaded.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.download_catalog()

    def _refresh_catalog_state(self) -> None:
        count = self.catalog.card_count()
        ready = count > 0
        self.search_button.setEnabled(ready)
        self.name_input.setEnabled(ready)
        self.number_input.setEnabled(ready)
        self.add_cards_button.setEnabled(ready and bool(self.search_results))
        self.catalog_button.setText("Update Card Catalog" if ready else "Download Card Catalog")
        self.catalog_status.setText(
            f"Local catalog ready: {count:,} cards. Searches work offline."
            if ready
            else "Local catalog not installed. Download it once to enable offline searching."
        )
        self.statusBar().showMessage(
            f"Inventory: {self.database.path} | Catalog: {self.catalog.path}"
        )

    def download_catalog(self) -> None:
        if self.catalog_thread is not None and self.catalog_thread.isRunning():
            return
        self.catalog_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.catalog_progress.setVisible(True)
        self.catalog_progress.setRange(0, 0)
        self.catalog_status.setText("Downloading card metadata. You can continue using Inventory.")

        self.catalog_thread = QThread(self)
        self.catalog_worker = CatalogDownloadWorker(self.api, self.catalog)
        self.catalog_worker.moveToThread(self.catalog_thread)
        self.catalog_thread.started.connect(self.catalog_worker.run)
        self.catalog_worker.progress.connect(self._catalog_progress)
        self.catalog_worker.succeeded.connect(self._catalog_downloaded)
        self.catalog_worker.failed.connect(self._catalog_failed)
        self.catalog_worker.finished.connect(self.catalog_thread.quit)
        self.catalog_worker.finished.connect(self.catalog_worker.deleteLater)
        self.catalog_thread.finished.connect(self.catalog_thread.deleteLater)
        self.catalog_thread.finished.connect(self._catalog_finished)
        self.catalog_thread.start()

    def _catalog_progress(self, imported: int, total: int, page: int) -> None:
        if total > 0:
            self.catalog_progress.setRange(0, total)
            self.catalog_progress.setValue(min(imported, total))
        self.catalog_status.setText(
            f"Downloading catalog: {imported:,} of {total:,} cards imported (page {page})"
        )

    def _catalog_downloaded(self, imported: int) -> None:
        self.catalog_status.setText(f"Catalog ready. {self.catalog.card_count():,} cards available offline.")
        QMessageBox.information(
            self,
            "Catalog Ready",
            f"The local Pokémon catalog is ready.\n\n{imported:,} card records were downloaded or updated.",
        )

    def _catalog_failed(self, message: str) -> None:
        self.catalog_status.setText("Catalog download paused. Existing downloaded pages were kept.")
        QMessageBox.warning(
            self,
            "Catalog Download Failed",
            f"{message}\n\nAny completed pages were saved. Click Update Card Catalog to try again.",
        )

    def _catalog_finished(self) -> None:
        self.catalog_progress.setVisible(False)
        self.catalog_button.setEnabled(True)
        self.catalog_thread = None
        self.catalog_worker = None
        self._refresh_catalog_state()

    def search_cards(self) -> None:
        name = self.name_input.text().strip()
        number = self.number_input.text().strip()
        if not name and not number:
            QMessageBox.information(self, "Search", "Enter a card name or collector number.")
            return
        if not self.catalog.is_ready():
            QMessageBox.information(self, "Catalog Required", "Download the local card catalog first.")
            return
        self.search_results = self.catalog.search_cards(name, number)
        self._show_search_results(self.search_results)

    def _show_search_results(self, results: list[dict]) -> None:
        self.results_table.setRowCount(len(results))
        for row_index, card in enumerate(results):
            values = [card["name"], card["set_name"], card["number"], card.get("rarity", ""), card["id"]]
            for column, value in enumerate(values):
                self.results_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.results_table.resizeColumnsToContents()
        self.add_cards_button.setEnabled(bool(results))
        self.statusBar().showMessage(
            f"Found {len(results)} matching cards locally" if results else "No matching cards found"
        )

    def add_current_card(self) -> None:
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.search_results):
            QMessageBox.information(self, "Add card", "Select a card first.")
            return
        self._add_rows([row])

    def add_selected_cards(self) -> None:
        rows = sorted({index.row() for index in self.results_table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "Add cards", "Select one or more cards first.")
            return
        self._add_rows(rows)

    def _add_rows(self, rows: list[int]) -> None:
        cards = [self.search_results[row] for row in rows if 0 <= row < len(self.search_results)]
        for card in cards:
            self.database.add_card(card)
        if cards:
            self.refresh_inventory()
            self.statusBar().showMessage(f"Added {len(cards)} card{'s' if len(cards) != 1 else ''} to inventory")

    def refresh_inventory(self) -> None:
        selected_id = None
        current_row = self.inventory_table.currentRow() if hasattr(self, "inventory_table") else -1
        if 0 <= current_row < len(self.inventory_rows):
            selected_id = self.inventory_rows[current_row]["card_id"]
        self.inventory_rows = self.database.list_inventory()
        self.inventory_table.setRowCount(len(self.inventory_rows))
        reselect = -1
        for row_index, card in enumerate(self.inventory_rows):
            values = [card["card_name"], card["set_name"], card["set_code"], card["collector_number"], card["rarity"], card["quantity"], card["card_id"]]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                if column == 5:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.inventory_table.setItem(row_index, column, item)
            if card["card_id"] == selected_id:
                reselect = row_index
        self.inventory_table.resizeColumnsToContents()
        if reselect >= 0:
            self.inventory_table.selectRow(reselect)
        elif self.inventory_rows:
            self.inventory_table.selectRow(0)
        else:
            self.quantity_input.setValue(0)

    def _selected_inventory_row(self) -> int:
        row = self.inventory_table.currentRow()
        if row < 0 or row >= len(self.inventory_rows):
            QMessageBox.information(self, "Quantity", "Select an inventory row first.")
            return -1
        return row

    def _sync_quantity_control(self) -> None:
        row = self.inventory_table.currentRow()
        if 0 <= row < len(self.inventory_rows):
            self.quantity_input.setValue(int(self.inventory_rows[row]["quantity"]))

    def change_selected_quantity(self, change: int) -> None:
        row = self._selected_inventory_row()
        if row < 0:
            return
        card = self.inventory_rows[row]
        quantity = max(0, int(card["quantity"]) + change)
        self.database.set_quantity(card["card_id"], quantity)
        self.refresh_inventory()

    def set_selected_quantity(self) -> None:
        row = self._selected_inventory_row()
        if row < 0:
            return
        self.database.set_quantity(self.inventory_rows[row]["card_id"], self.quantity_input.value())
        self.refresh_inventory()

    def export_inventory(self, file_type: str) -> None:
        rows = self.database.list_inventory()
        if not rows:
            QMessageBox.information(self, "Export", "There is no inventory to export yet.")
            return
        suggested = f"TKD_Card_Inventory.{file_type}"
        file_filter = "CSV Files (*.csv)" if file_type == "csv" else "Excel Files (*.xlsx)"
        filename, _ = QFileDialog.getSaveFileName(self, "Export inventory", suggested, file_filter)
        if not filename:
            return
        path = Path(filename)
        try:
            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False) if file_type == "csv" else frame.to_excel(path, index=False, sheet_name="Inventory")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export complete", f"Inventory exported to:\n{path}")

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self.refresh_inventory()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("TKD Card Inventory")
    app.setOrganizationName("Team KD")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
