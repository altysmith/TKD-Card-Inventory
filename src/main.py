from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QSettings, QThread, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .catalog import PokemonCatalog
from .catalog_worker import CatalogDownloadWorker
from .database import InventoryDatabase
from .pokemon_api import PokemonTCGClient
from .scanner_tab import ScannerTab
from .settings_tab import SettingsTab


class NaturalSortItem(QTableWidgetItem):
    def __lt__(self, other: QTableWidgetItem) -> bool:
        def key(value: str):
            return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]
        return key(self.text()) < key(other.text())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TKD Card Inventory")
        self.resize(1300, 820)
        self.settings = QSettings()
        self.database = InventoryDatabase()
        self.catalog = PokemonCatalog()
        self.api = PokemonTCGClient()
        self.search_results: list[dict] = []
        self.inventory_rows: list[dict] = []
        self.catalog_thread: QThread | None = None
        self.catalog_worker: CatalogDownloadWorker | None = None

        self.tabs = QTabWidget()
        self.scanner_tab = ScannerTab(
            self.catalog, self.database, self.refresh_inventory, self.settings
        )
        self.settings_tab = SettingsTab(self.settings)
        self.settings_tab.settings_changed.connect(self.scanner_tab.refresh_settings_display)
        self.tabs.addTab(self.scanner_tab, "Scanner")
        self.tabs.addTab(self._build_search_tab(), "Find & Add Cards")
        self.tabs.addTab(self._build_inventory_tab(), "Inventory")
        self.tabs.addTab(self.settings_tab, "Settings")
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
        self.name_input.setPlaceholderText("Card name")
        self.set_input = QLineEdit()
        self.set_input.setPlaceholderText("Set name or code")
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("Collector number")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_cards)
        for widget in (self.name_input, self.set_input, self.number_input):
            widget.returnPressed.connect(self.search_cards)
        controls.addWidget(self.name_input, 2)
        controls.addWidget(self.set_input, 2)
        controls.addWidget(self.number_input, 1)
        controls.addWidget(self.search_button)
        layout.addLayout(controls)

        self.results_table = QTableWidget(0, 7)
        self.results_table.setHorizontalHeaderLabels(
            ["Card", "Type", "Set", "Set Code", "Number", "Rarity", "Card ID"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSortingEnabled(True)
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

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Selected card quantity:"))
        minus = QPushButton("−1")
        minus.clicked.connect(lambda: self.change_selected_quantity(-1))
        controls.addWidget(minus)
        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(0, 999999)
        controls.addWidget(self.quantity_input)
        plus = QPushButton("+1")
        plus.clicked.connect(lambda: self.change_selected_quantity(1))
        controls.addWidget(plus)
        set_button = QPushButton("Set Quantity")
        set_button.clicked.connect(self.set_selected_quantity)
        controls.addWidget(set_button)
        controls.addStretch()
        controls.addWidget(QLabel("Click any column heading to sort."))
        layout.addLayout(controls)

        self.inventory_table = QTableWidget(0, 8)
        self.inventory_table.setHorizontalHeaderLabels(
            ["Card", "Type", "Set", "Set Code", "Number", "Rarity", "Quantity", "Card ID"]
        )
        self.inventory_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.inventory_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.inventory_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.inventory_table.setSortingEnabled(True)
        self.inventory_table.horizontalHeader().setStretchLastSection(True)
        self.inventory_table.itemSelectionChanged.connect(self._sync_quantity_control)
        layout.addWidget(self.inventory_table)
        return page

    def _offer_catalog_download(self) -> None:
        if QMessageBox.question(
            self, "Download Pokémon Catalog",
            "Download the local card catalog? Images are not downloaded."
        ) == QMessageBox.StandardButton.Yes:
            self.download_catalog()

    def _refresh_catalog_state(self) -> None:
        count = self.catalog.card_count()
        ready = count > 0
        for widget in (self.search_button, self.name_input, self.set_input, self.number_input):
            widget.setEnabled(ready)
        self.add_cards_button.setEnabled(ready and bool(self.search_results))
        self.catalog_button.setText("Update Card Catalog" if ready else "Download Card Catalog")
        self.catalog_status.setText(
            f"Local catalog ready: {count:,} cards. Searches work offline."
            if ready else "Local catalog not installed."
        )
        self.statusBar().showMessage(f"Inventory: {self.database.path} | Catalog: {self.catalog.path}")

    def download_catalog(self) -> None:
        if self.catalog_thread is not None and self.catalog_thread.isRunning():
            return
        self.catalog_button.setEnabled(False)
        self.catalog_progress.setVisible(True)
        self.catalog_progress.setRange(0, 0)
        self.catalog_status.setText("Updating rich card metadata. Inventory remains available.")
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
        self.catalog_status.setText(f"Updating catalog: {imported:,} of {total:,} cards (page {page})")

    def _catalog_downloaded(self, imported: int) -> None:
        QMessageBox.information(
            self, "Catalog Ready",
            f"Rich card metadata is ready. {imported:,} records were downloaded or updated."
        )

    def _catalog_failed(self, message: str) -> None:
        QMessageBox.warning(
            self, "Catalog Update Paused",
            f"{message}\n\nCompleted pages were saved. Click Update Card Catalog to resume."
        )

    def _catalog_finished(self) -> None:
        self.catalog_progress.setVisible(False)
        self.catalog_button.setEnabled(True)
        self.catalog_thread = None
        self.catalog_worker = None
        self._refresh_catalog_state()

    def search_cards(self) -> None:
        name = self.name_input.text().strip()
        set_query = self.set_input.text().strip()
        number = self.number_input.text().strip()
        if not name and not set_query and not number:
            QMessageBox.information(self, "Search", "Enter a card name, set, or collector number.")
            return
        self.search_results = self.catalog.search_cards(name=name, set_query=set_query, number=number)
        self._show_search_results(self.search_results)

    def _show_search_results(self, results: list[dict]) -> None:
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(len(results))
        for row, card in enumerate(results):
            values = [
                card["name"], card.get("card_category", ""), card["set_name"],
                card.get("set_code", ""), card["number"], card.get("rarity", ""), card["id"],
            ]
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, NaturalSortItem(str(value)))
        self.results_table.setSortingEnabled(True)
        self.results_table.resizeColumnsToContents()
        self.add_cards_button.setEnabled(bool(results))

    def add_current_card(self) -> None:
        row = self.results_table.currentRow()
        if 0 <= row < len(self.search_results):
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
            self.statusBar().showMessage(f"Added {len(cards)} card(s) to inventory")

    def refresh_inventory(self) -> None:
        selected_id = None
        current = self.inventory_table.currentRow() if hasattr(self, "inventory_table") else -1
        if 0 <= current < len(self.inventory_rows):
            selected_id = self.inventory_rows[current]["card_id"]
        self.inventory_rows = self.database.list_inventory()
        self.inventory_table.setSortingEnabled(False)
        self.inventory_table.setRowCount(len(self.inventory_rows))
        reselect = -1
        for row, card in enumerate(self.inventory_rows):
            values = [
                card["card_name"], card.get("card_category", ""), card["set_name"],
                card["set_code"], card["collector_number"], card["rarity"],
                card["quantity"], card["card_id"],
            ]
            for column, value in enumerate(values):
                item = NaturalSortItem(str(value or ""))
                if column == 6:
                    item.setData(Qt.ItemDataRole.EditRole, int(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.inventory_table.setItem(row, column, item)
            if card["card_id"] == selected_id:
                reselect = row
        self.inventory_table.setSortingEnabled(True)
        self.inventory_table.resizeColumnsToContents()
        if reselect >= 0:
            self.inventory_table.selectRow(reselect)
        elif self.inventory_rows:
            self.inventory_table.selectRow(0)
        else:
            self.quantity_input.setValue(0)

    def _selected_inventory_row(self) -> int:
        row = self.inventory_table.currentRow()
        return row if 0 <= row < len(self.inventory_rows) else -1

    def _sync_quantity_control(self) -> None:
        row = self._selected_inventory_row()
        if row >= 0:
            self.quantity_input.setValue(int(self.inventory_rows[row]["quantity"]))

    def change_selected_quantity(self, change: int) -> None:
        row = self._selected_inventory_row()
        if row < 0:
            return
        card = self.inventory_rows[row]
        self.database.set_quantity(card["card_id"], max(0, int(card["quantity"]) + change))
        self.refresh_inventory()

    def set_selected_quantity(self) -> None:
        row = self._selected_inventory_row()
        if row >= 0:
            self.database.set_quantity(self.inventory_rows[row]["card_id"], self.quantity_input.value())
            self.refresh_inventory()

    def export_inventory(self, file_type: str) -> None:
        rows = self.database.list_inventory()
        if not rows:
            QMessageBox.information(self, "Export", "There is no inventory to export yet.")
            return

        suggested_name = f"TKD_Card_Inventory.{file_type}"
        export_folder = str(self.settings.value("exports/default_folder", "")).strip()
        suggested_path = str(Path(export_folder) / suggested_name) if export_folder else suggested_name
        file_filter = "CSV Files (*.csv)" if file_type == "csv" else "Excel Files (*.xlsx)"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export inventory", suggested_path, file_filter
        )
        if not filename:
            return

        path = Path(filename)
        frame = pd.DataFrame(rows)
        if file_type == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False, sheet_name="Inventory")
        QMessageBox.information(self, "Export complete", f"Inventory exported to:\n{path}")

    def _on_tab_changed(self, index: int) -> None:
        if index == 2:
            self.refresh_inventory()
        elif index == 0:
            self.scanner_tab.refresh_settings_display()
        elif index == 3:
            self.settings_tab.load_settings()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.scanner_tab.shutdown()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("TKD Card Inventory")
    app.setOrganizationName("Team KD")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
