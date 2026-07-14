from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .database import InventoryDatabase
from .pokemon_api import PokemonTCGClient


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TKD Card Inventory")
        self.resize(1100, 700)

        self.database = InventoryDatabase()
        self.api = PokemonTCGClient()
        self.search_results: list[dict] = []
        self.inventory_rows: list[dict] = []

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_search_tab(), "Find & Add Cards")
        self.tabs.addTab(self._build_inventory_tab(), "Inventory")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage(f"Database: {self.database.path}")
        self.refresh_inventory()

    def _build_search_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        heading = QLabel("Search the Pokémon card catalog")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(heading)

        controls = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Card name, e.g. Charizard ex")
        self.number_input = QLineEdit()
        self.number_input.setPlaceholderText("Collector number, e.g. 125")
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search_cards)
        self.name_input.returnPressed.connect(self.search_cards)
        self.number_input.returnPressed.connect(self.search_cards)
        controls.addWidget(self.name_input, 2)
        controls.addWidget(self.number_input, 1)
        controls.addWidget(search_button)
        layout.addLayout(controls)

        selection_help = QLabel(
            "Select multiple rows with Ctrl+Click, or select a range with Shift+Click."
        )
        selection_help.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(selection_help)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Card", "Set", "Number", "Rarity", "Card ID"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.doubleClicked.connect(self.add_current_card)
        layout.addWidget(self.results_table)

        add_button = QPushButton("Add Selected Cards to Inventory")
        add_button.clicked.connect(self.add_selected_cards)
        layout.addWidget(add_button, alignment=Qt.AlignmentFlag.AlignRight)
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
        remove_button = QPushButton("Remove One")
        remove_button.clicked.connect(self.remove_selected_card)
        top.addWidget(remove_button)
        top.addWidget(csv_button)
        top.addWidget(excel_button)
        layout.addLayout(top)

        self.inventory_table = QTableWidget(0, 7)
        self.inventory_table.setHorizontalHeaderLabels(
            ["Card", "Set", "Set Code", "Number", "Rarity", "Quantity", "Card ID"]
        )
        self.inventory_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.inventory_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.inventory_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.inventory_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.inventory_table)
        return page

    def search_cards(self) -> None:
        name = self.name_input.text().strip()
        number = self.number_input.text().strip()
        if not name and not number:
            QMessageBox.information(self, "Search", "Enter a card name or collector number.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.search_results = self.api.search_cards(name, number)
        except Exception as exc:
            QMessageBox.critical(self, "Search failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.results_table.setRowCount(len(self.search_results))
        for row_index, card in enumerate(self.search_results):
            values = [
                card["name"],
                card["set_name"],
                card["number"],
                card.get("rarity", ""),
                card["id"],
            ]
            for column, value in enumerate(values):
                self.results_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.results_table.resizeColumnsToContents()
        self.statusBar().showMessage(f"Found {len(self.search_results)} matching cards")

    def add_current_card(self) -> None:
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.search_results):
            QMessageBox.information(self, "Add card", "Select a card first.")
            return
        self._add_rows([row])

    def add_selected_cards(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.results_table.selectionModel().selectedRows()}
        )
        if not selected_rows:
            QMessageBox.information(self, "Add cards", "Select one or more cards first.")
            return
        self._add_rows(selected_rows)

    def _add_rows(self, rows: list[int]) -> None:
        added_cards: list[dict] = []
        for row in rows:
            if 0 <= row < len(self.search_results):
                card = self.search_results[row]
                self.database.add_card(card)
                added_cards.append(card)

        if not added_cards:
            return

        self.refresh_inventory()
        if len(added_cards) == 1:
            card = added_cards[0]
            self.statusBar().showMessage(
                f"Added {card['name']} — {card['set_name']} {card['number']}"
            )
        else:
            self.statusBar().showMessage(
                f"Added {len(added_cards)} selected cards to inventory"
            )

    def refresh_inventory(self) -> None:
        self.inventory_rows = self.database.list_inventory()
        self.inventory_table.setRowCount(len(self.inventory_rows))
        for row_index, card in enumerate(self.inventory_rows):
            values = [
                card["card_name"],
                card["set_name"],
                card["set_code"],
                card["collector_number"],
                card["rarity"],
                card["quantity"],
                card["card_id"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value or ""))
                if column == 5:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.inventory_table.setItem(row_index, column, item)
        self.inventory_table.resizeColumnsToContents()

    def remove_selected_card(self) -> None:
        row = self.inventory_table.currentRow()
        if row < 0 or row >= len(self.inventory_rows):
            QMessageBox.information(self, "Remove card", "Select an inventory row first.")
            return
        self.database.remove_one(self.inventory_rows[row]["card_id"])
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
        frame = pd.DataFrame(rows)
        try:
            if file_type == "csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_excel(path, index=False, sheet_name="Inventory")
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
