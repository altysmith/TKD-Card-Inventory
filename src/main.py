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
    QSpinBox,
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
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_cards)
        self.name_input.returnPressed.connect(self.search_cards)
        self.number_input.returnPressed.connect(self.search_cards)
        controls.addWidget(self.name_input, 2)
        controls.addWidget(self.number_input, 1)
        controls.addWidget(self.search_button)
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

    def search_cards(self) -> None:
        name = self.name_input.text().strip()
        number = self.number_input.text().strip()
        if not name and not number:
            QMessageBox.information(self, "Search", "Enter a card name or collector number.")
            return

        # Clear the old result set immediately so a failed search cannot leave stale cards
        # available for selection and accidental inventory additions.
        self.search_results = []
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.search_button.setEnabled(False)
        self.statusBar().showMessage("Searching the Pokémon card catalog...")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = self.api.search_cards(name, number)
        except Exception as exc:
            self.statusBar().showMessage("Search failed")
            QMessageBox.warning(self, "Search failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.search_button.setEnabled(True)

        self.search_results = results
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

        if self.search_results:
            self.statusBar().showMessage(f"Found {len(self.search_results)} matching cards")
        else:
            self.statusBar().showMessage("No matching cards found")

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
        selected_card_id = None
        current_row = self.inventory_table.currentRow() if hasattr(self, "inventory_table") else -1
        if 0 <= current_row < len(self.inventory_rows):
            selected_card_id = self.inventory_rows[current_row]["card_id"]

        self.inventory_rows = self.database.list_inventory()
        self.inventory_table.setRowCount(len(self.inventory_rows))
        row_to_reselect = -1

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
            if card["card_id"] == selected_card_id:
                row_to_reselect = row_index

        self.inventory_table.resizeColumnsToContents()
        if row_to_reselect >= 0:
            self.inventory_table.selectRow(row_to_reselect)
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
        new_quantity = max(0, int(card["quantity"]) + change)
        self.database.set_quantity(card["card_id"], new_quantity)
        self.refresh_inventory()
        self.statusBar().showMessage(
            f"Updated {card['card_name']} quantity to {new_quantity}"
        )

    def set_selected_quantity(self) -> None:
        row = self._selected_inventory_row()
        if row < 0:
            return
        card = self.inventory_rows[row]
        new_quantity = self.quantity_input.value()
        self.database.set_quantity(card["card_id"], new_quantity)
        self.refresh_inventory()
        self.statusBar().showMessage(
            f"Updated {card['card_name']} quantity to {new_quantity}"
        )

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
