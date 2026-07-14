from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from PySide6.QtCore import QStandardPaths


class InventoryDatabase:
    def __init__(self) -> None:
        data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "inventory.db"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    card_id TEXT PRIMARY KEY,
                    card_name TEXT NOT NULL,
                    set_name TEXT NOT NULL,
                    set_code TEXT,
                    collector_number TEXT NOT NULL,
                    rarity TEXT,
                    image_url TEXT,
                    quantity INTEGER NOT NULL DEFAULT 1,
                    first_added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def add_card(self, card: dict[str, Any], quantity: int = 1) -> None:
        quantity = max(1, int(quantity))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inventory (
                    card_id, card_name, set_name, set_code,
                    collector_number, rarity, image_url, quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    quantity = quantity + excluded.quantity,
                    last_added = CURRENT_TIMESTAMP
                """,
                (
                    card["id"],
                    card["name"],
                    card["set_name"],
                    card.get("set_code", ""),
                    card["number"],
                    card.get("rarity", ""),
                    card.get("image_url", ""),
                    quantity,
                ),
            )

    def list_inventory(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT card_id, card_name, set_name, set_code,
                       collector_number, rarity, quantity,
                       first_added, last_added
                FROM inventory
                ORDER BY card_name COLLATE NOCASE, set_name, collector_number
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def set_quantity(self, card_id: str, quantity: int) -> None:
        quantity = max(0, int(quantity))
        with self._connect() as connection:
            if quantity == 0:
                connection.execute("DELETE FROM inventory WHERE card_id = ?", (card_id,))
            else:
                connection.execute(
                    """
                    UPDATE inventory
                    SET quantity = ?, last_added = CURRENT_TIMESTAMP
                    WHERE card_id = ?
                    """,
                    (quantity, card_id),
                )

    def remove_one(self, card_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT quantity FROM inventory WHERE card_id = ?", (card_id,)
            ).fetchone()
            if row is None:
                return
            if row["quantity"] <= 1:
                connection.execute("DELETE FROM inventory WHERE card_id = ?", (card_id,))
            else:
                connection.execute(
                    "UPDATE inventory SET quantity = quantity - 1 WHERE card_id = ?",
                    (card_id,),
                )
