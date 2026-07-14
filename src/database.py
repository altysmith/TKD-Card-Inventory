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
        self.catalog_path = data_dir / "pokemon_catalog.db"
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
                    card_category TEXT,
                    quantity INTEGER NOT NULL DEFAULT 1,
                    first_added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            existing = {row[1] for row in connection.execute("PRAGMA table_info(inventory)")}
            if "card_category" not in existing:
                connection.execute("ALTER TABLE inventory ADD COLUMN card_category TEXT")

    def sync_catalog_metadata(self) -> int:
        """Refresh immutable card details for existing inventory rows from the local catalog."""
        if not self.catalog_path.exists():
            return 0

        connection = self._connect()
        try:
            connection.execute("ATTACH DATABASE ? AS catalog_db", (str(self.catalog_path),))
            before = connection.total_changes
            connection.execute(
                """
                UPDATE inventory
                SET card_name = COALESCE(
                        (SELECT c.card_name FROM catalog_db.cards c
                         WHERE c.card_id = inventory.card_id),
                        card_name
                    ),
                    set_name = COALESCE(
                        (SELECT c.set_name FROM catalog_db.cards c
                         WHERE c.card_id = inventory.card_id),
                        set_name
                    ),
                    set_code = COALESCE(
                        NULLIF((SELECT c.set_code FROM catalog_db.cards c
                                WHERE c.card_id = inventory.card_id), ''),
                        set_code
                    ),
                    rarity = COALESCE(
                        NULLIF((SELECT c.rarity FROM catalog_db.cards c
                                WHERE c.card_id = inventory.card_id), ''),
                        rarity
                    ),
                    image_url = COALESCE(
                        NULLIF((SELECT c.image_url FROM catalog_db.cards c
                                WHERE c.card_id = inventory.card_id), ''),
                        image_url
                    ),
                    card_category = COALESCE(
                        NULLIF((SELECT c.card_category FROM catalog_db.cards c
                                WHERE c.card_id = inventory.card_id), ''),
                        card_category
                    )
                WHERE EXISTS (
                    SELECT 1 FROM catalog_db.cards c
                    WHERE c.card_id = inventory.card_id
                )
                """
            )
            changed = connection.total_changes - before
            connection.commit()
            connection.execute("DETACH DATABASE catalog_db")
            return changed
        finally:
            connection.close()

    def add_card(self, card: dict[str, Any], quantity: int = 1) -> None:
        quantity = max(1, int(quantity))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inventory (
                    card_id, card_name, set_name, set_code,
                    collector_number, rarity, image_url, card_category, quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    quantity = quantity + excluded.quantity,
                    card_category = CASE
                        WHEN excluded.card_category <> '' THEN excluded.card_category
                        ELSE inventory.card_category
                    END,
                    last_added = CURRENT_TIMESTAMP
                """,
                (
                    card["id"], card["name"], card["set_name"], card.get("set_code", ""),
                    card["number"], card.get("rarity", ""), card.get("image_url", ""),
                    card.get("card_category", ""), quantity,
                ),
            )

    def list_inventory(self) -> list[dict[str, Any]]:
        # Keep existing inventory rows enriched as catalog updates finish page by page.
        self.sync_catalog_metadata()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT card_id, card_name, set_name, set_code, collector_number,
                       rarity, card_category, quantity, first_added, last_added
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
                    "UPDATE inventory SET quantity = ?, last_added = CURRENT_TIMESTAMP WHERE card_id = ?",
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
