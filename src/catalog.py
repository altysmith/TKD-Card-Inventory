from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QStandardPaths


class PokemonCatalog:
    def __init__(self) -> None:
        data_dir = Path(QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "pokemon_catalog.db"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    card_id TEXT PRIMARY KEY,
                    card_name TEXT NOT NULL,
                    set_name TEXT NOT NULL,
                    set_code TEXT,
                    collector_number TEXT NOT NULL,
                    collector_number_numeric INTEGER,
                    printed_total INTEGER,
                    rarity TEXT,
                    image_url TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS catalog_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

            existing = {row[1] for row in connection.execute("PRAGMA table_info(cards)")}
            additions = {
                "supertype": "TEXT",
                "subtypes": "TEXT",
                "card_category": "TEXT",
                "pokemon_types": "TEXT",
                "hp": "TEXT",
                "regulation_mark": "TEXT",
                "rule_box": "INTEGER NOT NULL DEFAULT 0",
                "is_promo": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in additions.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE cards ADD COLUMN {name} {definition}")

            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_cards_name
                    ON cards(card_name COLLATE NOCASE);
                CREATE INDEX IF NOT EXISTS idx_cards_set_code_number
                    ON cards(set_code COLLATE NOCASE, collector_number_numeric);
                CREATE INDEX IF NOT EXISTS idx_cards_number
                    ON cards(collector_number_numeric);
                CREATE INDEX IF NOT EXISTS idx_cards_total_number
                    ON cards(printed_total, collector_number_numeric);
                CREATE INDEX IF NOT EXISTS idx_cards_category
                    ON cards(card_category COLLATE NOCASE);
                """
            )

            connection.execute(
                """
                UPDATE cards SET set_code = 'MEW', updated_at = CURRENT_TIMESTAMP
                WHERE LOWER(set_code) = 'sv3pt5'
                   OR LOWER(set_name) IN ('151', 'scarlet & violet—151', 'scarlet & violet-151')
                """
            )

    def card_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM cards").fetchone()
        return int(row["count"])

    def is_ready(self) -> bool:
        return self.card_count() > 0

    def replace_page(self, cards: Iterable[dict[str, Any]]) -> int:
        rows = []
        for card in cards:
            raw_number = str(card.get("raw_number") or card.get("number") or "")
            digits = "".join(ch for ch in raw_number if ch.isdigit())
            number_numeric = int(digits) if digits else None
            rows.append(
                (
                    card["id"],
                    card.get("name", "Unknown"),
                    card.get("set_name", "Unknown Set"),
                    card.get("set_code", ""),
                    raw_number,
                    number_numeric,
                    card.get("printed_total"),
                    card.get("rarity", ""),
                    card.get("image_url", ""),
                    card.get("supertype", ""),
                    json.dumps(card.get("subtypes", [])),
                    card.get("card_category", ""),
                    json.dumps(card.get("pokemon_types", [])),
                    card.get("hp", ""),
                    card.get("regulation_mark", ""),
                    int(bool(card.get("rule_box"))),
                    int(bool(card.get("is_promo"))),
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO cards (
                    card_id, card_name, set_name, set_code, collector_number,
                    collector_number_numeric, printed_total, rarity, image_url,
                    supertype, subtypes, card_category, pokemon_types, hp,
                    regulation_mark, rule_box, is_promo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    card_name=excluded.card_name, set_name=excluded.set_name,
                    set_code=excluded.set_code, collector_number=excluded.collector_number,
                    collector_number_numeric=excluded.collector_number_numeric,
                    printed_total=excluded.printed_total, rarity=excluded.rarity,
                    image_url=excluded.image_url, supertype=excluded.supertype,
                    subtypes=excluded.subtypes, card_category=excluded.card_category,
                    pokemon_types=excluded.pokemon_types, hp=excluded.hp,
                    regulation_mark=excluded.regulation_mark,
                    rule_box=excluded.rule_box, is_promo=excluded.is_promo,
                    updated_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
        return len(rows)

    def set_metadata(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO catalog_metadata (key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_metadata(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM catalog_metadata WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row is not None else default

    def search_cards(
        self,
        name: str = "",
        number: str = "",
        set_query: str = "",
        category: str = "",
        printed_total: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []

        if name.strip():
            clauses.append("card_name LIKE ? COLLATE NOCASE")
            parameters.append(f"%{name.strip()}%")

        if set_query.strip():
            clauses.append("(set_name LIKE ? COLLATE NOCASE OR set_code LIKE ? COLLATE NOCASE)")
            pattern = f"%{set_query.strip()}%"
            parameters.extend([pattern, pattern])

        if number.strip():
            raw_number = number.strip().split("/")[0]
            if raw_number.isdigit():
                clauses.append("collector_number_numeric = ?")
                parameters.append(int(raw_number))
            else:
                clauses.append("collector_number LIKE ? COLLATE NOCASE")
                parameters.append(raw_number)

        if printed_total is not None:
            clauses.append("printed_total = ?")
            parameters.append(int(printed_total))

        if category.strip():
            clauses.append("card_category = ? COLLATE NOCASE")
            parameters.append(category.strip())

        if not clauses:
            return []

        parameters.append(limit)
        query = f"""
            SELECT * FROM cards
            WHERE {' AND '.join(clauses)}
            ORDER BY card_name COLLATE NOCASE, set_name COLLATE NOCASE,
                     collector_number_numeric, collector_number
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_card(row) for row in rows]

    @staticmethod
    def _row_to_card(row: sqlite3.Row) -> dict[str, Any]:
        display_number = row["collector_number"]
        if row["printed_total"]:
            display_number = f"{display_number}/{row['printed_total']}"
        return {
            "id": row["card_id"],
            "name": row["card_name"],
            "number": display_number,
            "raw_number": row["collector_number"],
            "printed_total": row["printed_total"],
            "rarity": row["rarity"] or "",
            "set_name": row["set_name"],
            "set_code": row["set_code"] or "",
            "image_url": row["image_url"] or "",
            "supertype": row["supertype"] or "",
            "subtypes": json.loads(row["subtypes"] or "[]"),
            "card_category": row["card_category"] or "",
            "pokemon_types": json.loads(row["pokemon_types"] or "[]"),
            "hp": row["hp"] or "",
            "regulation_mark": row["regulation_mark"] or "",
            "rule_box": bool(row["rule_box"]),
            "is_promo": bool(row["is_promo"]),
        }
