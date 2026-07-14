from __future__ import annotations

import time
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal, Slot

from .catalog import PokemonCatalog
from .pokemon_api import PokemonTCGClient


RICH_SCHEMA_VERSION = "2"


class CatalogDownloadWorker(QObject):
    progress = Signal(int, int, int)
    succeeded = Signal(int)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, client: PokemonTCGClient, catalog: PokemonCatalog) -> None:
        super().__init__()
        self.client = client
        self.catalog = catalog

    @Slot()
    def run(self) -> None:
        page_size = 250
        existing_count = self.catalog.card_count()
        rich_refresh = self.catalog.get_metadata("rich_schema_version", "0") != RICH_SCHEMA_VERSION

        if rich_refresh:
            # Existing rows need to be revisited once so supertype/subtype fields are populated.
            page = 1
            processed = 0
            self.catalog.set_metadata("rich_refresh_page", "0")
        else:
            saved_page = self.catalog.get_metadata("last_completed_page", "0")
            try:
                last_completed_page = int(saved_page or 0)
            except ValueError:
                last_completed_page = 0
            last_completed_page = max(last_completed_page, existing_count // page_size)
            page = last_completed_page + 1
            processed = existing_count

        try:
            total_count = 0
            while True:
                cards = None
                last_error: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        cards, total_count = self.client.fetch_card_page(page, page_size)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < 3:
                            time.sleep(attempt * 2)

                if cards is None:
                    raise RuntimeError(
                        f"Catalog page {page} failed after 3 attempts. "
                        f"Progress through page {page - 1} was saved. {last_error}"
                    )
                if not cards:
                    break

                self.catalog.replace_page(cards)
                processed = min(total_count, processed + len(cards)) if rich_refresh else self.catalog.card_count()
                if rich_refresh:
                    self.catalog.set_metadata("rich_refresh_page", str(page))
                else:
                    self.catalog.set_metadata("last_completed_page", str(page))
                self.progress.emit(processed, total_count, page)

                if processed >= total_count or len(cards) < page_size:
                    self.catalog.set_metadata("catalog_complete", "true")
                    if rich_refresh:
                        self.catalog.set_metadata("rich_schema_version", RICH_SCHEMA_VERSION)
                        self.catalog.set_metadata("last_completed_page", str(page))
                    break
                page += 1

            self.catalog.set_metadata("last_updated", datetime.now(timezone.utc).isoformat())
            self.succeeded.emit(self.catalog.card_count())
        except Exception as exc:
            self.catalog.set_metadata("catalog_complete", "false")
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
