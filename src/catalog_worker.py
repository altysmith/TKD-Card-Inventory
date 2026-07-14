from __future__ import annotations

import time
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal, Slot

from .catalog import PokemonCatalog
from .pokemon_api import PokemonTCGClient


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

        saved_page = self.catalog.get_metadata("last_completed_page", "0")
        try:
            last_completed_page = int(saved_page or 0)
        except ValueError:
            last_completed_page = 0

        # Catalogs created before resume support do not have page metadata. Infer the
        # safest likely page from the number of complete 250-card batches already saved.
        inferred_page = existing_count // page_size
        last_completed_page = max(last_completed_page, inferred_page)
        page = last_completed_page + 1

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
                self.catalog.set_metadata("last_completed_page", str(page))

                current_count = self.catalog.card_count()
                self.progress.emit(current_count, total_count, page)

                if current_count >= total_count or len(cards) < page_size:
                    self.catalog.set_metadata("catalog_complete", "true")
                    break

                page += 1

            self.catalog.set_metadata(
                "last_updated",
                datetime.now(timezone.utc).isoformat(),
            )
            self.succeeded.emit(self.catalog.card_count())
        except Exception as exc:
            self.catalog.set_metadata("catalog_complete", "false")
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
