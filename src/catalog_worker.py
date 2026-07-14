from __future__ import annotations

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
        imported = 0
        page = 1
        page_size = 250
        try:
            while True:
                cards, total_count = self.client.fetch_card_page(page, page_size)
                if not cards:
                    break

                imported += self.catalog.replace_page(cards)
                self.progress.emit(imported, total_count, page)

                if imported >= total_count or len(cards) < page_size:
                    break
                page += 1

            self.catalog.set_metadata(
                "last_updated",
                datetime.now(timezone.utc).isoformat(),
            )
            self.succeeded.emit(imported)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
