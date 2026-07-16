from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal, Slot

from .catalog import PokemonCatalog
from .pokemon_api import PokemonTCGClient


RICH_SCHEMA_VERSION = "2"
CATALOG_PAGE_SIZE = 250
CATALOG_MAX_ATTEMPTS = 3
CATALOG_RETRY_BASE_SECONDS = 2

logger = logging.getLogger(__name__)


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
        page_size = CATALOG_PAGE_SIZE
        existing_count = self.catalog.card_count()
        rich_refresh = self.catalog.get_metadata("rich_schema_version", "0") != RICH_SCHEMA_VERSION

        if rich_refresh:
            try:
                completed = int(self.catalog.get_metadata("rich_refresh_page", "0") or 0)
            except ValueError:
                completed = 0
            page = completed + 1
            processed = completed * page_size
        else:
            try:
                completed = int(self.catalog.get_metadata("last_completed_page", "0") or 0)
            except ValueError:
                completed = 0
            completed = max(completed, existing_count // page_size)
            page = completed + 1
            processed = existing_count

        try:
            total_count = 0
            while True:
                cards = None
                last_error: Exception | None = None
                for attempt in range(1, CATALOG_MAX_ATTEMPTS + 1):
                    try:
                        cards, total_count = self.client.fetch_card_page(page, page_size)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < CATALOG_MAX_ATTEMPTS:
                            delay = attempt * CATALOG_RETRY_BASE_SECONDS
                            logger.warning(
                                "Catalog page %s attempt %s/%s failed; retrying in %ss: %s",
                                page,
                                attempt,
                                CATALOG_MAX_ATTEMPTS,
                                delay,
                                exc,
                            )
                            time.sleep(delay)
                        else:
                            logger.error(
                                "Catalog page %s attempt %s/%s failed: %s",
                                page,
                                attempt,
                                CATALOG_MAX_ATTEMPTS,
                                exc,
                            )

                if cards is None:
                    raise RuntimeError(
                        f"Catalog page {page} failed after {CATALOG_MAX_ATTEMPTS} attempts. "
                        f"Progress through page {page - 1} was saved. {last_error}"
                    )
                if not cards:
                    break

                self.catalog.replace_page(cards)
                processed = min(total_count, processed + len(cards)) if rich_refresh else self.catalog.card_count()
                metadata_key = "rich_refresh_page" if rich_refresh else "last_completed_page"
                self.catalog.set_metadata(metadata_key, str(page))
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
