from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .pokemon_api import PokemonTCGClient


class SearchWorker(QObject):
    succeeded = Signal(list)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, client: PokemonTCGClient, name: str, number: str) -> None:
        super().__init__()
        self.client = client
        self.name = name
        self.number = number

    @Slot()
    def run(self) -> None:
        try:
            results = self.client.search_cards(self.name, self.number)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(results)
        finally:
            self.finished.emit()
