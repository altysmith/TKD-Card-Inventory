import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.scan_logger import ScanLogger


class ScanLoggerTests(unittest.TestCase):
    def test_appends_header_and_scan_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan_history.csv"
            logger = ScanLogger(path)
            logger.append({"title_read": "Slowpoke", "decision": "review"})
            logger.append({"title_read": "Pikachu", "decision": "automatic"})
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(2, len(rows))
            self.assertEqual("Slowpoke", rows[0]["title_read"])
            self.assertEqual("automatic", rows[1]["decision"])

    def test_visible_path_finds_windows_store_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)
            intended = local / "Team KD" / "TKD Card Inventory" / "scan_history.csv"
            redirected = (
                local / "Packages" / "PythonSoftwareFoundation.Python.3.12_test"
                / "LocalCache" / "Local" / "Team KD" / "TKD Card Inventory"
                / "scan_history.csv"
            )
            redirected.parent.mkdir(parents=True)
            redirected.write_text("header\n", encoding="utf-8")
            with patch.dict("os.environ", {"LOCALAPPDATA": str(local)}):
                self.assertEqual(redirected, ScanLogger(intended).visible_path())


if __name__ == "__main__":
    unittest.main()
