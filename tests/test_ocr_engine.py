from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


# These tests exercise the pure recognition logic without requiring a camera or
# OpenCV installation in the test runner.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from src.ocr_engine import CardOCREngine


class CardOCREngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CardOCREngine()

    def test_combines_separate_blk_and_collector_fraction_attempts(self) -> None:
        attempts = [
            ("BLK", 91.0, "set region"),
            ("067/086", 96.0, "number region"),
        ]

        code, collector, total, matched_text, confidence = (
            self.engine._parse_attempts(attempts)
        )

        self.assertEqual("BLK", code)
        self.assertEqual("67", collector)
        self.assertEqual(86, total)
        self.assertEqual("067/086", matched_text)
        self.assertEqual(91.0, confidence)

    def test_trims_one_trailing_symbol_digit_from_printed_total(self) -> None:
        code, collector, total, _text, _confidence = self.engine._parse_attempts(
            [("067/0865", 61.0, "number binary")]
        )

        self.assertEqual("", code)
        self.assertEqual("67", collector)
        self.assertEqual(86, total)

    def test_extracts_set_code_before_english_language_marker(self) -> None:
        code, _collector, _total, _text, _confidence = self.engine._parse_attempts(
            [("BLK en", 80.0, "set region")]
        )

        self.assertEqual("BLK", code)

    def test_posted_live_attempts_recover_blk_and_dense_number(self) -> None:
        attempts = [
            ("LDD", 13.0, "EasyOCR full strip"),
            ("BLKII067/1085", 3.0, "EasyOCR full strip"),
            ("BLKC", 43.0, "EasyOCR set binary"),
            ("1067/10885", 38.0, "EasyOCR number binary"),
            ("GAK", 0.0, "Tesseract set enhanced"),
            ("0677066", 0.0, "Tesseract number enhanced"),
            ("067066", 0.0, "Tesseract number binary"),
        ]

        code, collector, total, matched_text, confidence = (
            self.engine._parse_attempts(attempts)
        )

        self.assertEqual("BLK", code)
        self.assertEqual("67", collector)
        self.assertEqual(66, total)
        self.assertEqual("0677066", matched_text)
        self.assertEqual(0.0, confidence)

    def test_garbage_text_does_not_report_full_identifier_evidence(self) -> None:
        texts = ["H", "11B05015", "1", "G86"]

        self.assertLess(self.engine._text_evidence(texts), 40)

    def test_valid_separate_identifier_parts_report_full_evidence(self) -> None:
        self.assertEqual(100, self.engine._text_evidence(["BLK", "067/086"]))

    def test_identifier_regions_overlap_without_covering_whole_strip(self) -> None:
        set_bounds, number_bounds = self.engine._identifier_region_bounds(1000, 100)

        self.assertEqual((140, 18, 370, 82), set_bounds)
        self.assertEqual((320, 18, 760, 82), number_bounds)
        self.assertLess(number_bounds[0], set_bounds[2])

    def test_catalog_candidates_rank_blk_above_other_set_codes(self) -> None:
        cards = [
            {"name": "Other", "set_name": "Other", "set_code": "SW"},
            {"name": "Genesect ex", "set_name": "Black Bolt", "set_code": "BLK"},
            {"name": "Other", "set_name": "Other", "set_code": "G1"},
        ]

        ranked = self.engine.rank_catalog_candidates(cards, "BLN")

        self.assertEqual("BLK", ranked[0]["set_code"])

    def test_failed_easyocr_variant_does_not_abort_other_attempts(self) -> None:
        class BrokenReader:
            def readtext(self, *_args, **_kwargs):
                raise RuntimeError("empty detection crop")

        self.assertEqual(
            [],
            self.engine._easy_read(
                BrokenReader(), object(), self.engine.SET_ALLOWLIST
            ),
        )

    def test_tesseract_common_windows_location_is_considered(self) -> None:
        expected = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        with patch.dict("os.environ", {"ProgramFiles": r"C:\Program Files"}):
            self.assertEqual(expected, self.engine._tesseract_candidates()[0])


if __name__ == "__main__":
    unittest.main()
