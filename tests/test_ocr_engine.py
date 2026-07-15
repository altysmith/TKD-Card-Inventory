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

    def test_targeted_number_beats_implausible_full_strip_fraction(self) -> None:
        attempts = [
            ("0XBK057/0051K", 8.0, "EasyOCR full strip"),
            ("BLK", 50.0, "EasyOCR set binary"),
            ("06700865", 9.0, "EasyOCR number binary"),
            ("067/066", 0.0, "Tesseract number enhanced"),
        ]

        code, collector, total, matched_text, confidence = (
            self.engine._parse_attempts(attempts)
        )

        self.assertEqual("BLK", code)
        self.assertEqual("67", collector)
        self.assertEqual(66, total)
        self.assertEqual("067/066", matched_text)
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

    def test_damaged_title_resolves_slowpoke_inside_scr(self) -> None:
        cards = [
            {
                "name": "Alcremie",
                "set_name": "Stellar Crown",
                "set_code": "SCR",
                "raw_number": "65",
                "number": "65/142",
                "printed_total": 142,
            },
            {
                "name": "Slowpoke",
                "set_name": "Stellar Crown",
                "set_code": "SCR",
                "raw_number": "57",
                "number": "57/142",
                "printed_total": 142,
            },
            {
                "name": "Sparkling Crystal",
                "set_name": "Stellar Crown",
                "set_code": "SCR",
                "raw_number": "142",
                "number": "142/142",
                "printed_total": 142,
            },
        ]

        ranked = self.engine.rank_catalog_candidates(
            cards, "SCR", name_hint="SWPKE"
        )

        self.assertEqual("Slowpoke", ranked[0]["name"])
        self.assertTrue(
            self.engine.decisive_title_match(ranked, "SCR", "SWPKE")
        )

    def test_ambiguous_title_is_not_declared_decisive(self) -> None:
        cards = [
            {"name": "Mew ex", "set_code": "MEW", "number": "151/165"},
            {"name": "Mew ex", "set_code": "MEW", "number": "193/165"},
        ]
        ranked = self.engine.rank_catalog_candidates(
            cards, "MEW", name_hint="MEWEX"
        )

        self.assertFalse(self.engine.decisive_title_match(ranked, "MEW", "MEWEX"))

    def test_collector_number_breaks_a_duplicate_title_tie(self) -> None:
        cards = [
            {
                "name": "Mew ex",
                "set_code": "MEW",
                "raw_number": "151",
                "number": "151/165",
            },
            {
                "name": "Mew ex",
                "set_code": "MEW",
                "raw_number": "193",
                "number": "193/165",
            },
        ]
        ranked = self.engine.rank_catalog_candidates(
            cards, "MEW", name_hint="MEWEX", collector_hint="193"
        )

        self.assertEqual("193", ranked[0]["raw_number"])
        self.assertTrue(
            self.engine.decisive_catalog_match(
                ranked, "MEW", "MEWEX", collector_hint="193"
            )
        )

    def test_exact_title_ignores_conflicting_set_and_keeps_same_name(self) -> None:
        cards = [
            {
                "name": "Slowpoke",
                "set_code": "SCR",
                "raw_number": "57",
                "number": "57/142",
                "printed_total": 142,
            },
            {
                "name": "Slowpoke",
                "set_code": "MEW",
                "raw_number": "79",
                "number": "79/165",
                "printed_total": 165,
            },
            {
                "name": "Spoink",
                "set_code": "FCO",
                "raw_number": "30",
                "number": "30/124",
                "printed_total": 124,
            },
        ]

        narrowed, used_set = self.engine.narrow_exact_name_candidates(
            cards,
            "Slowpoke",
            set_hint="FCO",
            trust_set_hint=False,
        )

        self.assertFalse(used_set)
        self.assertEqual(["Slowpoke", "Slowpoke"], [card["name"] for card in narrowed])
        self.assertNotIn("Spoink", [card["name"] for card in narrowed])

    def test_exact_title_uses_matching_set_as_secondary_evidence(self) -> None:
        cards = [
            {
                "name": "Slowpoke",
                "set_code": "SCR",
                "raw_number": "57",
                "number": "57/142",
                "printed_total": 142,
            },
            {
                "name": "Slowpoke",
                "set_code": "MEW",
                "raw_number": "79",
                "number": "79/165",
                "printed_total": 165,
            },
        ]

        narrowed, used_set = self.engine.narrow_exact_name_candidates(
            cards,
            "Slowpoke",
            set_hint="SCR",
            trust_set_hint=True,
        )

        self.assertTrue(used_set)
        self.assertEqual("57/142", narrowed[0]["number"])
        self.assertEqual(1, len(narrowed))

    def test_regulation_mark_narrows_exact_title_candidates(self) -> None:
        cards = [
            {
                "name": "Slowpoke",
                "set_code": "SCR",
                "raw_number": "57",
                "number": "57/142",
                "printed_total": 142,
                "regulation_mark": "H",
            },
            {
                "name": "Slowpoke",
                "set_code": "PRE",
                "raw_number": "18",
                "number": "18/131",
                "printed_total": 131,
                "regulation_mark": "H",
            },
            {
                "name": "Slowpoke",
                "set_code": "MEW",
                "raw_number": "79",
                "number": "79/165",
                "printed_total": 165,
                "regulation_mark": "G",
            },
        ]

        narrowed, _used_set = self.engine.narrow_exact_name_candidates(
            cards,
            "Slowpoke",
            regulation_mark="H",
            trust_set_hint=False,
        )

        self.assertEqual(["SCR", "PRE"], [card["set_code"] for card in narrowed])

    def test_posted_number_fragments_resolve_scr_after_regulation_filter(self) -> None:
        cards = [
            {
                "name": "Slowpoke",
                "set_name": "Prismatic Evolutions",
                "set_code": "PRE",
                "raw_number": "18",
                "number": "18/131",
                "printed_total": 131,
                "regulation_mark": "H",
            },
            {
                "name": "Slowpoke",
                "set_name": "Stellar Crown",
                "set_code": "SCR",
                "raw_number": "57",
                "number": "57/142",
                "printed_total": 142,
                "regulation_mark": "H",
            },
        ]
        number_hints = (("0372", 36.0), ("664", 16.0))

        ranked = self.engine.rank_number_fragment_candidates(cards, number_hints)

        self.assertEqual("SCR", ranked[0]["set_code"])
        self.assertTrue(
            self.engine.decisive_number_fragment_match(ranked, number_hints)
        )

    def test_regulation_picker_uses_strongest_single_letter(self) -> None:
        mark, confidence = self.engine._best_regulation_attempt(
            [
                ("I", 12.0, "regulation"),
                ("H", 99.8, "regulation"),
                ("SCR", 90.0, "regulation"),
            ]
        )

        self.assertEqual("H", mark)
        self.assertEqual(99.8, confidence)

    def test_title_picker_ignores_layout_labels(self) -> None:
        title, confidence = self.engine._best_title_attempt(
            [
                ("BASIC", 99.0, "title"),
                ("SWPKE", 92.0, "title"),
                ("HP", 98.0, "title"),
            ]
        )

        self.assertEqual("SWPKE", title)
        self.assertEqual(92.0, confidence)

    def test_title_region_avoids_right_side_hp_area(self) -> None:
        self.assertEqual((70, 25, 800, 203), self.engine._title_region_bounds(1000, 1400))

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
