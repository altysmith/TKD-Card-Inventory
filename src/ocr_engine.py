from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log1p
from typing import Any

import cv2
import numpy as np

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover
    pytesseract = None
    Output = None


@dataclass
class OCRResult:
    card_name: str = ""
    card_name_confidence: float = 0.0
    set_code: str = ""
    set_code_confidence: float = 0.0
    regulation_mark: str = ""
    regulation_mark_confidence: float = 0.0
    collector_number: str = ""
    number_hints: tuple[tuple[str, float], ...] = ()
    printed_total: int | None = None
    confidence: int = 0
    raw_text: str = ""
    processing_ms: int = 0
    sharpness: float = 0.0
    crop_image: Any = None
    processed_image: Any = None


class CardOCREngine:
    """Read card identity clues and reconcile them with the local catalog."""

    CARD_ASPECT_RATIO = 0.714
    SET_CODE = re.compile(r"(?<![A-Z])([A-Z]{3})(?![A-Z])")
    COLLECTOR_FRACTION = re.compile(r"(?<!\d)(\d{1,3})\s*[/|\\]\s*(\d{1,4})(?!\d)")
    MIXED_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    SET_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUMBER_ALLOWLIST = "0123456789/"
    TITLE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-' ."
    REGULATION_ALLOWLIST = "DEFGHI"
    _easy_reader: Any = None
    _easy_error: str = ""

    def available(self) -> bool:
        return easyocr is not None or pytesseract is not None

    @classmethod
    def _reader(cls):
        if easyocr is None:
            return None
        if cls._easy_reader is None and not cls._easy_error:
            try:
                cls._easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as exc:  # pragma: no cover
                cls._easy_error = str(exc)
        return cls._easy_reader

    @classmethod
    def crop_guided_area(cls, frame: Any, mode: str) -> Any:
        height, width = frame.shape[:2]
        if mode == "identifier":
            guide_width = int(width * 0.88)
            guide_height = int(height * 0.38)
        else:
            guide_height = int(height * 0.94)
            guide_width = int(guide_height * cls.CARD_ASPECT_RATIO)
            if guide_width > int(width * 0.92):
                guide_width = int(width * 0.92)
                guide_height = int(guide_width / cls.CARD_ASPECT_RATIO)
        left = max(0, (width - guide_width) // 2)
        top = max(0, (height - guide_height) // 2)
        return frame[top : top + guide_height, left : left + guide_width].copy()

    @staticmethod
    def _order_corners(points: Any) -> Any:
        ordered = np.zeros((4, 2), dtype="float32")
        coordinate_sums = points.sum(axis=1)
        coordinate_differences = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(coordinate_sums)]
        ordered[2] = points[np.argmax(coordinate_sums)]
        ordered[1] = points[np.argmin(coordinate_differences)]
        ordered[3] = points[np.argmax(coordinate_differences)]
        return ordered

    @classmethod
    def normalize_card_area(cls, area: Any, mode: str) -> Any:
        """Straighten a full-card capture when a credible card outline is visible."""
        if mode == "identifier" or area is None or area.size == 0:
            return area

        height, width = area.shape[:2]
        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 135)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
        )
        contours, _hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = float(width * height)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < frame_area * 0.38:
                break
            perimeter = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(corners) != 4 or not cv2.isContourConvex(corners):
                continue

            ordered = cls._order_corners(corners.reshape(4, 2).astype("float32"))
            top_width = np.linalg.norm(ordered[1] - ordered[0])
            bottom_width = np.linalg.norm(ordered[2] - ordered[3])
            left_height = np.linalg.norm(ordered[3] - ordered[0])
            right_height = np.linalg.norm(ordered[2] - ordered[1])
            measured_width = max(top_width, bottom_width)
            measured_height = max(left_height, right_height)
            if measured_height <= 0:
                continue
            measured_ratio = measured_width / measured_height
            if not 0.58 <= measured_ratio <= 0.84:
                continue

            target_height = max(400, int(round(measured_height)))
            target_width = max(286, int(round(target_height * cls.CARD_ASPECT_RATIO)))
            destination = np.array(
                [
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1],
                ],
                dtype="float32",
            )
            transform = cv2.getPerspectiveTransform(ordered, destination)
            return cv2.warpPerspective(area, transform, (target_width, target_height))
        return area

    @staticmethod
    def _title_region_bounds(width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(width * 0.07),
            int(height * 0.018),
            max(1, int(width * 0.80)),
            max(1, int(height * 0.145)),
        )

    @classmethod
    def _title_region(cls, area: Any, mode: str) -> Any | None:
        if mode == "identifier":
            return None
        height, width = area.shape[:2]
        left, top, right, bottom = cls._title_region_bounds(width, height)
        return area[top:bottom, left:right].copy()

    @staticmethod
    def _identifier_strip(area: Any, mode: str) -> Any:
        height, width = area.shape[:2]
        if mode == "identifier":
            return area[
                int(height * 0.58) : int(height * 0.94),
                int(width * 0.02) : int(width * 0.62),
            ].copy()
        return area[
            int(height * 0.900) : int(height * 0.972),
            int(width * 0.035) : int(width * 0.47),
        ].copy()

    @staticmethod
    def _identifier_region_bounds(width: int, height: int) -> tuple[
        tuple[int, int, int, int], tuple[int, int, int, int]
    ]:
        """Return overlapping bounds for the set badge and collector fraction."""
        top = int(height * 0.18)
        bottom = max(top + 1, int(height * 0.82))
        set_bounds = (
            int(width * 0.14),
            top,
            max(1, int(width * 0.37)),
            bottom,
        )
        number_bounds = (
            int(width * 0.32),
            top,
            max(1, int(width * 0.76)),
            bottom,
        )
        return set_bounds, number_bounds

    @classmethod
    def _identifier_regions(cls, identifier: Any) -> tuple[Any, Any]:
        height, width = identifier.shape[:2]
        set_bounds, number_bounds = cls._identifier_region_bounds(width, height)
        set_left, set_top, set_right, set_bottom = set_bounds
        number_left, number_top, number_right, number_bottom = number_bounds
        return (
            identifier[set_top:set_bottom, set_left:set_right].copy(),
            identifier[number_top:number_bottom, number_left:number_right].copy(),
        )

    @staticmethod
    def _regulation_region(identifier: Any) -> Any:
        height, width = identifier.shape[:2]
        return identifier[
            int(height * 0.05) : int(height * 0.90),
            int(width * 0.02) : int(width * 0.18),
        ].copy()

    @staticmethod
    def _prepare(image: Any, scale: float, threshold: bool = False) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(enlarged)
        denoised = cv2.bilateralFilter(clahe, 5, 28, 28)
        sharpened = cv2.addWeighted(
            denoised,
            1.7,
            cv2.GaussianBlur(denoised, (0, 0), 0.9),
            -0.7,
            0,
        )
        if not threshold:
            return sharpened
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(binary.mean()) < 127.0:
            binary = cv2.bitwise_not(binary)
        return binary

    @classmethod
    def _prepare_adaptive(cls, image: Any, scale: float) -> Any:
        processed = cls._prepare(image, scale, threshold=False)
        binary = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        if float(binary.mean()) < 127.0:
            binary = cv2.bitwise_not(binary)
        return binary

    @staticmethod
    def _easy_read(
        reader: Any, image: Any, allowlist: str
    ) -> list[tuple[str, float]]:
        if reader is None:
            return []
        try:
            results = reader.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="beamsearch",
                allowlist=allowlist,
                text_threshold=0.25,
                low_text=0.20,
                link_threshold=0.20,
                mag_ratio=1.5,
            )
        except Exception:
            # Individual threshold/crop variants can occasionally produce an
            # empty EasyOCR detection box. Other variants and Tesseract should
            # still get a chance to identify the card.
            return []
        attempts: list[tuple[str, float]] = []
        for item in results:
            if len(item) >= 3:
                text = str(item[1]).strip()
                score = float(item[2]) * 100.0
                if text:
                    attempts.append((text, score))
        return attempts

    @staticmethod
    def _tesseract_candidates() -> tuple[str, str]:
        return (
            os.path.join(
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                "Tesseract-OCR",
                "tesseract.exe",
            ),
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Programs",
                "Tesseract-OCR",
                "tesseract.exe",
            ),
        )

    @staticmethod
    def _tesseract_read(
        image: Any, psm: int, allowlist: str
    ) -> tuple[str, float]:
        if pytesseract is None or Output is None:
            return "", 0.0
        if shutil.which("tesseract") is None:
            for candidate in CardOCREngine._tesseract_candidates():
                if os.path.isfile(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    break
        config = (
            f"--oem 3 --psm {psm} "
            f"-c tessedit_char_whitelist={allowlist}"
        )
        tesseract_image = image
        if float(image.mean()) < 127.0:
            tesseract_image = cv2.bitwise_not(image)
        data = pytesseract.image_to_data(
            tesseract_image, output_type=Output.DICT, config=config
        )
        words: list[str] = []
        scores: list[float] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", [])):
            clean = str(text).strip()
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = -1
            if clean and score >= 0:
                words.append(clean)
                scores.append(score)
        return " ".join(words), (sum(scores) / len(scores) if scores else 0.0)

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _normalize_name(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @classmethod
    def _best_title_attempt(
        cls, attempts: list[tuple[str, float, str]]
    ) -> tuple[str, float]:
        ignored = {"BASIC", "STAGE", "TRAINER", "ITEM", "HP"}
        candidates: list[tuple[float, str, float]] = []
        for text, confidence, _source in attempts:
            clean = re.sub(r"\s+", " ", text).strip(" .-'\t\r\n")
            normalized = cls._normalize_name(clean)
            if len(normalized) < 3 or normalized in ignored or normalized.isdigit():
                continue
            rank = float(confidence) + min(15, len(normalized)) * 1.5
            candidates.append((rank, clean, float(confidence)))
        if not candidates:
            return "", 0.0
        _rank, text, confidence = max(candidates, key=lambda item: item[0])
        return text, confidence

    @classmethod
    def _best_regulation_attempt(
        cls, attempts: list[tuple[str, float, str]]
    ) -> tuple[str, float]:
        candidates = [
            (text.strip().upper(), float(confidence))
            for text, confidence, _source in attempts
            if len(text.strip()) == 1
            and text.strip().upper() in cls.REGULATION_ALLOWLIST
        ]
        return max(candidates, key=lambda item: item[1]) if candidates else ("", 0.0)

    @staticmethod
    def _numeric_normalize(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("O", "0").replace("Q", "0")
        normalized = normalized.replace("I", "1").replace("L", "1")
        normalized = normalized.replace("S", "5").replace("B", "8")
        return normalized

    @classmethod
    def _extract_set_code(cls, text: str, source: str) -> str:
        normalized = cls._normalize(text)
        exact = cls.SET_CODE.search(normalized)
        if exact:
            return exact.group(1)

        # Targeted set OCR often joins the language marker or a neighboring
        # glyph to the code: BLK en -> BLKEN, or BLK plus a symbol -> BLKC.
        if "set" in source.casefold():
            letters = re.sub(r"[^A-Z]", "", normalized)
            if len(letters) >= 3:
                return letters[:3]
        return ""

    @classmethod
    def _extract_dense_fraction(
        cls, text: str, source: str
    ) -> tuple[str, int | None]:
        """Recover a 3+3 fraction when OCR loses the printed slash."""
        if "number" not in source.casefold():
            return "", None
        digits = re.sub(r"\D", "", cls._numeric_normalize(text))
        if len(digits) not in {6, 7}:
            return "", None
        collector_value = int(digits[:3])
        total_value = int(digits[-3:])
        if 0 <= collector_value <= 999 and 1 <= total_value <= 999:
            return str(collector_value), total_value
        return "", None

    @classmethod
    def _collector_candidate_score(
        cls,
        collector: str,
        total: int | None,
        text: str,
        confidence: float,
        source: str,
    ) -> float:
        """Prefer targeted, identifier-shaped reads over full-strip guesses."""
        score = float(confidence)
        if "number" in source.casefold():
            score += 100.0
        if "/" in cls._normalize(text):
            score += 10.0
        if total is not None and total >= 10:
            score += 10.0
        elif total is not None:
            score -= 30.0
        if collector.isdigit() and total and int(collector) > total * 3:
            score -= 20.0
        return score

    def _parse_attempts(
        self, attempts: list[tuple[str, float, str]]
    ) -> tuple[str, str, int | None, str, float]:
        best_code = ""
        best_code_confidence = -1.0
        best_collector = ""
        best_total: int | None = None
        best_collector_confidence = -1.0
        best_collector_selection_score = float("-inf")
        best_text = ""
        best_confidence = 0.0

        for text, confidence, source in attempts:
            normalized = self._normalize(text)
            numeric = self._numeric_normalize(text)
            fraction_match = self.COLLECTOR_FRACTION.search(numeric)
            code = self._extract_set_code(text, source)
            collector = ""
            total: int | None = None
            if fraction_match:
                collector_value = int(fraction_match.group(1))
                total_text = fraction_match.group(2)
                # A nearby rarity/symbol glyph is often appended as one extra
                # digit (for example 067/0865). Printed totals fit in 3 digits.
                if len(total_text) == 4:
                    total_text = total_text[:3]
                total_value = int(total_text)
                if 0 <= collector_value <= 999 and 1 <= total_value <= 999:
                    collector = str(collector_value)
                    total = total_value
            if not collector:
                collector, total = self._extract_dense_fraction(text, source)

            if code and confidence >= 5.0 and confidence > best_code_confidence:
                best_code = code
                best_code_confidence = confidence
            collector_selection_score = self._collector_candidate_score(
                collector, total, text, confidence, source
            )
            if collector and collector_selection_score > best_collector_selection_score:
                best_collector = collector
                best_total = total
                best_collector_confidence = confidence
                best_collector_selection_score = collector_selection_score
                best_text = text
                best_confidence = confidence

        if not best_text and best_code:
            best_text = best_code
            best_confidence = max(0.0, best_code_confidence)
        elif best_code and best_collector:
            best_confidence = min(best_code_confidence, best_collector_confidence)

        return best_code, best_collector, best_total, best_text, best_confidence

    @staticmethod
    def set_code_similarity(set_hint: str, set_code: str) -> float:
        hint = re.sub(r"[^A-Z0-9]", "", set_hint.upper())
        code = re.sub(r"[^A-Z0-9]", "", set_code.upper())
        if not hint or not code:
            return 0.0
        return SequenceMatcher(None, hint, code).ratio()

    @classmethod
    def name_similarity(cls, name_hint: str, card_name: str) -> float:
        hint = cls._normalize_name(name_hint)
        name = cls._normalize_name(card_name)
        if not hint or not name:
            return 0.0
        if hint == name:
            return 1.0
        return SequenceMatcher(None, hint, name).ratio()

    @classmethod
    def catalog_candidate_score(
        cls,
        card: dict[str, Any],
        set_hint: str = "",
        name_hint: str = "",
        collector_hint: str = "",
        printed_total: int | None = None,
    ) -> float:
        score = 0.0
        if set_hint:
            score += cls.set_code_similarity(
                set_hint, str(card.get("set_code", ""))
            ) * 25.0
        if name_hint:
            score += cls.name_similarity(name_hint, str(card.get("name", ""))) * 60.0
        if collector_hint:
            expected = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
            actual = re.sub(
                r"\D", "", str(card.get("raw_number", card.get("number", ""))).split("/", 1)[0]
            )
            score += 12.0 if expected and expected == actual else -2.0
        if printed_total is not None:
            score += 5.0 if card.get("printed_total") == printed_total else -1.0
        return score

    @classmethod
    def rank_catalog_candidates(
        cls,
        cards: list[dict[str, Any]],
        set_hint: str,
        name_hint: str = "",
        collector_hint: str = "",
        printed_total: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank catalog rows using every independent clue that OCR recovered."""
        return sorted(
            cards,
            key=lambda card: (
                -cls.catalog_candidate_score(
                    card,
                    set_hint=set_hint,
                    name_hint=name_hint,
                    collector_hint=collector_hint,
                    printed_total=printed_total,
                ),
                str(card.get("name", "")).casefold(),
                str(card.get("set_name", "")).casefold(),
            ),
        )

    @classmethod
    def decisive_title_match(
        cls, cards: list[dict[str, Any]], set_hint: str, name_hint: str
    ) -> bool:
        """Require a strong title match with separation from the runner-up."""
        if not cards or not set_hint or not name_hint:
            return False
        best_set = cls.set_code_similarity(set_hint, str(cards[0].get("set_code", "")))
        best_name = cls.name_similarity(name_hint, str(cards[0].get("name", "")))
        second_name = (
            cls.name_similarity(name_hint, str(cards[1].get("name", "")))
            if len(cards) > 1
            else 0.0
        )
        return best_set >= 0.80 and best_name >= 0.62 and best_name - second_name >= 0.10

    @classmethod
    def decisive_catalog_match(
        cls,
        cards: list[dict[str, Any]],
        set_hint: str,
        name_hint: str,
        collector_hint: str = "",
    ) -> bool:
        if cls.decisive_title_match(cards, set_hint, name_hint):
            return True
        if not cards or not set_hint or not name_hint or not collector_hint:
            return False
        expected = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
        best_actual = re.sub(
            r"\D",
            "",
            str(cards[0].get("raw_number", cards[0].get("number", ""))).split("/", 1)[0],
        )
        second_actual = (
            re.sub(
                r"\D",
                "",
                str(cards[1].get("raw_number", cards[1].get("number", ""))).split("/", 1)[0],
            )
            if len(cards) > 1
            else ""
        )
        return (
            cls.set_code_similarity(set_hint, str(cards[0].get("set_code", ""))) >= 0.80
            and cls.name_similarity(name_hint, str(cards[0].get("name", ""))) >= 0.45
            and bool(expected)
            and best_actual == expected
            and second_actual != expected
        )

    @classmethod
    def narrow_exact_name_candidates(
        cls,
        cards: list[dict[str, Any]],
        name_hint: str,
        set_hint: str = "",
        collector_hint: str = "",
        printed_total: int | None = None,
        trust_set_hint: bool = True,
        regulation_mark: str = "",
    ) -> tuple[list[dict[str, Any]], bool]:
        """Let an exact title establish identity before optional identifier clues."""
        candidates = [
            card
            for card in cards
            if cls.name_similarity(name_hint, str(card.get("name", ""))) == 1.0
        ]
        if not candidates:
            return [], False

        used_set_hint = False
        if set_hint and trust_set_hint:
            set_matches = [
                card
                for card in candidates
                if cls.set_code_similarity(
                    set_hint, str(card.get("set_code", ""))
                )
                == 1.0
            ]
            if set_matches:
                candidates = set_matches
                used_set_hint = True

        if collector_hint:
            expected = re.sub(r"\D", "", collector_hint.split("/", 1)[0])
            number_matches = [
                card
                for card in candidates
                if re.sub(
                    r"\D",
                    "",
                    str(card.get("raw_number", card.get("number", ""))).split("/", 1)[0],
                )
                == expected
            ]
            if number_matches:
                candidates = number_matches

        if regulation_mark:
            regulation_matches = [
                card
                for card in candidates
                if str(card.get("regulation_mark", "")).upper()
                == regulation_mark.upper()
            ]
            if regulation_matches:
                candidates = regulation_matches

        if printed_total is not None:
            total_matches = [
                card
                for card in candidates
                if card.get("printed_total") == printed_total
            ]
            if total_matches:
                candidates = total_matches

        return candidates, used_set_hint

    @classmethod
    def number_fragment_score(
        cls, card: dict[str, Any], hints: tuple[tuple[str, float], ...]
    ) -> float:
        raw_number = re.sub(
            r"\D", "", str(card.get("raw_number", card.get("number", ""))).split("/", 1)[0]
        )
        printed_total = re.sub(r"\D", "", str(card.get("printed_total") or ""))
        variants: set[str] = set()
        if raw_number:
            variants.update({raw_number, raw_number.zfill(3)})
        if printed_total:
            variants.update({printed_total, printed_total.zfill(3)})
        if raw_number and printed_total:
            variants.update(
                {
                    raw_number + printed_total,
                    raw_number.zfill(3) + printed_total.zfill(3),
                }
            )
        evidence: list[float] = []
        for hint, confidence in hints:
            digits = re.sub(r"\D", "", hint)
            if len(digits) < 3 or not variants:
                continue
            similarity = max(
                SequenceMatcher(None, digits, variant).ratio()
                for variant in variants
            )
            length_weight = min(1.0, len(digits) / 4.0)
            confidence_weight = 0.65 + min(100.0, confidence) / 285.0
            evidence.append(similarity * length_weight * confidence_weight)
        return sum(sorted(evidence, reverse=True)[:2])

    @classmethod
    def rank_number_fragment_candidates(
        cls,
        cards: list[dict[str, Any]],
        hints: tuple[tuple[str, float], ...],
    ) -> list[dict[str, Any]]:
        return sorted(
            cards,
            key=lambda card: (
                -cls.number_fragment_score(card, hints),
                str(card.get("set_name", "")).casefold(),
                str(card.get("number", "")),
            ),
        )

    @classmethod
    def decisive_number_fragment_match(
        cls,
        cards: list[dict[str, Any]],
        hints: tuple[tuple[str, float], ...],
    ) -> bool:
        if not cards or len(cards) > 5:
            return False
        best = cls.number_fragment_score(cards[0], hints)
        second = cls.number_fragment_score(cards[1], hints) if len(cards) > 1 else 0.0
        return best >= 0.50 and best - second >= 0.20

    @classmethod
    def capture_frame_score(cls, frame: Any, mode: str) -> float:
        """Prefer readable identifier/title regions and penalize clipped glare."""
        area = cls.crop_guided_area(frame, mode)
        regions = [cls._identifier_strip(area, mode)]
        title = cls._title_region(area, mode)
        if title is not None and title.size:
            regions.append(title)
        scores: list[float] = []
        for region in regions:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast = float(gray.std())
            clipped = float(np.count_nonzero(gray >= 250)) / float(gray.size)
            scores.append(log1p(sharpness) * 25.0 + contrast - clipped * 420.0)
        return sum(scores) / len(scores)

    @staticmethod
    def _capture_quality(image: Any, sharpness: float) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        sharp_score = min(100.0, sharpness / 1.15)
        contrast_score = min(100.0, contrast * 2.2)
        return int(round((sharp_score * 0.62) + (contrast_score * 0.38)))

    @classmethod
    def _text_evidence(cls, texts: list[str]) -> int:
        """Score identifier-shaped evidence, not the volume of arbitrary OCR text."""
        normalized = [cls._normalize(text) for text in texts]
        numeric = [cls._numeric_normalize(text) for text in texts]
        has_code = any(cls.SET_CODE.search(text) for text in normalized)
        has_fraction = any(cls.COLLECTOR_FRACTION.search(text) for text in numeric)
        if has_code and has_fraction:
            return 100
        if has_fraction:
            return 65
        if has_code:
            return 35

        most_digits = max(
            (len(re.findall(r"\d", text)) for text in numeric),
            default=0,
        )
        has_separator = any(
            "/" in text or "|" in text or "\\" in text for text in texts
        )
        return min(30, most_digits * 3 + (6 if has_separator else 0))

    @staticmethod
    def _border(image: Any) -> Any:
        value = 255 if len(image.shape) == 2 else (0, 255, 0)
        return cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=value)

    def read_card(
        self, frame: Any, mode: str = "full", enhanced: bool = False
    ) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "No OCR engine is installed. Run pip install -r requirements.txt "
                "and restart the app."
            )

        started = time.perf_counter()
        guided_area = self.crop_guided_area(frame, mode)
        area = self.normalize_card_area(guided_area, mode)
        perspective_corrected = area is not guided_area
        identifier = self._identifier_strip(area, mode)
        title_region = self._title_region(area, mode)
        set_region, number_region = self._identifier_regions(identifier)
        regulation_region = self._regulation_region(identifier)
        gray_identifier = cv2.cvtColor(identifier, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_identifier, cv2.CV_64F).var())
        capture_quality = self._capture_quality(identifier, sharpness)

        scale = 12.0 if enhanced else 10.0
        target_scale = 10.0 if enhanced else 8.0
        processed = self._prepare(identifier, scale, threshold=False)
        set_processed = self._prepare(set_region, target_scale, threshold=False)
        set_binary = self._prepare(set_region, target_scale, threshold=True)
        number_processed = self._prepare(number_region, target_scale, threshold=False)
        number_binary = self._prepare(number_region, target_scale, threshold=True)
        regulation_binary = self._prepare(
            regulation_region, target_scale, threshold=True
        )
        title_processed = (
            self._prepare(title_region, 4.0 if enhanced else 3.0, threshold=False)
            if title_region is not None and title_region.size
            else None
        )
        title_binary = (
            self._prepare(title_region, 4.0 if enhanced else 3.0, threshold=True)
            if title_region is not None and title_region.size
            else None
        )

        attempts: list[tuple[str, float, str]] = []
        title_attempts: list[tuple[str, float, str]] = []
        regulation_attempts: list[tuple[str, float, str]] = []
        backend_lines: list[str] = []

        def add_attempts(source: str, found: list[tuple[str, float]]) -> None:
            attempts.extend((text, score, source) for text, score in found)

        reader = self._reader()
        if reader is not None:
            add_attempts(
                "EasyOCR full strip",
                self._easy_read(reader, processed, self.MIXED_ALLOWLIST),
            )
            add_attempts(
                "EasyOCR set binary",
                self._easy_read(reader, set_binary, self.SET_ALLOWLIST),
            )
            add_attempts(
                "EasyOCR number binary",
                self._easy_read(reader, number_binary, self.NUMBER_ALLOWLIST),
            )
            regulation_attempts.extend(
                (text, score, "EasyOCR regulation binary")
                for text, score in self._easy_read(
                    reader, regulation_binary, self.REGULATION_ALLOWLIST
                )
            )
            if title_processed is not None and title_binary is not None:
                title_attempts.extend(
                    (text, score, "EasyOCR title enhanced")
                    for text, score in self._easy_read(
                        reader, title_processed, self.TITLE_ALLOWLIST
                    )
                )
                title_attempts.extend(
                    (text, score, "EasyOCR title binary")
                    for text, score in self._easy_read(
                        reader, title_binary, self.TITLE_ALLOWLIST
                    )
                )
            if enhanced:
                add_attempts(
                    "EasyOCR set adaptive",
                    self._easy_read(
                        reader,
                        self._prepare_adaptive(set_region, target_scale),
                        self.SET_ALLOWLIST,
                    ),
                )
                add_attempts(
                    "EasyOCR number adaptive",
                    self._easy_read(
                        reader,
                        self._prepare_adaptive(number_region, target_scale),
                        self.NUMBER_ALLOWLIST,
                    ),
                )
            backend_lines.append(
                "Primary engine: EasyOCR (title, regulation, full, set, and number regions)"
            )
        elif self._easy_error:
            backend_lines.append(f"EasyOCR unavailable: {self._easy_error}")

        if pytesseract is not None and Output is not None:
            tesseract_passes = [
                ("Tesseract set enhanced", set_processed, 8, self.SET_ALLOWLIST),
                ("Tesseract set binary", set_binary, 8, self.SET_ALLOWLIST),
                (
                    "Tesseract number enhanced",
                    number_processed,
                    8,
                    self.NUMBER_ALLOWLIST,
                ),
                (
                    "Tesseract number binary",
                    number_binary,
                    8,
                    self.NUMBER_ALLOWLIST,
                ),
            ]
            if enhanced:
                tesseract_passes.extend(
                    [
                        (
                            "Tesseract set adaptive",
                            self._prepare_adaptive(set_region, target_scale),
                            13,
                            self.SET_ALLOWLIST,
                        ),
                        (
                            "Tesseract number adaptive",
                            self._prepare_adaptive(number_region, target_scale),
                            13,
                            self.NUMBER_ALLOWLIST,
                        ),
                    ]
                )
            for source, image, psm, allowlist in tesseract_passes:
                text, confidence = self._tesseract_read(image, psm, allowlist)
                if text:
                    attempts.append((text, confidence, source))
            backend_lines.append("Secondary engine: Tesseract (targeted set and number regions)")

        parsed = self._parse_attempts(attempts)

        set_code, collector_number, printed_total, matched_text, matched_confidence = parsed
        set_code_confidence = (
            max(
                (
                    score
                    for text, score, source in attempts
                    if self._extract_set_code(text, source) == set_code
                ),
                default=0.0,
            )
            if set_code
            else 0.0
        )
        card_name, card_name_confidence = self._best_title_attempt(title_attempts)
        regulation_mark, regulation_mark_confidence = self._best_regulation_attempt(
            regulation_attempts
        )
        number_hints = tuple(
            (re.sub(r"\D", "", self._numeric_normalize(text)), score)
            for text, score, source in attempts
            if "number" in source.casefold()
            and len(re.sub(r"\D", "", self._numeric_normalize(text))) >= 3
        )
        texts = [text for text, _score, _source in attempts if text]
        evidence = self._text_evidence(texts)

        if set_code and collector_number and printed_total:
            confidence = int(
                round(
                    matched_confidence * 0.60
                    + capture_quality * 0.20
                    + evidence * 0.20
                )
            )
        elif collector_number and printed_total:
            confidence = int(
                round(
                    matched_confidence * 0.65
                    + capture_quality * 0.25
                    + evidence * 0.10
                )
            )
        elif set_code:
            confidence = max(
                55,
                int(round(matched_confidence * 0.60 + capture_quality * 0.40)),
            )
        else:
            confidence = min(
                72,
                int(
                    round(
                        capture_quality * 0.40
                        + evidence * 0.40
                        + matched_confidence * 0.20
                    )
                ),
            )

        elapsed = int(round((time.perf_counter() - started) * 1000))
        raw_lines = [
            f"Capture quality: {capture_quality}%",
            f"Identifier evidence: {evidence}%",
            f"Card alignment: {'perspective corrected' if perspective_corrected else 'fixed guide'}",
            "Recognition strategy: title, regulation, set-code, and collector-number OCR, "
            "then catalog resolution",
            *backend_lines,
            "OCR attempts:",
        ]
        raw_lines.extend(
            f"  {index + 1}: [{source}] {text or '<nothing>'} ({score:.0f}%)"
            for index, (text, score, source) in enumerate(attempts)
        )
        raw_lines.extend(
            f"  {len(attempts) + index + 1}: [{source}] {text or '<nothing>'} ({score:.0f}%)"
            for index, (text, score, source) in enumerate(title_attempts)
        )
        raw_lines.extend(
            f"  {len(attempts) + len(title_attempts) + index + 1}: "
            f"[{source}] {text or '<nothing>'} ({score:.0f}%)"
            for index, (text, score, source) in enumerate(regulation_attempts)
        )
        if not attempts and not title_attempts and not regulation_attempts:
            raw_lines.append("  <nothing>")

        return OCRResult(
            card_name=card_name,
            card_name_confidence=card_name_confidence,
            set_code=set_code,
            set_code_confidence=set_code_confidence,
            regulation_mark=regulation_mark,
            regulation_mark_confidence=regulation_mark_confidence,
            collector_number=collector_number,
            number_hints=number_hints,
            printed_total=printed_total,
            confidence=max(0, min(100, confidence)),
            raw_text="\n".join(raw_lines),
            processing_ms=elapsed,
            sharpness=sharpness,
            crop_image=self._border(identifier),
            processed_image=self._border(processed),
        )
