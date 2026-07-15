from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import cv2

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
    set_code: str = ""
    collector_number: str = ""
    printed_total: int | None = None
    confidence: int = 0
    raw_text: str = ""
    processing_ms: int = 0
    sharpness: float = 0.0
    crop_image: Any = None
    processed_image: Any = None


class CardOCREngine:
    """Read a modern card's printed set code and collector fraction."""

    CARD_ASPECT_RATIO = 0.714
    SET_CODE = re.compile(r"(?<![A-Z])([A-Z]{3})(?![A-Z])")
    COLLECTOR_FRACTION = re.compile(r"(?<!\d)(\d{1,3})\s*[/|\\]\s*(\d{1,4})(?!\d)")
    MIXED_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    SET_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUMBER_ALLOWLIST = "0123456789/"
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
    def _numeric_normalize(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("O", "0").replace("Q", "0")
        normalized = normalized.replace("I", "1").replace("L", "1")
        normalized = normalized.replace("S", "5").replace("B", "8")
        return normalized

    def _parse_attempts(
        self, attempts: list[tuple[str, float, str]]
    ) -> tuple[str, str, int | None, str, float]:
        best_code = ""
        best_code_confidence = -1.0
        best_collector = ""
        best_total: int | None = None
        best_collector_confidence = -1.0
        best_text = ""
        best_confidence = 0.0

        for text, confidence, _source in attempts:
            normalized = self._normalize(text)
            numeric = self._numeric_normalize(text)
            code_match = self.SET_CODE.search(normalized)
            fraction_match = self.COLLECTOR_FRACTION.search(numeric)
            code = code_match.group(1) if code_match else ""
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

            if code and confidence >= 5.0 and confidence > best_code_confidence:
                best_code = code
                best_code_confidence = confidence
            if collector and confidence > best_collector_confidence:
                best_collector = collector
                best_total = total
                best_collector_confidence = confidence
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
    def rank_catalog_candidates(
        cls, cards: list[dict[str, Any]], set_hint: str
    ) -> list[dict[str, Any]]:
        """Put catalog set codes closest to the OCR hint first."""
        return sorted(
            cards,
            key=lambda card: (
                -cls.set_code_similarity(set_hint, str(card.get("set_code", ""))),
                str(card.get("name", "")).casefold(),
                str(card.get("set_name", "")).casefold(),
            ),
        )

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

    def read_card(self, frame: Any, mode: str = "full", enhanced: bool = False) -> OCRResult:
        if not self.available():
            raise RuntimeError("No OCR engine is installed. Run pip install -r requirements.txt and restart the app.")

        started = time.perf_counter()
        area = self.crop_guided_area(frame, mode)
        identifier = self._identifier_strip(area, mode)
        set_region, number_region = self._identifier_regions(identifier)
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

        attempts: list[tuple[str, float, str]] = []
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
            backend_lines.append("Primary engine: EasyOCR (full, set, and number regions)")
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
            confidence = max(55, int(round(matched_confidence * 0.60 + capture_quality * 0.40)))
        else:
            confidence = min(72, int(round(capture_quality * 0.40 + evidence * 0.40 + matched_confidence * 0.20)))

        elapsed = int(round((time.perf_counter() - started) * 1000))
        raw_lines = [
            f"Capture quality: {capture_quality}%",
            f"Identifier evidence: {evidence}%",
            "Recognition strategy: separate set-code and collector-number OCR, then catalog lookup",
            *backend_lines,
            "OCR attempts:",
        ]
        raw_lines.extend(
            f"  {index + 1}: [{source}] {text or '<nothing>'} ({score:.0f}%)"
            for index, (text, score, source) in enumerate(attempts)
        )
        if not attempts:
            raw_lines.append("  <nothing>")

        return OCRResult(
            set_code=set_code,
            collector_number=collector_number,
            printed_total=printed_total,
            confidence=max(0, min(100, confidence)),
            raw_text="\n".join(raw_lines),
            processing_ms=elapsed,
            sharpness=sharpness,
            crop_image=self._border(identifier),
            processed_image=self._border(processed),
        )
