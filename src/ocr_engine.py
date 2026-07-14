from __future__ import annotations

import re
import time
from dataclasses import dataclass
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
    COLLECTOR_FRACTION = re.compile(r"(?<!\d)(\d{1,3})\s*[/|\\]\s*(\d{1,3})(?!\d)")
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
        return binary

    @staticmethod
    def _easy_read(reader: Any, image: Any) -> list[tuple[str, float]]:
        if reader is None:
            return []
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/",
            text_threshold=0.25,
            low_text=0.20,
            link_threshold=0.20,
            mag_ratio=1.5,
        )
        attempts: list[tuple[str, float]] = []
        for item in results:
            if len(item) >= 3:
                text = str(item[1]).strip()
                score = float(item[2]) * 100.0
                if text:
                    attempts.append((text, score))
        return attempts

    @staticmethod
    def _tesseract_read(image: Any, psm: int) -> tuple[str, float]:
        if pytesseract is None or Output is None:
            return "", 0.0
        config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
        )
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
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
        self, attempts: list[tuple[str, float]]
    ) -> tuple[str, str, int | None, str, float]:
        best_code = ""
        best_collector = ""
        best_total: int | None = None
        best_text = ""
        best_confidence = 0.0

        for text, confidence in attempts:
            normalized = self._normalize(text)
            numeric = self._numeric_normalize(text)
            code_match = self.SET_CODE.search(normalized)
            fraction_match = self.COLLECTOR_FRACTION.search(numeric)
            code = code_match.group(1) if code_match else ""
            collector = ""
            total: int | None = None
            if fraction_match:
                collector_value = int(fraction_match.group(1))
                total_value = int(fraction_match.group(2))
                if 0 <= collector_value <= 999 and 1 <= total_value <= 999:
                    collector = str(collector_value)
                    total = total_value

            if code and collector:
                return code, collector, total, text, confidence
            if collector and (not best_collector or confidence > best_confidence):
                best_collector = collector
                best_total = total
                best_text = text
                best_confidence = confidence
            if code and not best_code:
                best_code = code
                if not best_text:
                    best_text = text
                    best_confidence = confidence

        return best_code, best_collector, best_total, best_text, best_confidence

    @staticmethod
    def _capture_quality(image: Any, sharpness: float) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        sharp_score = min(100.0, sharpness / 1.15)
        contrast_score = min(100.0, contrast * 2.2)
        return int(round((sharp_score * 0.62) + (contrast_score * 0.38)))

    @staticmethod
    def _text_evidence(texts: list[str]) -> int:
        joined = " ".join(texts).upper()
        letters = len(re.findall(r"[A-Z]", joined))
        digits = len(re.findall(r"\d", joined))
        slash = 20 if "/" in joined or "|" in joined or "\\" in joined else 0
        return min(100, letters * 6 + digits * 9 + slash)

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
        gray_identifier = cv2.cvtColor(identifier, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_identifier, cv2.CV_64F).var())
        capture_quality = self._capture_quality(identifier, sharpness)

        scale = 12.0 if enhanced else 10.0
        processed = self._prepare(identifier, scale, threshold=False)
        binary = self._prepare(identifier, scale, threshold=True)

        attempts: list[tuple[str, float]] = []
        backend_lines: list[str] = []
        reader = self._reader()
        if reader is not None:
            easy_attempts = self._easy_read(reader, processed)
            if enhanced:
                easy_attempts.extend(self._easy_read(reader, binary))
            attempts.extend(easy_attempts)
            backend_lines.append("Primary engine: EasyOCR")
        elif self._easy_error:
            backend_lines.append(f"EasyOCR unavailable: {self._easy_error}")

        parsed = self._parse_attempts(attempts)
        if not parsed[1]:
            for psm in (7, 13):
                text, confidence = self._tesseract_read(processed, psm)
                if text:
                    attempts.append((text, confidence))
            backend_lines.append("Fallback engine: Tesseract")
            parsed = self._parse_attempts(attempts)

        set_code, collector_number, printed_total, matched_text, matched_confidence = parsed
        texts = [text for text, _ in attempts if text]
        evidence = self._text_evidence(texts)

        if set_code and collector_number and printed_total:
            confidence = max(97, int(round(matched_confidence)))
        elif collector_number and printed_total:
            confidence = max(88, int(round(matched_confidence * 0.70 + capture_quality * 0.30)))
        elif set_code:
            confidence = max(55, int(round(matched_confidence * 0.60 + capture_quality * 0.40)))
        else:
            confidence = min(72, int(round(capture_quality * 0.40 + evidence * 0.40 + matched_confidence * 0.20)))

        elapsed = int(round((time.perf_counter() - started) * 1000))
        raw_lines = [
            f"Capture quality: {capture_quality}%",
            f"Identifier evidence: {evidence}%",
            "Recognition strategy: EasyOCR first, Tesseract fallback, catalog validation after OCR",
            *backend_lines,
            "OCR attempts:",
        ]
        raw_lines.extend(
            f"  {index + 1}: {text or '<nothing>'} ({score:.0f}%)"
            for index, (text, score) in enumerate(attempts)
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
