from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import cv2

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
    """Recognize a modern card from one focused printed identifier line."""

    CARD_ASPECT_RATIO = 0.714
    SET_CODE = re.compile(r"(?<![A-Z])([A-Z]{3})(?![A-Z])")
    COLLECTOR_FRACTION = re.compile(r"(?<!\d)(\d{1,3})\s*[/|\\]\s*(\d{1,3})(?!\d)")

    def available(self) -> bool:
        if pytesseract is None:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

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
        """Crop one short line containing both the printed set code and fraction."""
        height, width = area.shape[:2]
        if mode == "identifier":
            return area[
                int(height * 0.58) : int(height * 0.94),
                int(width * 0.02) : int(width * 0.62),
            ].copy()

        # Modern English cards place the three-letter acronym and fraction on one
        # line in the extreme lower-left. Keep only that line and avoid the nearby
        # copyright and illustrator text.
        return area[
            int(height * 0.900) : int(height * 0.972),
            int(width * 0.035) : int(width * 0.47),
        ].copy()

    @staticmethod
    def _prepare(image: Any, scale: float, threshold: bool = False) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(enlarged)
        denoised = cv2.bilateralFilter(clahe, 5, 30, 30)
        sharpened = cv2.addWeighted(
            denoised,
            1.9,
            cv2.GaussianBlur(denoised, (0, 0), 1.0),
            -0.9,
            0,
        )
        if not threshold:
            return sharpened
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _read(image: Any, psm: int) -> tuple[str, list[float]]:
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ "
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
        words: list[str] = []
        confidences: list[float] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", [])):
            clean = str(text).strip()
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = -1
            if clean and score >= 0:
                words.append(clean)
                confidences.append(score)
        return " ".join(words), confidences

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _numeric_normalize(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("O", "0").replace("Q", "0")
        normalized = normalized.replace("I", "1").replace("L", "1")
        normalized = normalized.replace("S", "5").replace("B", "8")
        return normalized

    def _parse_attempts(
        self, attempts: list[tuple[str, list[float]]]
    ) -> tuple[str, str, int | None, str]:
        best_code = ""
        best_fraction = ""
        best_total: int | None = None
        best_text = ""

        for text, _ in attempts:
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
                return code, collector, total, text
            if collector and not best_fraction:
                best_fraction = collector
                best_total = total
                best_text = text
            if code and not best_code:
                best_code = code
                if not best_text:
                    best_text = text

        return best_code, best_fraction, best_total, best_text

    @staticmethod
    def _average_confidence(confidences: list[float]) -> int:
        valid = [score for score in confidences if score >= 0]
        return int(round(sum(valid) / len(valid))) if valid else 0

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
        slash = 18 if "/" in joined or "|" in joined or "\\" in joined else 0
        return min(100, letters * 7 + digits * 8 + slash)

    @staticmethod
    def _border(image: Any) -> Any:
        value = 255 if len(image.shape) == 2 else (0, 255, 0)
        return cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=value)

    def read_card(self, frame: Any, mode: str = "full", enhanced: bool = False) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. Install Tesseract, then restart the app."
            )

        started = time.perf_counter()
        area = self.crop_guided_area(frame, mode)
        identifier = self._identifier_strip(area, mode)
        gray_identifier = cv2.cvtColor(identifier, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_identifier, cv2.CV_64F).var())
        capture_quality = self._capture_quality(identifier, sharpness)

        scale = 12.0 if enhanced else 10.0
        processed = self._prepare(identifier, scale, threshold=enhanced)
        attempts = [self._read(processed, 7), self._read(processed, 13)]
        if enhanced:
            attempts.extend(
                [
                    self._read(cv2.bitwise_not(processed), 7),
                    self._read(cv2.bitwise_not(processed), 11),
                ]
            )

        set_code, collector_number, printed_total, matched_text = self._parse_attempts(attempts)
        matched_confidence = 0
        for text, confidences in attempts:
            if text == matched_text:
                matched_confidence = self._average_confidence(confidences)
                break

        texts = [text for text, _ in attempts if text]
        evidence = self._text_evidence(texts)

        if set_code and collector_number and printed_total:
            confidence = max(96, matched_confidence)
        elif collector_number and printed_total:
            confidence = max(80, int(round(matched_confidence * 0.65 + capture_quality * 0.35)))
        elif set_code:
            confidence = max(55, int(round(matched_confidence * 0.60 + capture_quality * 0.40)))
        else:
            confidence = int(round(capture_quality * 0.42 + evidence * 0.38 + matched_confidence * 0.20))
            confidence = min(72, confidence)

        elapsed = int(round((time.perf_counter() - started) * 1000))
        raw_lines = [
            f"Capture quality: {capture_quality}%",
            f"Identifier evidence: {evidence}%",
            "Single crop: green border = set code + collector fraction",
            "OCR attempts:",
        ]
        raw_lines.extend(
            f"  {index + 1}: {text or '<nothing>'}" for index, (text, _) in enumerate(attempts)
        )

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
