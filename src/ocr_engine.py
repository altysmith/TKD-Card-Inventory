from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

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
    """Recognize modern Pokémon cards from the printed set code and collector fraction."""

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
    def _bottom_left_identifier(area: Any, mode: str) -> Any:
        """Keep only the modern printed set-code and collector-number baseline."""
        height, width = area.shape[:2]
        if mode == "identifier":
            return area[
                int(height * 0.52) : int(height * 0.96),
                int(width * 0.02) : int(width * 0.72),
            ].copy()

        # Tightened from the previous lower-third crop. This intentionally excludes
        # most copyright/rules text and keeps the set code plus collector fraction.
        return area[
            int(height * 0.875) : int(height * 0.978),
            int(width * 0.035) : int(width * 0.58),
        ].copy()

    @staticmethod
    def _split_identifier(identifier: Any) -> tuple[Any, Any]:
        """Create slightly overlapping crops for the code and fraction."""
        height, width = identifier.shape[:2]
        code_crop = identifier[
            int(height * 0.05) : int(height * 0.95),
            0 : max(1, int(width * 0.42)),
        ].copy()
        number_crop = identifier[
            int(height * 0.05) : int(height * 0.95),
            int(width * 0.25) : max(int(width * 0.26), int(width * 0.98)),
        ].copy()
        return code_crop, number_crop

    @staticmethod
    def _prepare(image: Any, scale: float, threshold: bool = False) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8)).apply(enlarged)
        denoised = cv2.bilateralFilter(clahe, 5, 30, 30)
        sharpened = cv2.addWeighted(
            denoised,
            1.85,
            cv2.GaussianBlur(denoised, (0, 0), 1.0),
            -0.85,
            0,
        )
        if not threshold:
            return sharpened
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _read(image: Any, psm: int, whitelist: str) -> tuple[str, list[float]]:
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
    def _normalize_code(text: str) -> str:
        normalized = re.sub(r"[^A-Z]", "", text.upper())
        return normalized if len(normalized) == 3 else ""

    @staticmethod
    def _normalize_number_text(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("O", "0").replace("Q", "0")
        normalized = normalized.replace("I", "1").replace("L", "1")
        normalized = normalized.replace("S", "5").replace("B", "8")
        return normalized

    def _extract_code(self, attempts: list[tuple[str, list[float]]]) -> tuple[str, str]:
        for text, _ in attempts:
            direct = self.SET_CODE.search(text.upper())
            if direct:
                return direct.group(1), text
            normalized = self._normalize_code(text)
            if normalized:
                return normalized, text
        return "", ""

    def _extract_fraction(
        self, attempts: list[tuple[str, list[float]]]
    ) -> tuple[str, int | None, str]:
        for text, _ in attempts:
            normalized = self._normalize_number_text(text)
            match = self.COLLECTOR_FRACTION.search(normalized)
            if match:
                collector = str(int(match.group(1)))
                total = int(match.group(2))
                if 1 <= total <= 999 and 0 <= int(collector) <= 999:
                    return collector, total, text
        return "", None, ""

    @staticmethod
    def _average_confidence(confidences: list[float]) -> int:
        valid = [score for score in confidences if score >= 0]
        return int(round(sum(valid) / len(valid))) if valid else 0

    @staticmethod
    def _text_evidence(texts: list[str], kind: str) -> int:
        joined = " ".join(texts).upper()
        if kind == "code":
            letters = len(re.findall(r"[A-Z]", joined))
            return min(100, letters * 22)
        digits = len(re.findall(r"\d", joined))
        slash_bonus = 25 if "/" in joined or "|" in joined or "\\" in joined else 0
        return min(100, digits * 10 + slash_bonus)

    @staticmethod
    def _capture_quality(image: Any, sharpness: float) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        sharp_score = min(100.0, sharpness / 1.25)
        contrast_score = min(100.0, contrast * 2.1)
        return int(round((sharp_score * 0.62) + (contrast_score * 0.38)))

    @staticmethod
    def _debug_canvas(left: Any, right: Any) -> Any:
        """Return labeled-looking bordered crops without writing files to disk."""
        left_bordered = cv2.copyMakeBorder(left, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        right_bordered = cv2.copyMakeBorder(right, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 255, 255))
        target_height = max(left_bordered.shape[0], right_bordered.shape[0])

        def pad(image: Any) -> Any:
            if image.shape[0] == target_height:
                return image
            bottom = target_height - image.shape[0]
            value = 0 if len(image.shape) == 2 else (0, 0, 0)
            return cv2.copyMakeBorder(image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=value)

        gap_shape = (target_height, 18) if len(left_bordered.shape) == 2 else (target_height, 18, 3)
        gap = np.zeros(gap_shape, dtype=left_bordered.dtype)
        return np.hstack((pad(left_bordered), gap, pad(right_bordered)))

    def read_card(self, frame: Any, mode: str = "full", enhanced: bool = False) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. Install Tesseract, then restart the app."
            )

        started = time.perf_counter()
        area = self.crop_guided_area(frame, mode)
        identifier = self._bottom_left_identifier(area, mode)
        code_crop, number_crop = self._split_identifier(identifier)

        gray_identifier = cv2.cvtColor(identifier, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_identifier, cv2.CV_64F).var())
        capture_quality = self._capture_quality(identifier, sharpness)

        scale = 11.0 if enhanced else 9.0
        code_processed = self._prepare(code_crop, scale, threshold=enhanced)
        number_processed = self._prepare(number_crop, scale, threshold=enhanced)

        code_attempts = [
            self._read(code_processed, 7, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            self._read(code_processed, 13, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ]
        number_attempts = [
            self._read(number_processed, 7, "0123456789/"),
            self._read(number_processed, 13, "0123456789/"),
        ]
        if enhanced:
            code_attempts.append(self._read(cv2.bitwise_not(code_processed), 11, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            number_attempts.append(self._read(cv2.bitwise_not(number_processed), 11, "0123456789/"))

        set_code, matched_code_text = self._extract_code(code_attempts)
        collector_number, printed_total, matched_number_text = self._extract_fraction(number_attempts)

        code_tesseract = 0
        for text, confidences in code_attempts:
            if text == matched_code_text:
                code_tesseract = self._average_confidence(confidences)
                break
        number_tesseract = 0
        for text, confidences in number_attempts:
            if text == matched_number_text:
                number_tesseract = self._average_confidence(confidences)
                break

        code_texts = [text for text, _ in code_attempts if text]
        number_texts = [text for text, _ in number_attempts if text]
        code_evidence = self._text_evidence(code_texts, "code")
        number_evidence = self._text_evidence(number_texts, "number")

        if set_code:
            code_score = max(88, code_tesseract)
        else:
            code_score = int(round((code_tesseract * 0.45) + (code_evidence * 0.55)))

        if collector_number and printed_total:
            number_score = max(94, number_tesseract)
        else:
            number_score = int(round((number_tesseract * 0.4) + (number_evidence * 0.6)))

        # Graded recognition does not collapse to zero merely because strict parsing failed.
        if set_code and collector_number and printed_total:
            confidence = 99
        elif collector_number and printed_total:
            confidence = max(82, int(round(number_score * 0.88 + code_score * 0.12)))
        elif set_code:
            confidence = max(58, int(round(code_score * 0.78 + number_score * 0.22)))
        else:
            confidence = int(round(
                capture_quality * 0.28 + code_score * 0.30 + number_score * 0.42
            ))
            confidence = min(74, confidence)

        elapsed = int(round((time.perf_counter() - started) * 1000))
        raw_lines = [
            f"Capture quality: {capture_quality}%",
            f"Set-code evidence: {code_score}%",
            f"Collector-number evidence: {number_score}%",
            "Crop borders: blue = set code, yellow = collector fraction",
            "Set-code attempts:",
        ]
        raw_lines.extend(f"  {index + 1}: {text or '<nothing>'}" for index, (text, _) in enumerate(code_attempts))
        raw_lines.append("Collector-number attempts:")
        raw_lines.extend(
            f"  {index + 1}: {text or '<nothing>'}" for index, (text, _) in enumerate(number_attempts)
        )

        return OCRResult(
            set_code=set_code,
            collector_number=collector_number,
            printed_total=printed_total,
            confidence=max(0, min(100, confidence)),
            raw_text="\n".join(raw_lines),
            processing_ms=elapsed,
            sharpness=sharpness,
            crop_image=self._debug_canvas(code_crop, number_crop),
            processed_image=self._debug_canvas(code_processed, number_processed),
        )
