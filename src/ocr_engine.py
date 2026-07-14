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
        """Keep the modern printed set-code and collector-number line."""
        height, width = area.shape[:2]
        if mode == "identifier":
            return area[int(height * 0.42) : int(height * 0.98), 0 : int(width * 0.82)].copy()
        return area[
            int(height * 0.82) : int(height * 0.985),
            0 : int(width * 0.68),
        ].copy()

    @staticmethod
    def _split_identifier(identifier: Any) -> tuple[Any, Any]:
        """Create overlapping crops because card layouts vary slightly by expansion."""
        height, width = identifier.shape[:2]
        # The printed three-letter code is at the far left. Keep some overlap so a
        # slightly shifted card still retains all three letters.
        code_crop = identifier[0:height, 0 : max(1, int(width * 0.46))].copy()
        # The collector fraction sits immediately to the right of the code.
        number_crop = identifier[
            0:height,
            int(width * 0.18) : max(int(width * 0.19), int(width * 0.98)),
        ].copy()
        return code_crop, number_crop

    @staticmethod
    def _prepare(image: Any, scale: float, threshold: bool = False) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(enlarged)
        denoised = cv2.bilateralFilter(clahe, 5, 35, 35)
        sharpened = cv2.addWeighted(
            denoised,
            2.0,
            cv2.GaussianBlur(denoised, (0, 0), 1.1),
            -1.0,
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
        # Only accept the exact modern three-letter layout. Do not invent a code
        # from longer words or unrelated copyright text.
        return normalized if len(normalized) == 3 else ""

    @staticmethod
    def _normalize_number_text(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        # Corrections are intentionally conservative and only applied in a numeric string.
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
    def _debug_canvas(left: Any, right: Any) -> Any:
        """Return a side-by-side image without writing temporary images to disk."""
        target_height = max(left.shape[0], right.shape[0])

        def pad(image: Any) -> Any:
            if image.shape[0] == target_height:
                return image
            bottom = target_height - image.shape[0]
            return cv2.copyMakeBorder(image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

        return np.hstack((pad(left), pad(right)))

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

        scale = 10.0 if enhanced else 8.0
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

        code_conf = 0
        for text, confidences in code_attempts:
            if text == matched_code_text:
                code_conf = self._average_confidence(confidences)
                break
        number_conf = 0
        for text, confidences in number_attempts:
            if text == matched_number_text:
                number_conf = self._average_confidence(confidences)
                break

        # A complete modern identifier is the strongest result. A collector fraction
        # alone is still useful for narrowing the local catalog, but should require review.
        if set_code and collector_number and printed_total:
            confidence = 99
        elif collector_number and printed_total:
            confidence = 88
        elif set_code:
            confidence = 55
        else:
            confidence = min(40, max(code_conf, number_conf))

        elapsed = int(round((time.perf_counter() - started) * 1000))
        raw_lines = ["Set-code attempts:"]
        raw_lines.extend(f"  {index + 1}: {text or '<nothing>'}" for index, (text, _) in enumerate(code_attempts))
        raw_lines.append("Collector-number attempts:")
        raw_lines.extend(
            f"  {index + 1}: {text or '<nothing>'}" for index, (text, _) in enumerate(number_attempts)
        )

        return OCRResult(
            set_code=set_code,
            collector_number=collector_number,
            printed_total=printed_total,
            confidence=confidence,
            raw_text="\n".join(raw_lines),
            processing_ms=elapsed,
            sharpness=sharpness,
            crop_image=self._debug_canvas(code_crop, number_crop),
            processed_image=self._debug_canvas(code_processed, number_processed),
        )
