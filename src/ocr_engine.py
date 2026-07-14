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
    """Read only the bottom-left card identifier area and validate Pokémon formats."""

    CARD_ASPECT_RATIO = 0.714
    STANDARD_NUMBER = re.compile(r"\b([A-Z]{0,3}\d{1,3})\s*[/|]\s*(\d{1,3})\b", re.I)
    CODE_AND_NUMBER = re.compile(
        r"\b([A-Z]{2,6})\s*[- ]?\s*(\d{1,3})\s*[/|]\s*(\d{1,3})\b", re.I
    )
    PROMO_NUMBER = re.compile(
        r"\b(SVP|SWSH|SM|XY|BW|MEP)\s*[- ]?\s*(\d{1,3})\b", re.I
    )
    GALLERY_NUMBER = re.compile(r"\b(TG|GG|RC)\s*(\d{1,3})\b", re.I)

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
        """Crop the small region that contains set mark/code and collector number."""
        height, width = area.shape[:2]
        if mode == "identifier":
            # In close-up mode the user places the identifier area inside the guide.
            return area[int(height * 0.40) : height, 0 : int(width * 0.78)].copy()

        # Standard modern card layout: identifiers sit in the extreme lower-left.
        return area[
            int(height * 0.80) : int(height * 0.995),
            int(width * 0.00) : int(width * 0.72),
        ].copy()

    @staticmethod
    def _prepare_identifier(image: Any, scale: float = 8.0) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        denoised = cv2.bilateralFilter(clahe, 5, 35, 35)
        sharpened = cv2.addWeighted(
            denoised,
            2.1,
            cv2.GaussianBlur(denoised, (0, 0), 1.1),
            -1.1,
            0,
        )
        return sharpened

    @staticmethod
    def _read(image: Any, psm: int, whitelist: str = "") -> tuple[str, list[float]]:
        config = f"--oem 3 --psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
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
    def _normalize_common_ocr_errors(text: str) -> str:
        normalized = text.upper().replace("|", "/").replace("\\", "/")
        normalized = re.sub(r"\s+", " ", normalized)
        # Correct common mistakes only when surrounded by digits.
        normalized = re.sub(r"(?<=\d)[OQ](?=\d)", "0", normalized)
        normalized = re.sub(r"(?<=\d)[IL](?=\d)", "1", normalized)
        normalized = re.sub(r"(?<=\d)S(?=\d)", "5", normalized)
        normalized = re.sub(r"(?<=\d)B(?=\d)", "8", normalized)
        return normalized.strip()

    def _find_identifiers(self, texts: list[str]) -> tuple[str, str, int | None, str, int]:
        for text in texts:
            normalized = self._normalize_common_ocr_errors(text)

            match = self.CODE_AND_NUMBER.search(normalized)
            if match:
                return match.group(1), match.group(2), int(match.group(3)), text, 99

            match = self.PROMO_NUMBER.search(normalized)
            if match:
                return match.group(1), match.group(2), None, text, 98

            match = self.STANDARD_NUMBER.search(normalized)
            if match:
                collector = match.group(1)
                total = int(match.group(2))
                # A complete collector fraction is highly useful even without a readable set code.
                return "", collector, total, text, 96

            match = self.GALLERY_NUMBER.search(normalized)
            if match:
                return "", f"{match.group(1)}{match.group(2)}", None, text, 95

        return "", "", None, "", 0

    def read_card(self, frame: Any, mode: str = "full", enhanced: bool = False) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. Install Tesseract, then restart the app."
            )

        started = time.perf_counter()
        area = self.crop_guided_area(frame, mode)
        identifier_region = self._bottom_left_identifier(area, mode)
        gray_crop = cv2.cvtColor(identifier_region, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

        processed = self._prepare_identifier(identifier_region, 8.0 if not enhanced else 10.0)
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- "

        attempts = [
            self._read(processed, 7, whitelist),
            self._read(processed, 11, whitelist),
        ]
        if enhanced:
            _, otsu = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            attempts.extend(
                [
                    self._read(otsu, 6, whitelist),
                    self._read(otsu, 12, whitelist),
                    self._read(closed, 7, whitelist),
                    self._read(cv2.bitwise_not(closed), 11, whitelist),
                ]
            )

        texts = [text for text, _ in attempts]
        set_code, collector_number, printed_total, matched_text, parsed_confidence = self._find_identifiers(texts)

        if matched_text:
            identifier_text, identifier_conf = next(
                ((text, conf) for text, conf in attempts if text == matched_text),
                (matched_text, []),
            )
        else:
            identifier_text, identifier_conf = max(
                attempts,
                key=lambda item: (sum(item[1]) / len(item[1])) if item[1] else -1,
            )

        # Recognition confidence is based primarily on a valid Pokémon identifier,
        # not Tesseract's generic word confidence.
        if parsed_confidence:
            confidence = parsed_confidence
        else:
            scores = [score for score in identifier_conf if score >= 0]
            confidence = min(45, int(round(sum(scores) / len(scores)))) if scores else 0

        elapsed = int(round((time.perf_counter() - started) * 1000))
        return OCRResult(
            set_code=set_code,
            collector_number=collector_number,
            printed_total=printed_total,
            confidence=max(0, min(100, confidence)),
            raw_text="\n".join(f"Attempt {index + 1}: {text}" for index, (text, _) in enumerate(attempts)),
            processing_ms=elapsed,
            sharpness=sharpness,
            crop_image=identifier_region.copy(),
            processed_image=processed.copy(),
        )
