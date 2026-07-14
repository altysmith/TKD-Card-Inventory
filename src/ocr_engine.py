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
    """Fast targeted OCR with temporary debug images returned to the UI."""

    CARD_ASPECT_RATIO = 0.714
    STANDARD_NUMBER = re.compile(r"\b([A-Z]{0,3}\d{1,3})\s*[/|]\s*(\d{1,3})\b", re.I)
    CODE_AND_NUMBER = re.compile(
        r"\b([A-Z]{2,6})\s*[- ]?\s*(\d{1,3})\s*[/|]\s*(\d{1,3})\b", re.I
    )
    PROMO_NUMBER = re.compile(r"\b(SVP|SWSH|SM|XY|BW|MEP)\s*[- ]?\s*(\d{1,3})\b", re.I)
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
    def _prepare_identifier(image: Any, scale: float) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        return cv2.addWeighted(clahe, 1.7, cv2.GaussianBlur(clahe, (0, 0), 1.2), -0.7, 0)

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
    def _clean_name(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9 .:'\-éÉ♀♂]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\b(?:BASIC|STAGE\s*[12]|TRAINER|ENERGY|HP\s*\d+)\b", "", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip(" -")

    def _find_identifiers(self, texts: list[str]) -> tuple[str, str, int | None, str]:
        for text in texts:
            normalized = text.upper().replace("|", "/")
            match = self.CODE_AND_NUMBER.search(normalized)
            if match:
                return match.group(1), match.group(2), int(match.group(3)), text
            match = self.PROMO_NUMBER.search(normalized)
            if match:
                return match.group(1), match.group(2), None, text
            match = self.STANDARD_NUMBER.search(normalized)
            if match:
                return "", match.group(1), int(match.group(2)), text
            match = self.GALLERY_NUMBER.search(normalized)
            if match:
                return "", f"{match.group(1)}{match.group(2)}", None, text
        return "", "", None, ""

    def read_card(self, frame: Any, mode: str = "full", enhanced: bool = False) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. Install Tesseract, then restart the app."
            )

        started = time.perf_counter()
        area = self.crop_guided_area(frame, mode)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(area, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        height, width = area.shape[:2]

        if mode == "identifier":
            identifier_region = area
            title_region = None
        else:
            title_region = area[0 : max(1, int(height * 0.16)), 0:width]
            identifier_region = area[int(height * 0.64) : int(height * 0.99), 0:width]

        scale = 4.5 if mode == "identifier" else 4.0
        processed = self._prepare_identifier(identifier_region, scale)
        whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/- "

        attempts = [self._read(processed, 7, whitelist), self._read(processed, 11, whitelist)]
        if enhanced:
            _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            attempts.extend([self._read(binary, 6, whitelist), self._read(binary, 12, whitelist)])

        texts = [text for text, _ in attempts]
        set_code, collector_number, printed_total, matched_text = self._find_identifiers(texts)
        if matched_text:
            identifier_text, identifier_conf = next(
                ((text, conf) for text, conf in attempts if text == matched_text), (matched_text, [])
            )
        else:
            identifier_text, identifier_conf = max(
                attempts, key=lambda item: (sum(item[1]) / len(item[1])) if item[1] else -1
            )

        card_name = ""
        title_text = ""
        title_conf: list[float] = []
        if title_region is not None:
            title_processed = self._prepare_identifier(title_region, 3.0)
            title_text, title_conf = self._read(title_processed, 7)
            card_name = self._clean_name(title_text)
            if len(card_name.split()) > 4:
                card_name = " ".join(card_name.split()[:4])

        scores = [score for score in identifier_conf if score >= 0]
        if not scores:
            scores = [score for score in title_conf if score >= 0]
        confidence = int(round(sum(scores) / len(scores))) if scores else 0
        elapsed = int(round((time.perf_counter() - started) * 1000))

        return OCRResult(
            card_name=card_name,
            set_code=set_code,
            collector_number=collector_number,
            printed_total=printed_total,
            confidence=max(0, min(100, confidence)),
            raw_text=f"Identifier: {identifier_text}\nTitle: {title_text}".strip(),
            processing_ms=elapsed,
            sharpness=sharpness,
            crop_image=identifier_region.copy(),
            processed_image=processed.copy(),
        )
