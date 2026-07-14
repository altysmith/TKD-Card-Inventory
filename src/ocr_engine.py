from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover - handled at runtime for user setup
    pytesseract = None
    Output = None


@dataclass
class OCRResult:
    card_name: str = ""
    set_code: str = ""
    collector_number: str = ""
    confidence: int = 0
    raw_text: str = ""


class CardOCREngine:
    """Extract likely card name and collector information from a guided card crop."""

    NUMBER_PATTERNS = (
        re.compile(r"\b([A-Z]{0,3}\d{1,3})\s*/\s*\d{1,3}\b", re.IGNORECASE),
        re.compile(r"\b(SVP|SWSH|SM|XY|BW)\s*[- ]?\s*(\d{1,3})\b", re.IGNORECASE),
        re.compile(r"\b(TG|GG|RC)(\d{1,3})\b", re.IGNORECASE),
    )

    def available(self) -> bool:
        if pytesseract is None:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def crop_guided_card(frame: Any) -> Any:
        """Crop the same centered 0.714-ratio card area drawn by the UI guide."""
        height, width = frame.shape[:2]
        guide_height = int(height * 0.82)
        guide_width = int(guide_height * 0.714)
        guide_width = min(guide_width, width)
        left = max(0, (width - guide_width) // 2)
        top = max(0, (height - guide_height) // 2)
        return frame[top : top + guide_height, left : left + guide_width].copy()

    @staticmethod
    def _prepare(image: Any, scale: float = 2.5) -> Any:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )

    @staticmethod
    def _clean_name(text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9 .:'\-éÉ♀♂]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\b(?:BASIC|STAGE\s*[12]|TRAINER|ENERGY|HP\s*\d+)\b", "", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip(" -")

    def read_card(self, frame: Any) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. Install Tesseract, then restart the app."
            )

        card = self.crop_guided_card(frame)
        height, width = card.shape[:2]
        top_region = card[0 : max(1, int(height * 0.22)), 0:width]
        bottom_region = card[int(height * 0.76) : height, 0:width]

        top_prepared = self._prepare(top_region, 3.0)
        bottom_prepared = self._prepare(bottom_region, 3.5)

        top_data = pytesseract.image_to_data(
            top_prepared,
            output_type=Output.DICT,
            config="--oem 3 --psm 6",
        )
        bottom_data = pytesseract.image_to_data(
            bottom_prepared,
            output_type=Output.DICT,
            config="--oem 3 --psm 6",
        )

        def words_and_conf(data: dict[str, Any]) -> tuple[list[str], list[float]]:
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
            return words, confidences

        top_words, top_conf = words_and_conf(top_data)
        bottom_words, bottom_conf = words_and_conf(bottom_data)
        top_text = " ".join(top_words)
        bottom_text = " ".join(bottom_words)
        raw_text = f"{top_text}\n{bottom_text}".strip()

        card_name = self._clean_name(top_text)
        if len(card_name.split()) > 6:
            card_name = " ".join(card_name.split()[:6])

        set_code = ""
        collector_number = ""
        for pattern in self.NUMBER_PATTERNS:
            match = pattern.search(bottom_text.replace("|", "/"))
            if not match:
                continue
            groups = match.groups()
            if len(groups) == 1:
                collector_number = groups[0].upper()
            elif groups[0].upper() in {"SVP", "SWSH", "SM", "XY", "BW"}:
                set_code = groups[0].upper()
                collector_number = groups[1]
            else:
                collector_number = "".join(groups).upper()
            break

        all_conf = [score for score in top_conf + bottom_conf if score >= 0]
        confidence = int(round(sum(all_conf) / len(all_conf))) if all_conf else 0

        return OCRResult(
            card_name=card_name,
            set_code=set_code,
            collector_number=collector_number,
            confidence=max(0, min(100, confidence)),
            raw_text=raw_text,
        )
