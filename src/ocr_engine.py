from __future__ import annotations

import re
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
    confidence: int = 0
    raw_text: str = ""


class CardOCREngine:
    """Read only the narrow card regions that contain useful identifiers."""

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
        height, width = frame.shape[:2]
        guide_height = int(height * 0.82)
        guide_width = min(width, int(guide_height * 0.714))
        left = max(0, (width - guide_width) // 2)
        top = max(0, (height - guide_height) // 2)
        return frame[top : top + guide_height, left : left + guide_width].copy()

    @staticmethod
    def _variants(image: Any, scale: float) -> list[Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        sharpened = cv2.addWeighted(
            normalized, 1.8, cv2.GaussianBlur(normalized, (0, 0), 2), -0.8, 0
        )
        adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9,
        )
        _, otsu = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return [sharpened, adaptive, otsu]

    @staticmethod
    def _read(image: Any, psm: int) -> tuple[str, list[float]]:
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config=f"--oem 3 --psm {psm}",
        )
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
        text = re.sub(
            r"\b(?:BASIC|STAGE\s*[12]|TRAINER|ENERGY|HP\s*\d+)\b",
            "", text, flags=re.I,
        )
        return re.sub(r"\s+", " ", text).strip(" -")

    def _find_number(self, texts: list[str]) -> tuple[str, str, str]:
        for text in texts:
            normalized = text.replace("|", "/")
            for pattern in self.NUMBER_PATTERNS:
                match = pattern.search(normalized)
                if not match:
                    continue
                groups = match.groups()
                if len(groups) == 1:
                    return "", groups[0].upper(), text
                if groups[0].upper() in {"SVP", "SWSH", "SM", "XY", "BW"}:
                    return groups[0].upper(), groups[1], text
                return "", "".join(groups).upper(), text
        return "", "", ""

    def read_card(self, frame: Any) -> OCRResult:
        if not self.available():
            raise RuntimeError(
                "Tesseract OCR is not installed or cannot be found. Install Tesseract, then restart the app."
            )

        card = self.crop_guided_card(frame)
        height, width = card.shape[:2]
        top_region = card[0 : max(1, int(height * 0.14)), 0:width]
        bottom_region = card[int(height * 0.82) : int(height * 0.98), 0:width]

        top_candidates: list[tuple[str, list[float]]] = []
        for variant in self._variants(top_region, 3.4):
            top_candidates.append(self._read(variant, 7))
            top_candidates.append(self._read(variant, 11))

        bottom_candidates: list[tuple[str, list[float]]] = []
        for variant in self._variants(bottom_region, 4.2):
            bottom_candidates.append(self._read(variant, 7))
            bottom_candidates.append(self._read(variant, 11))

        top_text, top_conf = max(
            top_candidates,
            key=lambda item: (sum(item[1]) / len(item[1])) if item[1] else -1,
        )
        bottom_texts = [text for text, _ in bottom_candidates]
        set_code, collector_number, matched_text = self._find_number(bottom_texts)

        if matched_text:
            bottom_text, bottom_conf = next(
                ((text, conf) for text, conf in bottom_candidates if text == matched_text),
                (matched_text, []),
            )
        else:
            bottom_text, bottom_conf = max(
                bottom_candidates,
                key=lambda item: (sum(item[1]) / len(item[1])) if item[1] else -1,
            )

        card_name = self._clean_name(top_text)
        if len(card_name.split()) > 4:
            card_name = " ".join(card_name.split()[:4])

        all_conf = [score for score in top_conf + bottom_conf if score >= 0]
        confidence = int(round(sum(all_conf) / len(all_conf))) if all_conf else 0

        return OCRResult(
            card_name=card_name,
            set_code=set_code,
            collector_number=collector_number,
            confidence=max(0, min(100, confidence)),
            raw_text=f"{top_text}\n{bottom_text}".strip(),
        )
