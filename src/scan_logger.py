from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


class ScanLogger:
    """Append lightweight scan outcomes for analysis without saving images."""

    EASTERN_TIME = ZoneInfo("America/New_York")
    FIELDS = (
        "timestamp_utc", "timestamp_eastern", "title_read", "set_read", "number_read",
        "regulation_read", "ocr_confidence", "title_confidence",
        "set_confidence", "regulation_confidence", "processing_ms",
        "capture_stability", "burst_frames", "candidate_count", "decision",
        "resolved_card_id", "resolved_name", "resolved_set", "resolved_number",
        "raw_ocr",
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def eastern_timestamp(cls, utc_text: str) -> str:
        try:
            value = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
        except ValueError:
            return ""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        eastern = value.astimezone(cls.EASTERN_TIME)
        return eastern.strftime("%Y-%m-%d %I:%M:%S %p %Z")

    def _upgrade_schema(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == list(self.FIELDS):
                return
            rows = list(reader)

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            writer.writeheader()
            for existing in rows:
                row = {field: existing.get(field, "") for field in self.FIELDS}
                if not row["timestamp_eastern"] and row["timestamp_utc"]:
                    row["timestamp_eastern"] = self.eastern_timestamp(
                        row["timestamp_utc"]
                    )
                writer.writerow(row)
        temporary.replace(self.path)

    def append(self, values: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._upgrade_schema()
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        row = {field: values.get(field, "") for field in self.FIELDS}
        row["timestamp_utc"] = values.get(
            "timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        row["timestamp_eastern"] = values.get(
            "timestamp_eastern", self.eastern_timestamp(str(row["timestamp_utc"]))
        )
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def visible_path(self) -> Path:
        """Return the Explorer-visible path when Windows Store Python redirects writes."""
        local_text = os.environ.get("LOCALAPPDATA", "")
        if not local_text:
            return self.path
        local = Path(local_text)
        try:
            relative = self.path.relative_to(local)
        except ValueError:
            return self.path
        package_root = local / "Packages"
        candidates = [
            root / "LocalCache" / "Local" / relative
            for root in package_root.glob("PythonSoftwareFoundation.Python.*")
        ]
        existing = [candidate for candidate in candidates if candidate.exists()]
        if existing:
            return max(existing, key=lambda candidate: candidate.stat().st_mtime)
        return self.path
