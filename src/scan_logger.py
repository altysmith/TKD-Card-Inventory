from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScanLogger:
    """Append lightweight scan outcomes for analysis without saving images."""

    FIELDS = (
        "timestamp_utc", "title_read", "set_read", "number_read",
        "regulation_read", "ocr_confidence", "title_confidence",
        "set_confidence", "regulation_confidence", "processing_ms",
        "capture_stability", "burst_frames", "candidate_count", "decision",
        "resolved_card_id", "resolved_name", "resolved_set", "resolved_number",
        "raw_ocr",
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, values: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        row = {field: values.get(field, "") for field in self.FIELDS}
        row["timestamp_utc"] = values.get(
            "timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
