"""Per-run output folder: every run records its config, prompts, scores and usage."""

import json
from datetime import datetime
from pathlib import Path

from hebqa.config import RUNS_DIR


class RunLog:
    def __init__(self, name: str, root: Path = RUNS_DIR):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = root / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=False)

    def write_json(self, filename: str, obj) -> Path:
        path = self.dir / filename
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_jsonl(self, filename: str, rows: list[dict]) -> Path:
        path = self.dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
