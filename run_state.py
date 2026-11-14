"""Checkpoint chỉ chứa hash và số frame; không lưu credentials/ảnh/annotation."""

import hashlib
import json
from pathlib import Path


STATE_FILE = Path(__file__).with_name(".annotation_state.json")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class RunState:
    def __init__(self, context, path=STATE_FILE):
        self.path = Path(path)
        self.key = fingerprint(context)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if not isinstance(self.data, dict):
            raise ValueError("Checkpoint không hợp lệ; kiểm tra file state.")
        self.completed = self.data.get(self.key, {})

    def next_action(self):
        for name, limit, apply in (("dry_1", 1, False), ("dry_5", 5, False),
                                   ("apply_1", 1, True), ("apply_5", 5, True),
                                   ("dry_all", None, False), ("apply_all", None, True)):
            if not self.completed.get(name):
                return limit, apply
        return None

    def next_dry_action(self):
        for name, limit in (("dry_1", 1), ("dry_5", 5), ("dry_all", None)):
            if not self.completed.get(name):
                return limit, False
        return None

    def require(self, limit, apply):
        stage = str(limit) if limit is not None else "all"
        required = []
        if limit == 5:
            required.append("dry_1")
        elif limit is None:
            required.extend(("dry_1", "dry_5"))
        if apply:
            required.append("dry_" + stage)
            if limit == 5:
                required.append("apply_1")
            elif limit is None:
                required.extend(("apply_1", "apply_5"))
        missing = [step for step in required if not self.completed.get(step)]
        if missing:
            raise ValueError("Cần hoàn thành trước: " + ", ".join(missing) + ". Bắt đầu python main.py --limit 1.")

    def record(self, limit, apply, processed):
        stage = str(limit) if limit is not None else "all"
        self.completed[("apply_" if apply else "dry_") + stage] = processed
        self.data[self.key] = self.completed
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)
