import json
import os
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional


class AccessLogger:
    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "access.log",
        max_bytes: int = 2 * 1024 * 1024,
        backup_count: int = 5,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_file)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = Lock()

    def write_event(
        self,
        event_type: str,
        person_name: Optional[str] = None,
        similarity: Optional[float] = None,
        result: Optional[str] = None,
        state_from: Optional[str] = None,
        state_to: Optional[str] = None,
        door_result: Optional[str] = None,
        failure_reason: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": timestamp,
            "time": timestamp,
            "log_type": "access",
            "event_type": event_type,
            "person_name": person_name,
            "similarity": similarity,
            "result": result,
            "state_from": state_from,
            "state_to": state_to,
            "door_result": door_result,
            "failure_reason": failure_reason,
            "source": source,
            "extra": extra or {},
        }
        record = {key: value for key, value in record.items() if value is not None}

        with self._lock:
            self._rotate_if_needed()
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _rotate_if_needed(self) -> None:
        if self.max_bytes <= 0 or self.backup_count <= 0:
            return

        if not os.path.exists(self.file_path):
            return

        if os.path.getsize(self.file_path) < self.max_bytes:
            return

        oldest = f"{self.file_path}.{self.backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)

        for index in range(self.backup_count - 1, 0, -1):
            source = f"{self.file_path}.{index}"
            target = f"{self.file_path}.{index + 1}"
            if os.path.exists(source):
                os.replace(source, target)

        os.replace(self.file_path, f"{self.file_path}.1")
