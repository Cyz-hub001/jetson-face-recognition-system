import json
import os
from datetime import datetime
from typing import Optional, Dict, Any


class AccessLogger:
    """
    业务事件日志
    一行一条 JSON，方便后续 Web 读取
    """

    def __init__(self, log_dir: str = "logs", log_file: str = "access.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_file)

    def write_event(
        self,
        event_type: str,
        person_name: Optional[str] = None,
        similarity: Optional[float] = None,
        result: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "person_name": person_name,
            "similarity": similarity,
            "result": result,
            "extra": extra or {}
        }

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
