import json
import logging
import os
from logging.handlers import RotatingFileHandler


class JsonLineFormatter(logging.Formatter):
    RESERVED_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "event_type": getattr(record, "event_type", "system"),
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in self.RESERVED_ATTRS or key in payload:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logger(
    name: str = "face_door_system",
    log_dir: str = "logs",
    log_file: str = "system.log",
    level: str = "INFO",
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_formatter = JsonLineFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"File logger unavailable, console only: {e}")

    return logger
