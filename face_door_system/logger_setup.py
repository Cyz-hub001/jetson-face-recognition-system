import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "face_door_system",
    log_dir: str = "logs",
    log_file: str = "system.log",
    level: str = "INFO"
) -> logging.Logger:
    """
    创建系统日志记录器
    - 控制台输出
    - 文件输出
    - 按大小轮转
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 handler
    try:
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, log_file),
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"File logger unavailable, console only: {e}")

    return logger
