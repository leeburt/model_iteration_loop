"""日志工具。

提供统一的 logger 创建和文件 handler 追加功能，
日志同时输出到 stdout 和文件。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def add_file_handler(logger: logging.Logger, log_file: str | Path) -> None:
    """为已有 logger 追加一个文件 handler，自动创建父目录。"""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    """创建并配置 logger，同时输出到 stdout 和日志文件。

    每次调用会清除已有 handler，避免重复输出。
    propagate 设为 False 以隔离日志。
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    add_file_handler(logger, path)

    return logger
