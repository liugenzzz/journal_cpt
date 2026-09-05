from __future__ import annotations

import logging
import os
from typing import Any

from .progress import ProgressBar


class _ProgressAwareHandler(logging.StreamHandler):
    """打日志前先把进度条那一行擦掉，打完再重绘，避免互相覆盖。"""

    def emit(self, record: logging.LogRecord) -> None:
        bar = ProgressBar.active
        if bar is not None:
            bar.clear_line()
        try:
            super().emit(record)
        finally:
            if bar is not None:
                bar.redraw()


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    name = str(level or os.getenv("JOURNAL_CPT_LOG_LEVEL") or "INFO").upper()
    return getattr(logging, name, logging.INFO)


# pypdf 解析结构有瑕疵的 PDF 时会刷 "Object N 0 found" 之类的警告，
# 它走自己的 logger，--log-level 管不到，这里统一压掉。
for _noisy in ("pypdf", "PIL", "PIL.PngImagePlugin", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


def configure_logger(name: str, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    stream_handler = _ProgressAwareHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


class RunState:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.metrics: dict[str, Any] = {}
        self.checkpoints: dict[str, Any] = {}

    def error(self, payload: dict[str, Any]) -> None:
        self.logger.error("pipeline_error %s", payload)

    def increment(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = int(self.metrics.get(key, 0)) + amount
        # 每个计数都打 INFO 会把日志刷爆，降到 DEBUG；
        # 要看就 --log-level DEBUG。
        self.logger.debug("metric %s=%s", key, self.metrics[key])

    def checkpoint(self, key: str, payload: Any) -> None:
        self.checkpoints[key] = payload
        self.logger.debug("checkpoint %s=%s", key, payload)
