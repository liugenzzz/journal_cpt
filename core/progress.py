from __future__ import annotations

import sys
import threading
import time
from typing import Any


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressBar:
    """极简进度条，写 stderr，不依赖 tqdm。

    非 TTY（重定向到文件、nohup）时自动退化成每次一行的纯文本进度，
    不会往日志里塞一堆回车符。
    """

    active: "ProgressBar | None" = None
    _lock = threading.RLock()

    def __init__(self, total: int, desc: str = "", enabled: bool = True, width: int = 24) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.width = width
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.current = ""
        self.started_at = time.time()
        self.enabled = bool(enabled) and self.total > 0
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._last_len = 0

    def __enter__(self) -> "ProgressBar":
        if self.enabled:
            with ProgressBar._lock:
                ProgressBar.active = self
            self._render()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.enabled:
            self._render(final=True)
            if self.is_tty:
                sys.stderr.write("\n")
                sys.stderr.flush()
            with ProgressBar._lock:
                if ProgressBar.active is self:
                    ProgressBar.active = None

    # ---- 供 logging handler 调用，打日志前后清屏/重绘 ----
    def clear_line(self) -> None:
        if self.enabled and self.is_tty and self._last_len:
            sys.stderr.write("\r" + " " * self._last_len + "\r")
            sys.stderr.flush()
            self._last_len = 0

    def redraw(self) -> None:
        if self.enabled and self.is_tty:
            self._render()

    def set_current(self, text: str) -> None:
        self.current = text or ""
        if self.enabled and self.is_tty:
            self._render()

    def advance(self, ok: bool = True, current: str = "") -> None:
        with ProgressBar._lock:
            self.done += 1
            if ok:
                self.ok += 1
            else:
                self.failed += 1
            if current:
                self.current = current
            self._render()

    def _render(self, final: bool = False) -> None:
        if not self.enabled:
            return
        ratio = self.done / self.total if self.total else 1.0
        elapsed = time.time() - self.started_at
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0.0
        filled = int(self.width * ratio)
        bar = "█" * filled + "░" * (self.width - filled)
        head = f"{self.desc} " if self.desc else ""
        stats = f"{self.done}/{self.total} {ratio * 100:5.1f}%"
        counts = f"✓{self.ok}" + (f" ✗{self.failed}" if self.failed else "")
        timing = f"{_fmt_seconds(elapsed)}<{_fmt_seconds(eta)}"

        if not self.is_tty:
            # 非 TTY：每完成一个打一行，方便 nohup 日志里看进度
            name = f" | {self.current}" if self.current else ""
            sys.stderr.write(f"[进度] {head}{stats} | {counts} | {timing}{name}\n")
            sys.stderr.flush()
            return

        name = self.current
        line = f"\r{head}[{bar}] {stats} | {counts} | {timing}"
        if name:
            room = max(0, 110 - len(line))
            if room > 6:
                if len(name) > room:
                    name = name[: room - 1] + "…"
                line += f" | {name}"
        pad = max(0, self._last_len - len(line))
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._last_len = len(line)
