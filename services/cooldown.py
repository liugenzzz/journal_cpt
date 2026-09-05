from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any


def runtime_dir(cfg: dict[str, Any], name: str, fallback: Path | None = None) -> Path | None:
    """output_root/.runtime/<name>，和 MinerU 槽位放在同一处。"""
    output_root = str(cfg.get("runtime", {}).get("output_root") or "").strip()
    root = Path(output_root) if output_root else fallback
    if root is None:
        return None
    return root / ".runtime" / name


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return cleaned or "provider"


class SharedCooldown:
    """跨进程共享的 provider 冷却状态。

    journal 级跑在 ProcessPoolExecutor 里，每个子进程各建一份 pool，冷却状态原本
    互不可见：一个 provider 挂掉后每个进程都要各自踩一次才会进冷却，
    故障被放大而不是被隔离。

    冷却是低频写（只在失败时）、高频读（每次调度都要看），所以：
      - 写立即落盘，让其他进程尽快看到；
      - 读带一个短 TTL 的内存缓存，避免每次调度都打文件系统；
      - 每个 provider 一个文件，写之间互不干扰，不需要读-改-写，也就没有丢更新的竞态。

    时间戳一律用 wall clock：time.monotonic() 的原点是每个进程各自的，跨进程不可比。
    """

    def __init__(self, directory: Path | None, ttl_seconds: float = 1.0) -> None:
        self.directory = directory
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._cache: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _path(self, name: str) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / f"{_safe_name(name)}.cooldown"

    def _read_until(self, name: str) -> float:
        path = self._path(name)
        if path is None:
            return 0.0
        try:
            return float(path.read_text(encoding="utf-8").strip() or 0.0)
        except (OSError, ValueError):
            return 0.0

    def until(self, name: str) -> float:
        """返回冷却截止的 wall clock 时间戳；未冷却返回 0。"""
        if self.directory is None:
            return 0.0
        now = time.time()
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None and now - cached[0] < self.ttl_seconds:
                return cached[1]
        until = self._read_until(name)
        with self._lock:
            self._cache[name] = (now, until)
        return until

    def cooling_down(self, name: str) -> bool:
        return self.until(name) > time.time()

    def mark(self, name: str, seconds: float) -> None:
        path = self._path(name)
        if path is None or seconds <= 0:
            return
        until = time.time() + float(seconds)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
            tmp_path.write_text(f"{until:.3f}", encoding="utf-8")
            tmp_path.replace(path)
        except OSError:
            return
        with self._lock:
            self._cache[name] = (time.time(), until)

    def clear(self, name: str) -> None:
        path = self._path(name)
        if path is None:
            return
        try:
            path.unlink()
        except OSError:
            pass
        with self._lock:
            self._cache[name] = (time.time(), 0.0)
