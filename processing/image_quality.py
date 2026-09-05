from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


def _filter_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("crop_filter", {})


def crop_filter_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_filter_cfg(cfg).get("enabled", True))


def should_skip_crop_block(block_type: str, cfg: dict[str, Any], text: str = "") -> bool:
    if not crop_filter_enabled(cfg):
        return False
    crop_cfg = _filter_cfg(cfg)
    skip_types = set(crop_cfg.get("skip_block_types", []))
    if block_type in skip_types:
        return True
    clean_text = text.strip()
    min_text_chars = int(crop_cfg.get("min_text_chars", 0) or 0)
    if min_text_chars and len(clean_text) < min_text_chars:
        return True
    patterns = crop_cfg.get("skip_text_patterns", [])
    return any(re.search(str(pattern), clean_text) for pattern in patterns)


def is_valid_crop_box(width: int, height: int, cfg: dict[str, Any]) -> bool:
    if not crop_filter_enabled(cfg):
        return width > 0 and height > 0
    crop_cfg = _filter_cfg(cfg)
    min_width = int(crop_cfg.get("min_width", 1))
    min_height = int(crop_cfg.get("min_height", 1))
    min_area = int(crop_cfg.get("min_area", 1))
    return width >= min_width and height >= min_height and width * height >= min_area


def is_meaningful_image(image: Any, cfg: dict[str, Any]) -> bool:
    if not crop_filter_enabled(cfg):
        return True
    try:
        from PIL import ImageStat  # type: ignore
    except ImportError:
        return True

    crop_cfg = _filter_cfg(cfg)
    width, height = image.size
    if not is_valid_crop_box(width, height, cfg):
        return False

    gray = image.convert("L")
    total_pixels = max(1, width * height)
    white_threshold = int(crop_cfg.get("white_pixel_threshold", 245))
    histogram = gray.histogram()
    white_pixels = sum(histogram[white_threshold:])
    blank_ratio = white_pixels / total_pixels
    non_white_ratio = 1.0 - blank_ratio
    stddev = float(ImageStat.Stat(gray).stddev[0])

    max_blank_ratio = float(crop_cfg.get("max_blank_ratio", 0.985))
    min_non_white_ratio = float(crop_cfg.get("min_non_white_ratio", 0.005))
    min_intensity_stddev = float(crop_cfg.get("min_intensity_stddev", 3.0))
    return (
        blank_ratio < max_blank_ratio
        and non_white_ratio >= min_non_white_ratio
        and stddev >= min_intensity_stddev
    )


def is_complete_text_crop(image: Any, cfg: dict[str, Any]) -> bool:
    if not crop_filter_enabled(cfg):
        return True
    crop_cfg = _filter_cfg(cfg)
    width, height = image.size
    if not is_valid_crop_box(width, height, cfg):
        return False

    gray = image.convert("L")
    threshold = int(crop_cfg.get("white_pixel_threshold", 245))
    ink_points: list[tuple[int, int]] = []
    for y in range(height):
        for x, pixel in enumerate(gray.crop((0, y, width, y + 1)).getdata()):
            if int(pixel) < threshold:
                ink_points.append((x, y))

    if not ink_points:
        return False

    min_x = min(point[0] for point in ink_points)
    max_x = max(point[0] for point in ink_points)
    min_y = min(point[1] for point in ink_points)
    max_y = max(point[1] for point in ink_points)
    margin = int(crop_cfg.get("min_text_content_margin", 0) or 0)
    if min_x < margin or min_y < margin or width - 1 - max_x < margin or height - 1 - max_y < margin:
        return False

    band = max(1, int(crop_cfg.get("edge_ink_band", 1) or 1))
    edge_pixels = 0
    edge_ink = 0
    for x, y in ink_points:
        if x < band or y < band or width - band <= x or height - band <= y:
            edge_ink += 1
    edge_pixels = max(1, (width * min(band, height) * 2) + (height * min(band, width) * 2))
    max_edge_ink_ratio = float(crop_cfg.get("max_edge_ink_ratio", 0.0) or 0.0)
    return edge_ink / edge_pixels <= max_edge_ink_ratio


def _quality_signature(cfg: dict[str, Any]) -> tuple[Any, ...]:
    """判定结果只取决于这几个配置项，用它们做缓存键的一部分。"""
    crop_cfg = _filter_cfg(cfg)
    return (
        int(crop_cfg.get("min_width", 1)),
        int(crop_cfg.get("min_height", 1)),
        int(crop_cfg.get("min_area", 1)),
        int(crop_cfg.get("white_pixel_threshold", 245)),
        float(crop_cfg.get("max_blank_ratio", 0.985)),
        float(crop_cfg.get("min_non_white_ratio", 0.005)),
        float(crop_cfg.get("min_intensity_stddev", 3.0)),
    )


@lru_cache(maxsize=8192)
def _meaningful_image_file_cached(path_str: str, mtime_ns: int, size: int, cfg_ref: "_CfgRef") -> bool:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path_str) as image:
            return is_meaningful_image(image, cfg_ref.cfg)
    except Exception:
        return False


class _CfgRef:
    """让 cfg 能进 lru_cache 的键：相等性只看 signature，实际读的还是原 cfg。"""

    __slots__ = ("cfg", "signature")

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.signature = _quality_signature(cfg)

    def __hash__(self) -> int:
        return hash(self.signature)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _CfgRef) and self.signature == other.signature


def is_meaningful_image_file(path: Path, cfg: dict[str, Any]) -> bool:
    """按 (路径, mtime, 大小, 判定参数) 缓存。

    同一张页面图会被多个任务的多个样本反复送进校验，每次都是
    open + convert("L") + histogram + stddev 三趟全量扫描；180dpi 整页图约 390 万像素。
    mtime 和大小进键，--force-rebuild 重新生成图片后不会命中旧结果。
    """
    if not crop_filter_enabled(cfg):
        return path.exists()
    try:
        stat = path.stat()
    except OSError:
        return False
    if not path.is_file():
        return False
    return _meaningful_image_file_cached(str(path), stat.st_mtime_ns, stat.st_size, _CfgRef(cfg))
