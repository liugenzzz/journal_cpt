from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .image_quality import (
    is_complete_text_crop,
    is_meaningful_image,
    is_valid_crop_box,
    should_skip_crop_block,
)
from ..core.io_utils import write_json
from ..core.models import JournalBlockRecord, JournalPageRecord


def _page_from_dict(payload: dict[str, Any]) -> JournalPageRecord:
    blocks = [JournalBlockRecord(**block) for block in payload.get("blocks", [])]
    payload = dict(payload)
    payload["blocks"] = blocks
    return JournalPageRecord(**payload)


def _bbox(block: JournalBlockRecord, page_width: int, page_height: int) -> tuple[int, int, int, int] | None:
    if len(block.bbox) != 4:
        return None
    left, top, right, bottom = [int(round(value)) for value in block.bbox]
    left = max(0, min(left, page_width))
    right = max(0, min(right, page_width))
    top = max(0, min(top, page_height))
    bottom = max(0, min(bottom, page_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def _is_text_crop_block(block_type: str, cfg: dict[str, Any]) -> bool:
    return block_type in set(cfg.get("crop_filter", {}).get("text_block_types", []))


def _visual_container_coverage_ok(
    box: tuple[int, int, int, int],
    visual_boxes: list[tuple[int, int, int, int]],
    cfg: dict[str, Any],
) -> bool:
    crop_cfg = cfg.get("crop_filter", {})
    min_coverage = float(crop_cfg.get("min_visual_container_coverage", 0.0) or 0.0)
    if not min_coverage:
        return True
    box_area = _area(box)
    if not box_area:
        return False
    overlap_ratio = float(crop_cfg.get("visual_container_overlap_ratio", 0.9) or 0.9)
    containing_boxes = [
        visual_box
        for visual_box in visual_boxes
        if _area(visual_box) > box_area and _intersection_area(box, visual_box) / box_area >= overlap_ratio
    ]
    if not containing_boxes:
        return True
    best_coverage = max(box_area / _area(visual_box) for visual_box in containing_boxes if _area(visual_box))
    return best_coverage >= min_coverage


def _crop_page_worker(args: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    page_payload, output_dir_text, cfg = args
    page = _page_from_dict(page_payload)
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return asdict(page)

    output_dir = Path(output_dir_text)
    page_image = output_dir / page.page_image
    if not page_image.exists():
        return asdict(page)

    with Image.open(page_image) as image:
        page.width, page.height = image.size
        visual_types = set(cfg.get("crop_filter", {}).get("visual_block_types", []))
        visual_boxes = [
            box
            for item in page.blocks
            if item.block_type in visual_types
            for box in [_bbox(item, page.width, page.height)]
            if box is not None
        ]
        for block in page.blocks:
            if should_skip_crop_block(block.block_type, cfg, block.text):
                block.image_path = ""
                continue
            box = _bbox(block, page.width, page.height)
            if box is None:
                continue
            left, top, right, bottom = box
            if not is_valid_crop_box(right - left, bottom - top, cfg):
                block.image_path = ""
                continue
            if not _visual_container_coverage_ok(box, visual_boxes, cfg):
                block.image_path = ""
                continue
            if block.image_path and (output_dir / block.image_path).exists():
                try:
                    with Image.open(output_dir / block.image_path) as existing:
                        valid_existing = is_meaningful_image(existing, cfg)
                        if valid_existing and _is_text_crop_block(block.block_type, cfg):
                            valid_existing = is_complete_text_crop(existing, cfg)
                except Exception:
                    valid_existing = False
                if valid_existing:
                    continue
                block.image_path = ""
                continue
            padding = int(cfg.get("crop_filter", {}).get("padding", 0) or 0)
            crop_left = max(0, left - padding)
            crop_top = max(0, top - padding)
            crop_right = min(page.width, right + padding)
            crop_bottom = min(page.height, bottom + padding)
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            if not is_meaningful_image(crop, cfg):
                block.image_path = ""
                continue
            if _is_text_crop_block(block.block_type, cfg) and not is_complete_text_crop(crop, cfg):
                block.image_path = ""
                continue
            relative = cfg["paths"]["block_images"].format(
                page_no=page.page_index,
                block_id=block.block_id,
                block_type=block.block_type,
            )
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
            block.image_path = relative
    return asdict(page)


def crop_blocks(pages: list[JournalPageRecord], output_dir: Path, cfg: dict[str, Any]) -> None:
    if not cfg["runtime"]["crop_blocks"]:
        return
    crop_workers = max(1, int(cfg["runtime"].get("crop_workers", 1)))
    jobs = [(asdict(page), str(output_dir), cfg) for page in pages]
    if crop_workers == 1 or len(jobs) <= 1:
        results = [_crop_page_worker(job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=crop_workers) as executor:
            futures = [executor.submit(_crop_page_worker, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    pages_by_index = {_page_from_dict(payload).page_index: _page_from_dict(payload) for payload in results}
    for index, page in enumerate(pages):
        updated_page = pages_by_index.get(page.page_index, page)
        pages[index] = updated_page
        write_json(output_dir / updated_page.normalized_page_path, asdict(updated_page))

