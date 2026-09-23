from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text, read_json, safe_name, stable_json_hash, write_json, write_jsonl
from ..core.models import (
    JournalArticleRecord,
    JournalBlockRecord,
    JournalPageRecord,
    JournalRecord,
    NormalizeResult,
)


NORMALIZER_VERSION = "paragraph_crosspage_ncolumn_sellerwatermark_v3"

SUBFIGURE_MARK_RE = re.compile(r"[\(（]\s*([A-Za-z])\s*[\)）]")
FIGURE_LABEL_RE = re.compile(r"((?:\u56fe|Fig\.?|Figure)\s*[0-9\uff10-\uff19]+)", flags=re.IGNORECASE)
FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")


def _watermark_text_patterns(cfg: dict[str, Any]) -> list[str]:
    return [str(pattern) for pattern in cfg.get("watermark", {}).get("text_patterns", [])]


def _looks_like_watermark_text(text: str, cfg: dict[str, Any]) -> bool:
    text = clean_text(text)
    return bool(text) and any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _watermark_text_patterns(cfg))


def _page_from_dict(payload: dict[str, Any]) -> JournalPageRecord:
    blocks = [JournalBlockRecord(**block) for block in payload.get("blocks", [])]
    blocks = _sort_blocks_in_reading_order(blocks)
    payload = dict(payload)
    payload["blocks"] = blocks
    return JournalPageRecord(**payload)


def _item_page(item: dict[str, Any], fallback: int) -> int:
    for key in ("page_idx", "page_index", "page", "page_no"):
        value = item.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_idx" else value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            return number + 1 if key == "page_idx" else number
    return fallback


def _bbox(item: dict[str, Any]) -> list[float]:
    value = item.get("bbox") or item.get("box") or item.get("poly")
    if isinstance(value, list):
        flat: list[float] = []
        for part in value:
            if isinstance(part, list):
                flat.extend(float(x) for x in part if isinstance(x, (int, float)))
            elif isinstance(part, (int, float)):
                flat.append(float(part))
        if len(flat) >= 4:
            return flat[:4]
    return []


def _box(block: JournalBlockRecord) -> tuple[float, float, float, float] | None:
    if len(block.bbox) != 4:
        return None
    left, top, right, bottom = [float(value) for value in block.bbox]
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _top(block: JournalBlockRecord) -> float:
    box = _box(block)
    return box[1] if box else float(block.reading_order)


def _left(block: JournalBlockRecord) -> float:
    box = _box(block)
    return box[0] if box else 0.0


def _bottom(block: JournalBlockRecord) -> float:
    box = _box(block)
    return box[3] if box else _top(block)


def _center_x(block: JournalBlockRecord) -> float:
    box = _box(block)
    return (box[0] + box[2]) / 2 if box else 0.0


def _center_y(block: JournalBlockRecord) -> float:
    box = _box(block)
    return (box[1] + box[3]) / 2 if box else _top(block)


_COLUMN_WORDS = {1: "single", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _is_two_column_mode(column_mode: str) -> bool:
    return "two_column" in str(column_mode or "")


def is_multi_column_mode(column_mode: str) -> bool:
    """两栏及以上。三栏杂志不能再走"单栏按 y 排"的老路。"""
    mode = str(column_mode or "")
    return any(f"{word}_column" in mode for count, word in _COLUMN_WORDS.items() if count >= 2)


def column_count(column_mode: str) -> int:
    mode = str(column_mode or "")
    for count, word in sorted(_COLUMN_WORDS.items(), reverse=True):
        if f"{word}_column" in mode:
            return count
    return 0


def _column_name(index: int, total: int) -> str:
    """两栏沿用 left/right，下游既有逻辑和导出标签都不用改；三栏以上用 col1..colN。"""
    if total == 1:
        return "single"
    if total == 2:
        return "left" if index == 0 else "right"
    return f"col{index + 1}"


def _sort_blocks_in_reading_order(blocks: list[JournalBlockRecord]) -> list[JournalBlockRecord]:
    return sorted(blocks, key=lambda item: (item.reading_order or 9999, _top(item), _left(item)))


def _block_type(raw: str, cfg: dict[str, Any]) -> str:
    return str(cfg["block_types"].get(raw.lower(), cfg["block_types"]["unknown"]))


def _item_text(item: dict[str, Any]) -> str:
    keys = (
        "text",
        "content",
        "md_content",
        "markdown",
        "caption",
        "img_caption",
        "image_caption",
        "table_caption",
        "image_footnote",
        "latex",
        "html",
    )
    return clean_text(" ".join(clean_text(item.get(key)) for key in keys))


def _iter_image_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"img_path", "image_path", "table_img_path", "figure_path", "path", "file_path"}:
                text = clean_text(child)
                if text:
                    refs.append(text)
            else:
                refs.extend(_iter_image_refs(child))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_iter_image_refs(item))
    return refs


def _item_image_ref(item: dict[str, Any]) -> str:
    refs = _iter_image_refs(item)
    return refs[0] if refs else ""


def _resolve_extracted_image(image_ref: str, image_map: dict[str, str]) -> str:
    if not image_ref:
        return ""
    if image_ref in image_map:
        return image_map[image_ref]
    normalized = image_ref.replace("\\", "/")
    for key, value in image_map.items():
        key_normalized = key.replace("\\", "/")
        if key_normalized == normalized or Path(key_normalized).name == Path(normalized).name:
            return value
    return ""


def _page_size(output_dir: Path, page_image: str, blocks: list[JournalBlockRecord]) -> tuple[int, int]:
    path = output_dir / page_image
    if path.exists():
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as image:
                return int(image.width), int(image.height)
        except Exception:
            pass
    max_right = max((box[2] for block in blocks if (box := _box(block))), default=0.0)
    max_bottom = max((box[3] for block in blocks if (box := _box(block))), default=0.0)
    return int(max_right), int(max_bottom)


def _layout_size(blocks: list[JournalBlockRecord], image_width: int, image_height: int) -> tuple[int, int]:
    max_right = max((box[2] for block in blocks if (box := _box(block))), default=0.0)
    max_bottom = max((box[3] for block in blocks if (box := _box(block))), default=0.0)
    layout_width = int(max_right) if max_right > 0 else image_width
    layout_height = int(max_bottom) if max_bottom > 0 else image_height
    return layout_width, layout_height


def _looks_like_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"(第\s*)?\d{1,4}\s*(页)?", text.strip()))


def _looks_like_reference(text: str) -> bool:
    return bool(re.match(r"^\s*(\[\s*\d+\s*\]|\d+\.\s+)", text)) or "参考文献" in text


def _looks_like_section_heading(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) > 80:
        return False
    if re.match(r"^(摘\s*要|关键词|Abstract|Key\s*words?)[:：]?", text, flags=re.IGNORECASE):
        return False
    patterns = [
        r"^\d+(\.\d+)*\s+[\u4e00-\u9fffA-Za-z].{1,60}$",
        r"^[一二三四五六七八九十]+[、.]\s*[\u4e00-\u9fffA-Za-z].{1,60}$",
        r"^(引言|前言|方法|模型|试验|实验|仿真|结果|分析|讨论|结论|结束语)$",
    ]
    return any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _caption_kind(text: str) -> str:
    text = clean_text(text)
    if re.match(r"^(图|Fig\.?|Figure)\s*[0-9一二三四五六七八九十]+", text, flags=re.IGNORECASE):
        return "figure_caption"
    if re.match(r"^(表|Table)\s*[0-9一二三四五六七八九十]+", text, flags=re.IGNORECASE):
        return "table_caption"
    return ""


def _caption_label(text: str) -> str:
    text = clean_text(text)
    match = re.search(
        r"(图\s*[0-9一二三四五六七八九十]+|表\s*[0-9一二三四五六七八九十]+|式\s*[（(]?\s*[0-9]+|Fig\.?\s*[0-9]+|Figure\s*[0-9]+|Table\s*[0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    return clean_text(match.group(1)) if match else ""


def _subfigure_marker(text: str) -> tuple[str, int, int] | None:
    match = SUBFIGURE_MARK_RE.search(clean_text(text))
    if not match:
        return None
    return match.group(1).lower(), match.start(), match.end()


def _figure_label_matches(text: str) -> list[re.Match[str]]:
    return list(FIGURE_LABEL_RE.finditer(clean_text(text)))


def _normalize_figure_label_key(label: str) -> str:
    label = clean_text(label).translate(FULLWIDTH_DIGIT_TRANS)
    digit_match = re.search(r"\d+", label)
    if digit_match:
        return f"figure:{digit_match.group(0)}"
    return re.sub(r"\s+", "", label.lower())


def _figure_label_after_subfigure(text: str) -> str:
    marker = _subfigure_marker(text)
    min_start = marker[2] if marker else 0
    for match in _figure_label_matches(text):
        if match.start() >= min_start:
            return clean_text(match.group(1))
    return ""


def _block_has_figure_label(block: JournalBlockRecord, label_key: str) -> bool:
    return any(_normalize_figure_label_key(match.group(1)) == label_key for match in _figure_label_matches(block.text))


def _subfigure_label_text(labels: list[str]) -> str:
    return "、".join(f"({label})" for label in labels if label)


def _union_box(blocks: list[JournalBlockRecord]) -> tuple[float, float, float, float] | None:
    boxes = [box for block in blocks if (box := _box(block)) is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _caption_blocks_for_subfigure_group(
    page: JournalPageRecord,
    label: str,
    members: list[JournalBlockRecord],
) -> list[JournalBlockRecord]:
    label_key = _normalize_figure_label_key(label)
    member_ids = {member.block_id for member in members}
    group_box = _union_box(members)
    scored: list[tuple[float, JournalBlockRecord]] = []
    for block in page.blocks:
        if block.block_id in member_ids or not clean_text(block.text) or not _block_has_figure_label(block, label_key):
            continue
        if block.semantic_role not in {"figure_caption", "table_caption"} and not _figure_label_matches(block.text):
            continue
        block_box = _box(block)
        if group_box is None or block_box is None:
            scored.append((float(block.reading_order or 9999), block))
            continue
        horizontal_overlap = max(0.0, min(group_box[2], block_box[2]) - max(group_box[0], block_box[0]))
        overlap_ratio = horizontal_overlap / max(1.0, min(group_box[2] - group_box[0], block_box[2] - block_box[0]))
        if overlap_ratio < 0.15:
            continue
        vertical_gap = min(abs(block_box[1] - group_box[3]), abs(group_box[1] - block_box[3]))
        scored.append((vertical_gap, block))
    return [block for _, block in sorted(scored, key=lambda item: item[0])[:3]]


def _crop_subfigure_group_image(
    page: JournalPageRecord,
    output_dir: Path | None,
    members: list[JournalBlockRecord],
    cfg: dict[str, Any],
    group_id: str,
) -> str:
    if output_dir is None:
        return ""
    page_image = output_dir / page.page_image
    if not page_image.exists():
        return ""
    union = _union_box(members)
    if union is None:
        return ""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return ""
    with Image.open(page_image) as image:
        page.width, page.height = image.size
        padding = int(cfg.get("crop_filter", {}).get("padding", 0) or 0)
        left = max(0, int(round(union[0])) - padding)
        top = max(0, int(round(union[1])) - padding)
        right = min(page.width, int(round(union[2])) + padding)
        bottom = min(page.height, int(round(union[3])) + padding)
        if right <= left or bottom <= top:
            return ""
        relative = cfg["paths"]["block_images"].format(
            page_no=page.page_index,
            block_id=safe_name(group_id, "subfigure_group"),
            block_type="figure",
        )
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(target)
    return relative


def _caption_note_for_subfigure_chunk(
    label: str,
    chunk_labels: list[str],
    all_labels: list[str],
    chunk_count: int,
) -> str:
    chunk_text = _subfigure_label_text(chunk_labels)
    if chunk_count > 1:
        remaining = [item for item in all_labels if item not in set(chunk_labels)]
        remaining_text = _subfigure_label_text(remaining)
        if remaining_text:
            return f"子图抽取说明：本裁剪包含{label}的{chunk_text}子图；同一图号还包含{remaining_text}子图，已分片抽取。"
        return f"子图抽取说明：本裁剪包含{label}的{chunk_text}子图；同一图号已分片抽取。"
    return f"子图抽取说明：本裁剪已合并{label}的{chunk_text}子图。"


def _semantic_role(
    block_type: str,
    text: str,
    box: tuple[float, float, float, float] | None,
    page_height: int,
    page_no: int,
    cfg: dict[str, Any] | None = None,
) -> str:
    text = clean_text(text)
    # 卖家水印优先判定：它在页脚，但 MinerU 不一定把它切成 footer 类型。
    if looks_like_seller_watermark(text, cfg or {}):
        return "watermark"
    if block_type == "header":
        return "journal_header"
    if block_type == "footer":
        return "footer"
    if block_type == "page_number" or _looks_like_page_number(text):
        return "page_number"
    if box and page_height:
        if box[1] <= page_height * 0.05 and len(text) < 80:
            return "journal_header"
        if box[3] >= page_height * 0.95 and len(text) < 80:
            return "footer"
    caption = _caption_kind(text)
    if caption:
        return caption
    if block_type == "formula":
        return "formula"
    if re.match(r"^摘\s*要", text) or text.lower().startswith("abstract"):
        return "abstract"
    if text.startswith("关键词") or re.match(r"^key\s*words?", text, flags=re.IGNORECASE):
        return "keywords"
    if text.startswith("http://") or text.startswith("https://"):
        return "journal_header"
    if "DOI" in text.upper() or "中图分类号" in text or "文献标识码" in text or "收稿日期" in text:
        return "journal_header"
    if text.startswith("\\* 通信作者") or text.startswith("* 通信作者") or text.startswith("通信作者") or text.startswith("引用格式"):
        return "footer"
    if _looks_like_reference(text):
        return "reference"
    if _looks_like_section_heading(text) or block_type in {"title", "section_title"}:
        if page_no == 1 and len(text) > 8 and not re.match(r"^\d", text):
            return "article_title"
        return "section_heading"
    return "body" if text else "unknown"


def _layout_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return (cfg or {}).get("layout", {})


def _column_bands(blocks: list[JournalBlockRecord], width: float, cfg: dict[str, Any] | None) -> list[tuple[float, float]]:
    """从"窄块"的水平覆盖投影里找出每一栏的左右边界。

    原来的做法是假设两栏、按页宽中线劈开，三栏版面直接崩：中间那栏正好横跨中线，
    会被 crosses_mid 判成跨栏图表，读序整个乱掉。
    这里改成先找栏：把明显跨栏的宽块排除掉，剩下的窄块在 x 轴上投影，
    连续被覆盖的区间就是一栏，区间之间的空白就是栏间距。栏数由数据决定。
    """
    layout = _layout_cfg(cfg)
    bins = max(50, int(layout.get("column_scan_bins", 400)))
    min_gutter = float(layout.get("min_gutter_ratio", 0.012))
    min_band = float(layout.get("min_band_ratio", 0.06))
    max_columns = max(1, int(layout.get("max_columns", 6)))
    ratios = [float(r) for r in layout.get("narrow_scan_ratios", [0.45, 0.55, 0.65, 0.75, 0.85])]

    def scan(max_narrow: float) -> list[tuple[float, float]]:
        cover = [0] * bins
        for block in blocks:
            box = _box(block)
            if not box:
                continue
            if (box[2] - box[0]) >= width * max_narrow:
                continue
            lo = max(0, min(bins - 1, int(box[0] / width * bins)))
            hi = max(lo + 1, min(bins, int(math.ceil(box[2] / width * bins))))
            for index in range(lo, hi):
                cover[index] += 1

        gap_bins = max(1, int(round(min_gutter * bins)))
        runs: list[tuple[int, int]] = []
        start: int | None = None
        gap = 0
        for index in range(bins):
            if cover[index] > 0:
                if start is None:
                    start = index
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= gap_bins:
                    runs.append((start, index - gap + 1))
                    start = None
                    gap = 0
        if start is not None:
            runs.append((start, bins))
        return [
            (lo / bins * width, hi / bins * width)
            for lo, hi in runs
            if (hi - lo) / bins >= min_band
        ]

    # 单一阈值分不开这两种情况：非对称两栏里 0.57 宽的正文栏是真栏，
    # 三栏页里 0.65 宽的两栏跨图是假栏。
    # 但跨栏块只会把相邻栏"粘成一条"，永远不会把一栏劈成两条 ——
    # 所以在几个阈值里取栏数最多的那次结果，就是真实栏数。
    best: list[tuple[float, float]] = []
    for ratio in sorted(set(ratios)):
        found = scan(ratio)
        if len(found) > len(best) and len(found) <= max_columns:
            best = found
    return best


def _assign_columns(blocks: list[JournalBlockRecord], width: int, cfg: dict[str, Any] | None = None) -> str:
    if not blocks:
        return "unknown"
    if width <= 0:
        width = int(max((box[2] for block in blocks if (box := _box(block))), default=0))
    if width <= 0:
        return "unknown"

    bands = _column_bands(blocks, float(width), cfg)
    if not bands:
        # 整页通栏（大图跨页、满版照片）时所有块都比 max_narrow_ratio 宽，
        # 投影里找不到栏。这时把内容范围当成唯一一栏，行为和改造前一致。
        boxes = [box for block in blocks if (box := _box(block))]
        if not boxes:
            for block in blocks:
                block.column = "unknown"
                block.column_index = -1
                block.span_kind = "single_column"
            return "unknown"
        bands = [(min(box[0] for box in boxes), max(box[2] for box in boxes))]
    total = len(bands)

    overlap_ratio = float(_layout_cfg(cfg).get("band_overlap_ratio", 0.35))
    column_counts = [0] * total
    full_count = 0
    cross_count = 0
    for block in blocks:
        box = _box(block)
        if not box:
            block.column = "unknown"
            block.column_index = -1
            block.span_kind = "single_column"
            continue
        block_width = max(1e-6, box[2] - box[0])
        hit = []
        for index, (low, high) in enumerate(bands):
            overlap = min(box[2], high) - max(box[0], low)
            if overlap > min(block_width, high - low) * overlap_ratio:
                hit.append(index)
        if len(hit) == 1:
            index = hit[0]
            block.column = _column_name(index, total)
            block.column_index = index
            block.span_kind = "single_column"
            column_counts[index] += 1
        elif len(hit) > 1:
            block.column = "full_width"
            block.column_index = -1
            if len(hit) >= total:
                block.span_kind = "full_width"
                full_count += 1
            else:
                block.span_kind = "cross_column"
                cross_count += 1
        else:
            # 没压到任何栏（页边噪声之类）：归到中心点最近的一栏。
            center = _center_x(block)
            index = min(range(total), key=lambda i: abs(center - (bands[i][0] + bands[i][1]) / 2.0))
            block.column = _column_name(index, total)
            block.column_index = index
            block.span_kind = "single_column"
            column_counts[index] += 1

    # 旧代码要求"每栏至少 2 块"，是为了防止按中线硬切造成的假两栏。
    # 现在栏是从投影聚类出来的，本身就可靠，只要该栏有内容就算数 ——
    # 否则三栏页里某一栏只有一段正文时，会被误报成两栏。
    populated = sum(1 for count in column_counts if count >= 1)
    spanning = full_count + cross_count
    if populated >= 2:
        word = _COLUMN_WORDS.get(populated, f"{populated}")
        return f"mixed_full_width_and_{word}_column" if spanning else f"{word}_column"
    if spanning and any(column_counts):
        return "mixed_full_width_and_single_column"
    if spanning:
        return "full_width"
    if any(column_counts):
        return "single_column"
    return "unknown"


def _noise(block: JournalBlockRecord) -> bool:
    if block.semantic_role == "watermark":
        return True
    return block.semantic_role in {"journal_header", "footer", "page_number", "unknown"} and not clean_text(block.text)


def _skip_from_train_text(block: JournalBlockRecord, page_type: str) -> bool:
    if block.semantic_role in {"journal_header", "footer", "page_number", "watermark"}:
        return True
    if page_type in {"cover", "editorial_board", "table_of_contents", "advertisement_or_notice", "blank"}:
        return True
    if page_type == "references" or block.semantic_role == "reference":
        return True
    return False


def _visible_blocks(page: JournalPageRecord) -> list[JournalBlockRecord]:
    return [
        block
        for block in sorted(page.blocks, key=lambda item: item.reading_order or 9999)
        if clean_text(block.markdown or block.text) and not _skip_from_train_text(block, page.page_type)
    ]


def _visible_text(page: JournalPageRecord) -> str:
    return "\n".join(clean_text(block.markdown or block.text) for block in _visible_blocks(page))


def _looks_like_list_item(text: str) -> bool:
    return bool(re.match(r"^\s*(\d+|[一二三四五六七八九十]+)\s*[)）.、]", clean_text(text)))


def _ends_paragraph(text: str) -> bool:
    text = clean_text(text).rstrip()
    if not text:
        return True
    if re.search(r"(。|！|？|；|;|\.|\?|!|：|:|”|’|\)|）|\])$", text):
        return True
    return False


def _join_paragraph_text(previous: str, current: str) -> str:
    previous = clean_text(previous).rstrip()
    current = clean_text(current).lstrip()
    if not previous:
        return current
    if not current:
        return previous
    if current[0] in "，。；：、！？,.!?;:%)]}）】”’":
        return previous + current
    if re.search(r"[A-Za-z0-9]$", previous) and re.match(r"^[A-Za-z0-9]", current):
        return previous + " " + current
    return previous + current


def _paragraph_role(block: JournalBlockRecord) -> str:
    if block.semantic_role in {"article_title", "abstract", "keywords", "section_heading", "figure_caption", "table_caption", "formula"}:
        return block.semantic_role
    if block.block_type in {"table", "figure", "formula"}:
        return block.block_type
    return "body"


def _force_new_paragraph(
    *,
    page: JournalPageRecord,
    block: JournalBlockRecord,
    text: str,
    current: dict[str, Any] | None,
    before_first_abstract: bool,
) -> bool:
    if current is None:
        return True
    role = _paragraph_role(block)
    if page.page_type == "article_first_page" and before_first_abstract:
        return True
    if role in {"article_title", "abstract", "keywords", "section_heading", "figure_caption", "table_caption", "table", "figure", "formula"}:
        return True
    if current.get("semantic_role") in {"article_title", "abstract", "keywords", "section_heading", "figure_caption", "table_caption", "table", "figure", "formula"}:
        return True
    if _looks_like_list_item(text):
        return True
    if _ends_paragraph(str(current.get("text") or "")):
        return True
    return False


def _new_paragraph(block: JournalBlockRecord, page: JournalPageRecord, text: str, index: int) -> dict[str, Any]:
    return {
        "paragraph_id": f"p{page.page_index:03d}_para{index:03d}",
        "text": clean_text(text),
        "block_ids": [block.block_id],
        "source_pages": [page.page_index],
        "start_page": page.page_index,
        "end_page": page.page_index,
        "semantic_role": _paragraph_role(block),
        "column": block.column,
        "continues_to_next_page": False,
    }


def _append_to_paragraph(paragraph: dict[str, Any], block: JournalBlockRecord, page: JournalPageRecord, text: str) -> None:
    paragraph["text"] = _join_paragraph_text(str(paragraph.get("text") or ""), text)
    paragraph.setdefault("block_ids", []).append(block.block_id)
    pages = paragraph.setdefault("source_pages", [])
    if page.page_index not in pages:
        pages.append(page.page_index)
    paragraph["end_page"] = page.page_index
    paragraph["continues_to_next_page"] = int(paragraph.get("start_page") or page.page_index) != page.page_index


def _reconstruct_paragraphs(pages: list[JournalPageRecord]) -> None:
    pages_by_article: dict[str, list[JournalPageRecord]] = {}
    for page in pages:
        page.visible_text = _visible_text(page)
        page.paragraphs = []
        page.full_text = ""
        pages_by_article.setdefault(page.article_id, []).append(page)

    for article_pages in pages_by_article.values():
        current: dict[str, Any] | None = None
        paragraph_count_by_page: dict[int, int] = {}

        def flush() -> None:
            nonlocal current
            if current is None:
                return
            start_page = int(current.get("start_page") or 0)
            target = next((page for page in article_pages if page.page_index == start_page), None)
            if target is not None:
                target.paragraphs.append(current)
            current = None

        for page in sorted(article_pages, key=lambda item: item.page_index):
            before_first_abstract = page.page_type == "article_first_page"
            for block in _visible_blocks(page):
                text = clean_text(block.markdown or block.text)
                if not text:
                    continue
                force_new = _force_new_paragraph(
                    page=page,
                    block=block,
                    text=text,
                    current=current,
                    before_first_abstract=before_first_abstract,
                )
                if force_new:
                    flush()
                    paragraph_count_by_page[page.page_index] = paragraph_count_by_page.get(page.page_index, 0) + 1
                    current = _new_paragraph(block, page, text, paragraph_count_by_page[page.page_index])
                else:
                    _append_to_paragraph(current, block, page, text)
                if block.semantic_role == "abstract":
                    before_first_abstract = False
            if page.page_type != "article_first_page":
                before_first_abstract = False
        flush()

    for page in pages:
        page.full_text = "\n\n".join(clean_text(paragraph.get("text")) for paragraph in page.paragraphs if clean_text(paragraph.get("text")))


def _column_index_of(block: JournalBlockRecord) -> int:
    """栏序号；老数据没有 column_index 时退回按 left/right 推断。"""
    index = getattr(block, "column_index", -1)
    if isinstance(index, int) and index >= 0:
        return index
    return {"left": 0, "right": 1}.get(block.column, -1)


def _emit_column_band(blocks: list[JournalBlockRecord]) -> list[JournalBlockRecord]:
    """一个纵向带内：从左到右逐栏输出，每栏内部从上到下。"""
    in_columns = [block for block in blocks if _column_index_of(block) >= 0]
    other = [block for block in blocks if _column_index_of(block) < 0]
    ordered = sorted(in_columns, key=lambda item: (_column_index_of(item), _top(item), _left(item)))
    return ordered + sorted(other, key=lambda item: (_top(item), _left(item)))


def _recover_reading_order(blocks: list[JournalBlockRecord], column_mode: str) -> list[JournalBlockRecord]:
    content = [block for block in blocks if not _noise(block)]
    if not is_multi_column_mode(column_mode):
        ordered = sorted(content, key=lambda item: (_top(item), _left(item), item.reading_order))
    else:
        column_blocks = [block for block in content if _column_index_of(block) >= 0]
        distinct_columns = {_column_index_of(block) for block in column_blocks}
        separators = sorted(
            [block for block in content if _column_index_of(block) < 0],
            key=lambda item: (_top(item), _left(item), item.reading_order),
        )
        if len(distinct_columns) < 2:
            ordered = sorted(content, key=lambda item: (_top(item), _left(item), item.reading_order))
        else:
            ordered = []
            emitted_column_ids: set[int] = set()
            for separator in separators:
                band = [
                    block
                    for block in column_blocks
                    if id(block) not in emitted_column_ids and _top(block) < _top(separator)
                ]
                ordered.extend(_emit_column_band(band))
                emitted_column_ids.update(id(block) for block in band)
                ordered.append(separator)
            remaining = [block for block in column_blocks if id(block) not in emitted_column_ids]
            ordered.extend(_emit_column_band(remaining))
    for index, block in enumerate(ordered, start=1):
        block.reading_order = index
    return ordered


def _two_column_left_then_right_ok(blocks: list[JournalBlockRecord], column_mode: str) -> bool:
    """同一个纵向带里，栏序号必须单调不减：读完右栏又跳回左栏就是读序出错。

    原来只认 left/right 两栏；三栏以上按 column_index 判断。
    """
    if not is_multi_column_mode(column_mode):
        return True
    highest = -1
    seen: set[int] = set()
    for block in _sort_blocks_in_reading_order([item for item in blocks if not _noise(item)]):
        index = _column_index_of(block)
        if index < 0:
            highest = -1  # 跨栏块结束当前带，下一带允许从头开始
            continue
        seen.add(index)
        if index < highest:
            return False
        highest = max(highest, index)
    return len(seen) >= 2


def _matches_any(compact: str, patterns: Any) -> bool:
    for pattern in patterns or []:
        try:
            if re.search(str(pattern), compact, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def looks_like_seller_watermark(text: str, cfg: dict[str, Any]) -> bool:
    """卖家加的推广文字（逐页页脚、二维码页）。

    这批杂志是买来的整期扫描件，每页页脚都印着一行
    "PDF过刊杂志收藏购买微信：bfwz888888"，OCR 后会混进每一页的正文。
    命中的块标成 watermark，既有的 _skip_from_train_text / routing.skip_roles
    就会把它从训练文本里剔除。
    """
    noise_cfg = cfg.get("page_noise", {})
    if not bool(noise_cfg.get("enabled", True)):
        return False
    return _matches_any(clean_text(text), noise_cfg.get("strong_patterns", []))


def _noise_signal_count(compact: str, cfg: dict[str, Any]) -> int:
    """命中了几种不同的广告/声明特征。"""
    noise_cfg = cfg.get("page_noise", {})
    hits = 0
    for pattern in noise_cfg.get("patterns", []):
        try:
            if re.search(str(pattern), compact, flags=re.IGNORECASE):
                hits += 1
        except re.error:
            continue
    return hits


def _looks_like_advertisement(compact: str, cfg: dict[str, Any]) -> bool:
    """订阅广告、杂志社声明、二维码推广页 —— 买来的整期扫描件里夹带的噪声。

    要求命中多个不同特征：正文里偶尔出现一个"网址"或"电话"是正常的，
    但同时出现"邮发代号 + 定价 + 微信 + 声明"就基本只能是广告页。
    """
    noise_cfg = cfg.get("page_noise", {})
    if not bool(noise_cfg.get("enabled", True)):
        return False
    min_signals = max(1, int(noise_cfg.get("min_signals", 2)))
    strong_signals = max(min_signals, int(noise_cfg.get("strong_signals", 4)))
    max_text_chars = max(0, int(noise_cfg.get("max_text_chars", 1200)))
    # 强特征命中一条就够：纯二维码推广页正文只有二三十个字，凑不出多个特征。
    if _matches_any(compact, noise_cfg.get("strong_patterns", [])):
        return True
    hits = _noise_signal_count(compact, cfg)
    if hits >= strong_signals:
        return True
    return hits >= min_signals and len(compact) <= max_text_chars


def _page_type(
    page_no: int,
    text: str,
    blocks: list[JournalBlockRecord],
    column_mode: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    cfg = cfg or {}
    compact = clean_text(text)
    if len(compact) < 8 and not blocks:
        return "blank"
    if len(compact) < 20 and blocks:
        return "scan_only"
    if re.search(r"(目录|目次|CONTENTS)", compact, flags=re.IGNORECASE):
        return "table_of_contents"
    if re.search(r"(编委会|主\s*编|副主编|顾问|Editorial\s+Board)", compact, flags=re.IGNORECASE):
        return "editorial_board"
    if re.search(r"(^|\s)(参考文献|References)\s*$", compact, flags=re.IGNORECASE) or compact.startswith("参考文献"):
        return "references"
    # 广告/声明页要在"正文页"判定之前拦下，否则字数够长就会被当成 article_body。
    if _looks_like_advertisement(compact, cfg):
        return "advertisement_or_notice"
    if page_no == 1 and not re.search(r"(摘要|关键词|Abstract|Key\s*words?)", compact, flags=re.IGNORECASE):
        if re.search(r"(主管|主办|出版|ISSN|CN\s*\d+|第\s*\d+\s*卷)", compact, flags=re.IGNORECASE):
            return "cover"
    has_abstract = re.search(r"(摘\s*要|摘要|Abstract)", compact, flags=re.IGNORECASE)
    has_keywords = re.search(r"(关键词|Key\s*words?)", compact, flags=re.IGNORECASE)
    has_title = any(block.semantic_role == "article_title" for block in blocks)
    if has_abstract and has_keywords and page_no > 1 and re.search(r"(参考文献|作者简介|References)", compact, flags=re.IGNORECASE):
        return "article_body"
    if (has_abstract and has_keywords) or (has_title and has_abstract):
        return "article_first_page"
    if len(compact) >= 80:
        return "article_body"
    return "unknown"


def _title_candidates(page: JournalPageRecord) -> list[JournalBlockRecord]:
    candidates = [block for block in page.blocks if block.semantic_role == "article_title" and clean_text(block.text)]
    if candidates:
        return candidates
    top_text = [
        block
        for block in page.blocks
        if block.semantic_role in {"section_heading", "body"} and clean_text(block.text) and _top(block) < max(1, page.height) * 0.35
    ]
    return sorted(top_text, key=lambda item: (_top(item), _left(item)))[:1]


def _extract_prefixed_text(page: JournalPageRecord, prefixes: tuple[str, ...]) -> str:
    for block in page.blocks:
        text = clean_text(block.text)
        if any(text.startswith(prefix) for prefix in prefixes):
            return text
    return ""


def _keywords(text: str) -> list[str]:
    text = re.sub(r"^(关键词|Key\s*words?)\s*[:：]?", "", text, flags=re.IGNORECASE).strip()
    return [part.strip(" ;；,，") for part in re.split(r"[;；,，]\s*", text) if part.strip(" ;；,，")]


def _authors_after_title(page: JournalPageRecord, title_block: JournalBlockRecord | None) -> list[str]:
    if title_block is None:
        return []
    title_bottom = _bottom(title_block)
    candidates = [
        block
        for block in page.blocks
        if clean_text(block.text)
        and title_bottom <= _top(block) <= title_bottom + max(60, page.height * 0.08)
        and block.semantic_role not in {"abstract", "keywords", "journal_header", "footer", "page_number"}
    ]
    if not candidates:
        return []
    text = clean_text(candidates[0].text)
    text = re.sub(r"[\d,，\s]*(\(|（).*$", "", text)
    parts = [part.strip() for part in re.split(r"[,，、\s]+", text) if 1 < len(part.strip()) <= 8]
    return parts[:12]


def _doi(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0) if match else ""


def _make_article(page: JournalPageRecord, journal: JournalRecord, end_page: int) -> JournalArticleRecord:
    title_blocks = _title_candidates(page)
    title_block = title_blocks[0] if title_blocks else None
    title = clean_text(title_block.text if title_block else "") or journal.journal_name
    abstract = _extract_prefixed_text(page, ("摘要", "摘 要", "Abstract"))
    keyword_text = _extract_prefixed_text(page, ("关键词", "Key words", "Keywords"))
    article_id = safe_name(title, f"article_p{page.page_index:03d}")
    return JournalArticleRecord(
        journal_id=journal.journal_id,
        article_id=article_id,
        article_title=title,
        source_pdf=journal.source_pdf,
        start_page=page.page_index,
        end_page=end_page,
        page_indices=list(range(page.page_index, end_page + 1)),
        authors=_authors_after_title(page, title_block),
        journal_name=journal.journal_name,
        year=journal.year,
        volume=journal.volume,
        issue=journal.issue,
        pages=f"{page.page_label}-{end_page}",
        doi=_doi(page.full_text),
        abstract=abstract,
        keywords=_keywords(keyword_text),
        confidence=0.82 if title_blocks else 0.45,
    )


def _split_articles(pages: list[JournalPageRecord], journal: JournalRecord, cfg: dict[str, Any]) -> list[JournalArticleRecord]:
    starts = [page for page in pages if page.page_type == "article_first_page"]
    if not starts:
        trainable_pages = [page for page in pages if page.page_type in cfg["page_types"]["trainable"]]
        if not trainable_pages:
            return []
        first = trainable_pages[0]
        last = trainable_pages[-1]
        article = _make_article(first, journal, last.page_index)
        article.article_id = cfg["default_article_id"]
        if not article.article_title:
            article.article_title = cfg["default_article_title"]
        return [article]

    articles: list[JournalArticleRecord] = []
    for index, start_page in enumerate(starts):
        next_start = starts[index + 1].page_index if index + 1 < len(starts) else pages[-1].page_index + 1
        end_page = next_start - 1
        while end_page >= start_page.page_index:
            page = pages[end_page - 1]
            if page.page_type in {"references", "table_of_contents", "editorial_board", "cover", "blank"} and end_page > start_page.page_index:
                end_page -= 1
                continue
            break
        articles.append(_make_article(start_page, journal, max(start_page.page_index, end_page)))
    return articles


def _assign_articles(pages: list[JournalPageRecord], articles: list[JournalArticleRecord], cfg: dict[str, Any]) -> None:
    by_page: dict[int, JournalArticleRecord] = {}
    for article in articles:
        for page_index in article.page_indices:
            by_page[page_index] = article
    for page in pages:
        article = by_page.get(page.page_index)
        if article is None:
            page.article_id = cfg["default_article_id"]
            page.article_title = cfg["default_article_title"]
            page.article_role = "unknown"
            continue
        page.article_id = article.article_id
        page.article_title = article.article_title
        if page.page_index == article.start_page:
            page.article_role = "first_page"
        elif page.page_index == article.end_page:
            page.article_role = "last_page"
        else:
            page.article_role = "body"


def _same_label_context(label: str, page: JournalPageRecord, target_id: str) -> list[JournalBlockRecord]:
    if not label:
        return []
    result: list[JournalBlockRecord] = []
    normalized_label = re.sub(r"\s+", "", label.lower())
    for block in page.blocks:
        if block.block_id == target_id:
            continue
        text = clean_text(block.text)
        if not text:
            continue
        if normalized_label and normalized_label in re.sub(r"\s+", "", text.lower()):
            result.append(block)
    return result


def _nearest_captions(page: JournalPageRecord, target: JournalBlockRecord, caption_types: set[str]) -> list[JournalBlockRecord]:
    target_box = _box(target)
    if target_box is None:
        return []
    captions = [
        block
        for block in page.blocks
        if block.block_id != target.block_id
        and block.semantic_role in caption_types
        and clean_text(block.text)
        and _box(block) is not None
    ]
    scored: list[tuple[float, JournalBlockRecord]] = []
    for caption in captions:
        caption_box = _box(caption)
        if caption_box is None:
            continue
        horizontal_overlap = max(0.0, min(target_box[2], caption_box[2]) - max(target_box[0], caption_box[0]))
        overlap_ratio = horizontal_overlap / max(1.0, min(target_box[2] - target_box[0], caption_box[2] - caption_box[0]))
        if overlap_ratio < 0.2:
            continue
        vertical_gap = min(abs(caption_box[1] - target_box[3]), abs(target_box[1] - caption_box[3]))
        scored.append((vertical_gap, caption))
    return [caption for _, caption in sorted(scored, key=lambda item: item[0])[:2]]


def _build_subfigure_group_links(
    page: JournalPageRecord,
    cfg: dict[str, Any],
    output_dir: Path | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    figure_blocks = [
        block
        for block in page.blocks
        if block.block_type == "figure" and _subfigure_marker(block.text) is not None and _box(block) is not None
    ]
    if not figure_blocks:
        return [], set()

    max_per_crop = max(1, int(cfg.get("generation", {}).get("max_subfigures_per_crop", 2)))
    links: list[dict[str, Any]] = []
    grouped_ids: set[str] = set()
    pending_by_column: dict[str, list[JournalBlockRecord]] = {}

    def emit_group(label: str, members: list[JournalBlockRecord]) -> None:
        if len(members) < 2:
            return
        all_labels = [marker[0] for member in members if (marker := _subfigure_marker(member.text)) is not None]
        chunk_count = (len(members) + max_per_crop - 1) // max_per_crop
        label_key = _normalize_figure_label_key(label)
        for chunk_index in range(chunk_count):
            chunk = members[chunk_index * max_per_crop : (chunk_index + 1) * max_per_crop]
            if not chunk:
                continue
            target = chunk[0]
            chunk_labels = [marker[0] for member in chunk if (marker := _subfigure_marker(member.text)) is not None]
            group_id = f"{target.block_id}_{label_key.replace(':', '_')}_part{chunk_index + 1}"
            captions = _caption_blocks_for_subfigure_group(page, label, chunk)
            caption_ids = {caption.block_id for caption in captions}
            chunk_ids = {member.block_id for member in chunk}
            related = _same_label_context(label, page, target.block_id)
            related = [block for block in related if block.block_id not in caption_ids and block.block_id not in chunk_ids]
            evidence_blocks = [*chunk, *captions, *related[: int(cfg.get("generation", {}).get("max_related_context_blocks", 8))]]
            links.append(
                {
                    "target_block_id": target.block_id,
                    "target_type": target.block_type,
                    "object_label": label,
                    "caption_block_ids": [block.block_id for block in captions],
                    "related_context_block_ids": [block.block_id for block in related],
                    "evidence_block_ids": [block.block_id for block in evidence_blocks],
                    "context_policy": "subfigure_group_same_label_with_chunked_crop",
                    "is_subfigure_group": True,
                    "group_member_block_ids": [block.block_id for block in chunk],
                    "all_group_member_block_ids": [block.block_id for block in members],
                    "subfigure_labels": chunk_labels,
                    "all_subfigure_labels": all_labels,
                    "chunk_index": chunk_index + 1,
                    "chunk_count": chunk_count,
                    "target_image": _crop_subfigure_group_image(page, output_dir, chunk, cfg, group_id),
                    "caption_note": _caption_note_for_subfigure_chunk(label, chunk_labels, all_labels, chunk_count),
                }
            )
            grouped_ids.update(block.block_id for block in chunk)

    for block in sorted(figure_blocks, key=lambda item: (item.column, _top(item), _left(item), item.reading_order)):
        column = block.column or "unknown"
        pending = pending_by_column.setdefault(column, [])
        label = _figure_label_after_subfigure(block.text)
        if label:
            members = [*pending, block]
            emit_group(label, members)
            pending_by_column[column] = []
        else:
            pending.append(block)
            pending_by_column[column] = pending[-6:]
    return links, grouped_ids


def _link_visual_context(page: JournalPageRecord, cfg: dict[str, Any], output_dir: Path | None = None) -> None:
    visual_types = set(cfg["routing"].get("visual_block_types", []))
    caption_types = {"figure_caption", "table_caption"}
    links, grouped_visual_ids = _build_subfigure_group_links(page, cfg, output_dir)
    for target in [block for block in page.blocks if block.block_type in visual_types]:
        if target.block_id in grouped_visual_ids:
            continue
        captions = _nearest_captions(page, target, caption_types)
        label = ""
        for caption in captions:
            label = _caption_label(caption.text)
            if label:
                break
        if not label:
            label = _caption_label(target.text)
        related = _same_label_context(label, page, target.block_id)
        related = [block for block in related if block.block_id not in {caption.block_id for caption in captions}]
        evidence_blocks = [target, *captions, *related[: int(cfg["generation"].get("max_related_context_blocks", 8))]]
        links.append(
            {
                "target_block_id": target.block_id,
                "target_type": target.block_type,
                "object_label": label,
                "caption_block_ids": [block.block_id for block in captions],
                "related_context_block_ids": [block.block_id for block in related],
                "evidence_block_ids": [block.block_id for block in evidence_blocks],
                "context_policy": "same_page_nearest_caption_then_same_label_reference",
            }
        )
    page.figure_links = links
    page.semantic_links = list(links)


def _can_reuse_normalized(
    page: JournalPageRecord,
    image_map: dict[str, str],
    cfg: dict[str, Any],
    mineru_content_hash: str,
) -> bool:
    if page.mineru_content_hash != mineru_content_hash:
        return False
    if getattr(page, "normalizer_version", "") != NORMALIZER_VERSION:
        return False
    if not bool(cfg["runtime"].get("crop_blocks")) and any(block.image_path for block in page.blocks):
        return False
    if not _two_column_left_then_right_ok(page.blocks, page.column_mode):
        return False
    if not image_map:
        return True
    visual_types = set(cfg["routing"]["visual_block_types"])
    visual_blocks = [block for block in page.blocks if block.block_type in visual_types]
    if not visual_blocks:
        return True
    return any(block.extracted_image_path and block.text for block in visual_blocks)


def _build_page(
    journal: JournalRecord,
    output_dir: Path,
    page_no: int,
    items: list[dict[str, Any]],
    mineru_parse_path: str,
    image_map: dict[str, str],
    mineru_content_hash: str,
    total_pages: int,
    cfg: dict[str, Any],
) -> JournalPageRecord:
    blocks: list[JournalBlockRecord] = []
    for index, item in enumerate(items, start=1):
        raw_type = clean_text(item.get("type")) or cfg["block_types"]["unknown"]
        block_type = _block_type(raw_type, cfg)
        text = _item_text(item)
        caption_type = _caption_kind(text)
        if block_type == "text" and caption_type:
            block_type = caption_type
        block = JournalBlockRecord(
            block_id=f"p{page_no:03d}_b{index:03d}",
            block_type=block_type,
            bbox=_bbox(item),
            text=text,
            markdown=clean_text(item.get("markdown") or item.get("md_content")),
            reading_order=index,
            confidence=float(item.get("confidence") or item.get("score") or 0.0),
            extracted_image_path=_resolve_extracted_image(_item_image_ref(item), image_map),
        )
        blocks.append(block)

    page_image = cfg["paths"]["page_images"].format(page_no=page_no)
    width, height = _page_size(output_dir, page_image, blocks)
    layout_width, layout_height = _layout_size(blocks, width, height)
    for block in blocks:
        block.semantic_role = _semantic_role(block.block_type, block.text, _box(block), layout_height, page_no, cfg)
        if _looks_like_watermark_text(block.text, cfg):
            block.semantic_role = "watermark"
    column_mode = _assign_columns(blocks, layout_width, cfg)
    ordered = _recover_reading_order(blocks, column_mode)
    blocks = _sort_blocks_in_reading_order(ordered)
    raw_text = "\n".join(block.text for block in ordered if clean_text(block.text))
    page_type = _page_type(page_no, raw_text, blocks, column_mode, cfg)
    visible_text = "\n".join(block.text for block in ordered if clean_text(block.text) and not _skip_from_train_text(block, page_type))
    page = JournalPageRecord(
        journal_id=journal.journal_id,
        journal_name=journal.journal_name,
        source_pdf=journal.source_pdf,
        page_index=page_no,
        page_label=str(page_no),
        page_type=page_type,
        article_id=cfg["default_article_id"],
        article_title=cfg["default_article_title"],
        article_role="unknown",
        page_image=page_image,
        width=width,
        height=height,
        column_mode=column_mode,
        blocks=blocks,
        reading_order_blocks=[block.block_id for block in ordered],
        full_text=visible_text,
        visible_text=visible_text,
        titles=[asdict(block) for block in blocks if block.semantic_role in {"article_title", "section_heading"}],
        tables=[asdict(block) for block in blocks if block.block_type == "table"],
        figures=[asdict(block) for block in blocks if block.block_type == "figure"],
        formulas=[asdict(block) for block in blocks if block.block_type == "formula"],
        prev_page=page_no - 1 if page_no > 1 else None,
        next_page=page_no + 1 if page_no < total_pages else None,
        mineru_parse_path=mineru_parse_path,
        mineru_content_hash=mineru_content_hash,
        normalizer_version=NORMALIZER_VERSION,
    )
    if "two_column" in column_mode:
        page.uncertainty_notes.append("reading order reconstructed from coordinates; verify cross-column floating objects")
        visual_types = set(cfg["routing"].get("visual_block_types", []))
        if any(block.block_type in visual_types and block.column == "full_width" for block in blocks):
            page.uncertainty_notes.append(
                "cross-column visual blocks split two-column reading bands; read left band above barrier, then right band above barrier, then the barrier"
            )
    _link_visual_context(page, cfg, output_dir)
    page.normalized_page_path = cfg["paths"]["normalized_page"].format(page_no=page_no)
    return page


def normalize_pages(
    journal: JournalRecord,
    content: list[dict[str, Any]],
    mineru_parse_path: str,
    image_map: dict[str, str],
    cfg: dict[str, Any],
) -> NormalizeResult:
    output_dir = Path(journal.output_dir)
    grouped: dict[int, list[dict[str, Any]]] = {}
    mineru_content_hash = stable_json_hash(content)
    fallback_page = 1
    for item in content:
        page_no = _item_page(item, fallback_page)
        grouped.setdefault(page_no, []).append(item)
        fallback_page = page_no

    total_pages = max(journal.page_count, max(grouped.keys(), default=0))
    pages: list[JournalPageRecord] = []
    for page_no in range(1, total_pages + 1):
        relative = cfg["paths"]["normalized_page"].format(page_no=page_no)
        normalized_path = output_dir / relative
        if bool(cfg["runtime"].get("reuse_normalized")) and normalized_path.exists():
            cached_page = _page_from_dict(read_json(normalized_path))
            if _can_reuse_normalized(cached_page, image_map, cfg, mineru_content_hash):
                pages.append(cached_page)
                continue
        page = _build_page(
            journal,
            output_dir,
            page_no,
            grouped.get(page_no, []),
            mineru_parse_path,
            image_map,
            mineru_content_hash,
            total_pages,
            cfg,
        )
        pages.append(page)

    articles = _split_articles(pages, journal, cfg)
    _assign_articles(pages, articles, cfg)
    _reconstruct_paragraphs(pages)
    for page in pages:
        write_json(output_dir / page.normalized_page_path, asdict(page))
    write_jsonl(output_dir / cfg["paths"]["articles_manifest"], (asdict(article) for article in articles))
    return NormalizeResult(pages=pages, articles=articles)


