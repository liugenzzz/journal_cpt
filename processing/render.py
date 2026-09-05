from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import append_jsonl
from ..core.models import JournalRecord


def _render_page_worker(args: tuple[str, str, dict[str, Any], int, dict[str, Any]]) -> tuple[str, dict[str, Any] | None, str, int]:
    source_pdf, output_dir_text, cfg, page_index, journal_payload = args
    page_no = page_index + 1
    try:
        import fitz  # type: ignore
    except ImportError:
        return "", None, "missing_fitz", page_no

    output_dir = Path(output_dir_text)
    relative = cfg["paths"]["page_images"].format(page_no=page_no)
    target = output_dir / relative
    if target.exists() and not bool(cfg["runtime"].get("rerender_pages")):
        return relative, None, "skipped", page_no

    scale = float(cfg["render"]["dpi"]) / 72.0
    target.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source_pdf) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pixmap.save(str(target))
    return relative, {**journal_payload, "page_index": page_no, "page_label": str(page_no), "page_image": relative}, "rendered", page_no


def render_pages(journal: JournalRecord, cfg: dict[str, Any]) -> list[str]:
    output_dir = Path(journal.output_dir)
    logger = logging.getLogger(f"{cfg['logger_name']}.{journal.journal_id}")
    rendered: list[str] = []
    if not cfg["runtime"]["render_pages"]:
        logger.info("render_pages disabled journal_id=%s", journal.journal_id)
        return rendered
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.warning("render_pages skipped journal_id=%s reason=missing_pymupdf", journal.journal_id)
        return rendered

    with fitz.open(journal.source_pdf) as document:
        page_count = document.page_count

    page_workers = max(1, int(cfg["runtime"].get("page_workers", 1)))
    jobs = [(journal.source_pdf, str(output_dir), cfg, page_index, asdict(journal)) for page_index in range(page_count)]
    page_dir = (output_dir / Path(cfg["paths"]["page_images"].format(page_no=1)).parent).resolve()
    logger.info(
        "render_pages start journal_id=%s pages=%s workers=%s output_dir=%s",
        journal.journal_id,
        page_count,
        page_workers,
        page_dir,
    )
    if page_workers == 1 or len(jobs) <= 1:
        results = [_render_page_worker(job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=page_workers) as executor:
            futures = [executor.submit(_render_page_worker, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    rendered_count = 0
    skipped_count = 0
    for relative, manifest_row, status, _ in sorted(results, key=lambda item: item[3]):
        if relative:
            rendered.append(relative)
        if status == "rendered":
            rendered_count += 1
        elif status == "skipped":
            skipped_count += 1
        if manifest_row is not None:
            append_jsonl(output_dir / cfg["paths"]["pages_manifest"], manifest_row)
    logger.info(
        "render_pages completed journal_id=%s pages=%s rendered=%s skipped=%s output_dir=%s",
        journal.journal_id,
        page_count,
        rendered_count,
        skipped_count,
        page_dir,
    )
    return rendered



