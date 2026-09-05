from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from ..core.io_utils import clean_text
from ..core.models import JournalArticleRecord, JournalBlockRecord, JournalPageRecord, JournalRecord, JournalSampleJob


def _enabled(enabled_tasks: dict[str, bool], task_type: str) -> bool:
    return bool(enabled_tasks.get(task_type, False))


def _block_payload(block: JournalBlockRecord) -> dict[str, Any]:
    return asdict(block)


def _page_payload(page: JournalPageRecord, *, include_blocks: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "page_index": page.page_index,
        "page_label": page.page_label,
        "page_type": page.page_type,
        "article_id": page.article_id,
        "article_title": page.article_title,
        "article_role": page.article_role,
        "page_image": page.page_image,
        "column_mode": page.column_mode,
        "reading_order_blocks": page.reading_order_blocks,
        "full_text": page.full_text,
        "visible_text": page.visible_text,
        "paragraphs": page.paragraphs,
        "figure_links": page.figure_links,
    }
    if include_blocks:
        payload["blocks"] = [asdict(block) for block in page.blocks]
    return payload


def _article_payload(article: JournalArticleRecord | None) -> dict[str, Any]:
    return asdict(article) if article else {}


def _article_by_id(articles: list[JournalArticleRecord]) -> dict[str, JournalArticleRecord]:
    return {article.article_id: article for article in articles}


def _is_low_value_page(page: JournalPageRecord, cfg: dict[str, Any]) -> bool:
    return page.page_type in set(cfg["routing"].get("low_value_page_types", []))


def _trainable_page(page: JournalPageRecord, cfg: dict[str, Any]) -> bool:
    return bool(clean_text(page.full_text or page.visible_text)) and not _is_low_value_page(page, cfg)


def _page_window_same_article(
    pages: list[JournalPageRecord],
    start_index: int,
    window_size: int,
) -> list[JournalPageRecord]:
    first = pages[start_index]
    result: list[JournalPageRecord] = []
    for page in pages[start_index : start_index + max(1, window_size)]:
        if page.article_id != first.article_id:
            break
        result.append(page)
    return result


def _window_text(page_window: list[JournalPageRecord]) -> str:
    return "\n\n".join(clean_text(page.full_text) for page in page_window if clean_text(page.full_text))


def _text_blocks(page: JournalPageRecord, text_types: set[str], min_chars: int) -> list[JournalBlockRecord]:
    return [block for block in page.blocks if block.block_type in text_types and len(clean_text(block.text)) >= min_chars]


def _section_heading_blocks(page: JournalPageRecord) -> list[JournalBlockRecord]:
    return [block for block in page.blocks if block.semantic_role == "section_heading" and clean_text(block.text)]


def _controlled_blocks(page: JournalPageRecord, heading: JournalBlockRecord, min_chars: int) -> list[JournalBlockRecord]:
    ordered = sorted(page.blocks, key=lambda item: item.reading_order or 9999)
    try:
        start = ordered.index(heading)
    except ValueError:
        return []
    controlled: list[JournalBlockRecord] = []
    for block in ordered[start + 1 :]:
        if block.semantic_role == "section_heading":
            break
        if block.semantic_role in {"journal_header", "footer", "page_number", "reference"}:
            continue
        if clean_text(block.text):
            controlled.append(block)
    if len(clean_text(" ".join(block.text for block in controlled))) < min_chars:
        return []
    return controlled


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def _figure_link_for(page: JournalPageRecord, block: JournalBlockRecord) -> dict[str, Any]:
    for link in page.figure_links:
        if link.get("target_block_id") == block.block_id:
            return link
    return {
        "target_block_id": block.block_id,
        "target_type": block.block_type,
        "caption_block_ids": [],
        "related_context_block_ids": [],
        "evidence_block_ids": [block.block_id],
        "context_policy": "target_block_only",
    }


def _grouped_non_target_visual_ids(page: JournalPageRecord) -> set[str]:
    skipped: set[str] = set()
    for link in page.figure_links:
        if not bool(link.get("is_subfigure_group")):
            continue
        target_id = str(link.get("target_block_id") or "")
        for block_id in link.get("group_member_block_ids", []):
            block_id = str(block_id)
            if block_id and block_id != target_id:
                skipped.add(block_id)
    return skipped


def _blocks_by_id(page: JournalPageRecord) -> dict[str, JournalBlockRecord]:
    return {block.block_id: block for block in page.blocks}


def _related_blocks_for_visual(page: JournalPageRecord, target: JournalBlockRecord, cfg: dict[str, Any]) -> list[JournalBlockRecord]:
    by_id = _blocks_by_id(page)
    link = _figure_link_for(page, target)
    evidence_ids = [str(item) for item in link.get("evidence_block_ids", []) if str(item)]
    blocks = [by_id[block_id] for block_id in evidence_ids if block_id in by_id]
    if not blocks or blocks[0].block_id != target.block_id:
        blocks = [target] + [block for block in blocks if block.block_id != target.block_id]
    max_blocks = int(cfg["generation"].get("max_related_context_blocks", 8))
    return blocks[: max(1, max_blocks)]


def _image_for_visual(page: JournalPageRecord, block: JournalBlockRecord, link: dict[str, Any] | None = None) -> str:
    if isinstance(link, dict) and clean_text(link.get("target_image")):
        return str(link.get("target_image"))
    return block.extracted_image_path or page.page_image


def _is_domain_corpus_noise_text(text: str, role: str, page: JournalPageRecord) -> bool:
    text = clean_text(text)
    if not text:
        return True
    if text.upper() == "OSID:":
        return True
    if text.startswith(("作者简介", "作者简介：", "作者简介:")):
        return True
    if re.match(r"^Abstract\s*:", text, flags=re.IGNORECASE):
        return page.page_index > 1
    if re.match(r"^Key\s*words?\s*:", text, flags=re.IGNORECASE) or re.match(r"^Keywords?\s*:", text, flags=re.IGNORECASE):
        return page.page_index > 1
    if role in {"abstract", "keywords"} and page.page_index > 1 and re.match(r"^[A-Za-z]", text):
        return True
    if re.search(r"[,，]\s*(19|20)\d{2}\.?$", text) and len(text) <= 90:
        if any(marker in text for marker in ("大学", "出版社", "研究院", "科技", "Society", "Press")):
            return True
    return False


def _domain_corpus_raw_text(page_window: list[JournalPageRecord], skip_roles: set[str]) -> str:
    pages_text: list[str] = []
    for page in page_window:
        lines: list[str] = []
        page_label = clean_text(page.page_label) or str(page.page_index)
        lines.append(f"[页码：{page_label}]")
        if page.paragraphs:
            for paragraph in page.paragraphs:
                role = str(paragraph.get("semantic_role") or "")
                if role in skip_roles:
                    continue
                text = clean_text(paragraph.get("text"))
                if _is_domain_corpus_noise_text(text, role, page):
                    continue
                if text:
                    lines.append(text)
        else:
            for block in sorted(page.blocks, key=lambda item: item.reading_order or 9999):
                if block.semantic_role in skip_roles:
                    continue
                text = clean_text(block.markdown or block.text)
                if _is_domain_corpus_noise_text(text, block.semantic_role, page):
                    continue
                if text:
                    lines.append(text)
        text = "\n".join(lines).strip()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _article_pages(article: JournalArticleRecord, pages: list[JournalPageRecord], cfg: dict[str, Any]) -> list[JournalPageRecord]:
    return [page for page in pages if page.article_id == article.article_id and _trainable_page(page, cfg)]


def _build_domain_corpus_jobs(
    journal: JournalRecord,
    pages: list[JournalPageRecord],
    articles: list[JournalArticleRecord],
    cfg: dict[str, Any],
) -> list[JournalSampleJob]:
    routing = cfg["routing"]
    min_chars = int(routing.get("min_domain_corpus_text_chars", routing.get("min_page_text_chars", 120)))
    target_chars = int(routing.get("domain_corpus_target_input_chars", 8000))
    max_pages = max(1, int(routing.get("domain_corpus_window", 4)))
    skip_roles = {"journal_header", "footer", "page_number", "reference", "unknown"}

    jobs: list[JournalSampleJob] = []
    article_map = _article_by_id(articles)
    article_ids = list(dict.fromkeys(page.article_id for page in pages if _trainable_page(page, cfg)))
    for article_id in article_ids:
        article = article_map.get(article_id)
        article_pages = [page for page in pages if page.article_id == article_id and _trainable_page(page, cfg)]
        window: list[JournalPageRecord] = []
        window_chars = 0

        def emit() -> None:
            nonlocal window, window_chars
            if not window:
                return
            raw_text = _domain_corpus_raw_text(window, skip_roles)
            if len(clean_text(raw_text)) >= min_chars:
                jobs.append(
                    JournalSampleJob(
                        "domain_knowledge_corpus",
                        journal,
                        window[0],
                        article=article,
                        page_window=list(window),
                        source={
                            "journal_name": journal.journal_name,
                            "article": _article_payload(article),
                            "raw_text": raw_text,
                            "page_window": [_page_payload(page) for page in window],
                        },
                    )
                )
            window = []
            window_chars = 0

        for page in article_pages:
            text = clean_text(page.full_text)
            if not text:
                continue
            window.append(page)
            window_chars += len(text)
            if window_chars >= target_chars or len(window) >= max_pages:
                emit()
        emit()
    return jobs


def build_sample_jobs(
    journal: JournalRecord,
    pages: list[JournalPageRecord],
    articles: list[JournalArticleRecord],
    cfg: dict[str, Any],
) -> list[JournalSampleJob]:
    jobs: list[JournalSampleJob] = []
    routing = cfg["routing"]
    enabled = routing["enabled_tasks"]
    article_map = _article_by_id(articles)
    text_types = set(routing.get("text_block_types", ["text", "paragraph", "list"]))
    visual_types = set(routing.get("visual_block_types", ["figure", "table", "formula"]))
    min_page_text = int(routing["min_page_text_chars"])
    min_block_text = int(routing["min_block_text_chars"])
    min_section_text = int(routing.get("min_section_text_chars", 220))
    cross_min_images = max(2, int(routing.get("cross_page_min_images", 2)))
    cross_max_images = max(cross_min_images, int(routing.get("cross_page_max_images", 3)))
    cross_window = min(max(cross_min_images, int(routing.get("cross_page_window", cross_max_images))), cross_max_images)
    article_window = int(routing.get("article_window", 8))
    method_keywords = list(routing.get("method_keywords", []))
    claim_keywords = list(routing.get("claim_keywords", []))
    conclusion_keywords = list(routing.get("conclusion_keywords", []))

    for index, page in enumerate(pages):
        page_text = clean_text(page.full_text)
        has_page_text = len(page_text) >= min_page_text
        article = article_map.get(page.article_id)

        if _enabled(enabled, "page_to_journal_layout_description") and page.page_type != "blank":
            jobs.append(
                JournalSampleJob(
                    "page_to_journal_layout_description",
                    journal,
                    page,
                    article=article,
                    images=[page.page_image],
                    source={"page": _page_payload(page), "article": _article_payload(article)},
                )
            )

        if _enabled(enabled, "article_metadata_extraction") and page.page_type == "article_first_page":
            jobs.append(
                JournalSampleJob(
                    "article_metadata_extraction",
                    journal,
                    page,
                    article=article,
                    images=[page.page_image],
                    source={"page": _page_payload(page), "article": _article_payload(article)},
                )
            )

        if _enabled(enabled, "two_column_reading_order_reconstruction") and (
            "two_column" in page.column_mode or page.figure_links or page.uncertainty_notes
        ):
            jobs.append(
                JournalSampleJob(
                    "two_column_reading_order_reconstruction",
                    journal,
                    page,
                    article=article,
                    images=[page.page_image],
                    source={"page": _page_payload(page), "article": _article_payload(article)},
                )
            )

        if _trainable_page(page, cfg):
            headings = _section_heading_blocks(page)
            if _enabled(enabled, "section_heading_scope_alignment"):
                for heading in headings:
                    controlled = _controlled_blocks(page, heading, min_block_text)
                    if not controlled:
                        continue
                    jobs.append(
                        JournalSampleJob(
                            "section_heading_scope_alignment",
                            journal,
                            page,
                            block=heading,
                            article=article,
                            blocks=[heading, *controlled],
                            source={
                                "heading": _block_payload(heading),
                                "controlled_blocks": [_block_payload(block) for block in controlled],
                                "page": _page_payload(page, include_blocks=False),
                                "article": _article_payload(article),
                            },
                        )
                    )

            if _enabled(enabled, "section_keypoint_summary"):
                for heading in headings:
                    controlled = _controlled_blocks(page, heading, min_section_text)
                    if not controlled:
                        continue
                    jobs.append(
                        JournalSampleJob(
                            "section_keypoint_summary",
                            journal,
                            page,
                            block=heading,
                            article=article,
                            blocks=[heading, *controlled],
                            source={
                                "section_title": heading.text,
                                "section_blocks": [_block_payload(block) for block in controlled],
                                "page": _page_payload(page, include_blocks=False),
                                "article": _article_payload(article),
                            },
                        )
                    )
                if not headings and has_page_text:
                    blocks = _text_blocks(page, text_types, min_block_text)
                    if len(clean_text(" ".join(block.text for block in blocks))) >= min_section_text:
                        jobs.append(
                            JournalSampleJob(
                                "section_keypoint_summary",
                                journal,
                                page,
                                article=article,
                                blocks=blocks,
                                source={
                                    "section_title": page.article_title or page.page_label,
                                    "section_blocks": [_block_payload(block) for block in blocks],
                                    "page": _page_payload(page, include_blocks=False),
                                    "article": _article_payload(article),
                                },
                            )
                        )

            if _enabled(enabled, "figure_table_formula_to_text"):
                grouped_non_targets = _grouped_non_target_visual_ids(page)
                for block in [item for item in page.blocks if item.block_type in visual_types]:
                    if block.block_id in grouped_non_targets:
                        continue
                    figure_link = _figure_link_for(page, block)
                    related_blocks = _related_blocks_for_visual(page, block, cfg)
                    jobs.append(
                        JournalSampleJob(
                            "figure_table_formula_to_text",
                            journal,
                            page,
                            block=block,
                            article=article,
                            blocks=related_blocks,
                            images=[_image_for_visual(page, block, figure_link)],
                            source={
                                "target_block": _block_payload(block),
                                "visual_context_blocks": [_block_payload(item) for item in related_blocks],
                                "figure_link": figure_link,
                                "context_binding_rule": "only target visual, nearest caption, and same-label references are supplied",
                                "page": _page_payload(page, include_blocks=False),
                                "article": _article_payload(article),
                            },
                        )
                    )

            if _enabled(enabled, "method_experiment_condition_extraction") and _contains_any(page_text, method_keywords):
                jobs.append(
                    JournalSampleJob(
                        "method_experiment_condition_extraction",
                        journal,
                        page,
                        article=article,
                        blocks=_text_blocks(page, text_types, min_block_text),
                        source={"page": _page_payload(page), "article": _article_payload(article), "matched_keywords": method_keywords},
                    )
                )

            if _enabled(enabled, "evidence_to_claim_chain") and _contains_any(page_text, claim_keywords):
                images = [page.page_image]
                jobs.append(
                    JournalSampleJob(
                        "evidence_to_claim_chain",
                        journal,
                        page,
                        article=article,
                        images=images,
                        source={"page": _page_payload(page), "article": _article_payload(article), "matched_keywords": claim_keywords},
                    )
                )

        if _enabled(enabled, "cross_page_article_context") and _trainable_page(page, cfg):
            page_window = [
                item
                for item in _page_window_same_article(pages, index, cross_window)
                if _trainable_page(item, cfg)
            ][:cross_max_images]
            if len(page_window) >= cross_min_images and len(clean_text(_window_text(page_window))) >= min_page_text:
                jobs.append(
                    JournalSampleJob(
                        "cross_page_article_context",
                        journal,
                        page,
                        article=article,
                        page_window=page_window,
                        images=[item.page_image for item in page_window],
                        source={
                            "page_window": [_page_payload(item) for item in page_window],
                            "article": _article_payload(article),
                            "image_count_policy": f"{cross_min_images}-{cross_max_images} page images",
                        },
                    )
                )

    if _enabled(enabled, "article_contribution_conclusion"):
        for article in articles:
            article_pages = _article_pages(article, pages, cfg)[:article_window]
            if not article_pages:
                continue
            article_text = clean_text(_window_text(article_pages))
            if len(article_text) >= int(routing.get("min_article_text_chars", 800)) or _contains_any(article_text, conclusion_keywords):
                jobs.append(
                    JournalSampleJob(
                        "article_contribution_conclusion",
                        journal,
                        article_pages[0],
                        article=article,
                        page_window=article_pages,
                        source={"article": _article_payload(article), "page_window": [_page_payload(page) for page in article_pages]},
                    )
                )

    if _enabled(enabled, "domain_knowledge_corpus"):
        jobs.extend(_build_domain_corpus_jobs(journal, pages, articles, cfg))

    return jobs
