from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JournalRecord:
    journal_id: str
    journal_name: str
    source_pdf: str
    output_dir: str
    page_count: int
    file_size: int
    file_hash: str
    status: str
    created_at: str
    pipeline_version: str
    year: str = ""
    volume: str = ""
    issue: str = ""


@dataclass
class JournalBlockRecord:
    block_id: str
    block_type: str
    semantic_role: str = "unknown"
    bbox: list[float] = field(default_factory=list)
    text: str = ""
    markdown: str = ""
    reading_order: int = 0
    column: str = "unknown"
    span_kind: str = "single_column"
    confidence: float = 0.0
    image_path: str = ""
    extracted_image_path: str = ""


@dataclass
class JournalArticleRecord:
    journal_id: str
    article_id: str
    article_title: str
    source_pdf: str
    start_page: int
    end_page: int
    page_indices: list[int] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    journal_name: str = ""
    year: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class JournalPageRecord:
    journal_id: str
    journal_name: str
    source_pdf: str
    page_index: int
    page_label: str
    page_type: str
    article_id: str
    article_title: str
    article_role: str
    page_image: str
    width: int
    height: int
    column_mode: str
    blocks: list[JournalBlockRecord] = field(default_factory=list)
    reading_order_blocks: list[str] = field(default_factory=list)
    full_text: str = ""
    visible_text: str = ""
    paragraphs: list[dict[str, Any]] = field(default_factory=list)
    titles: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    semantic_links: list[dict[str, Any]] = field(default_factory=list)
    figure_links: list[dict[str, Any]] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    prev_page: int | None = None
    next_page: int | None = None
    mineru_parse_path: str = ""
    mineru_content_hash: str = ""
    normalized_page_path: str = ""
    normalizer_version: str = ""


@dataclass(frozen=True)
class JournalSampleJob:
    task_type: str
    journal: JournalRecord
    page: JournalPageRecord
    block: JournalBlockRecord | None = None
    article: JournalArticleRecord | None = None
    page_window: list[JournalPageRecord] = field(default_factory=list)
    blocks: list[JournalBlockRecord] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizeResult:
    pages: list[JournalPageRecord]
    articles: list[JournalArticleRecord]


@dataclass(frozen=True)
class PipelineOptions:
    config_path: Path | None = None
    input_dir: Path | None = None
    input_journals: list[dict[str, str]] | None = None
    output_root: Path | None = None
    reuse_mineru: bool | None = None
    skip_vlm: bool | None = None
    journal_workers: int | None = None
    max_workers: int | None = None
    page_workers: int | None = None
    crop_workers: int | None = None
    mineru_workers: int | None = None
    mineru_retry_count: int | None = None
    mineru_min_page_coverage: float | None = None
    reuse_normalized: bool | None = None
    reuse_samples: bool | None = None
    reuse_exports: bool | None = None
    force_rebuild: bool | None = None
    min_page_text_chars: int | None = None
    min_block_text_chars: int | None = None
    task_types: list[str] | None = None
    limit_journals: int | None = None
    recursive: bool | None = None
    log_level: str | None = None
    progress: bool | None = None
