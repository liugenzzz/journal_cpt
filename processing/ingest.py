from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.io_utils import file_sha256, safe_name, utc_now
from ..core.models import JournalRecord


def page_count(pdf_path: Path, cfg: dict[str, Any]) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as document:
                return int(document.page_count)
        except Exception:
            pattern = str(cfg["pdf_page_regex"]).encode("ascii")
            return len(re.findall(pattern, pdf_path.read_bytes()))


def _journal_id_for(pdf_path: Path, journal_name: str, cfg: dict[str, Any]) -> str:
    configured_ids = cfg.get("journal_ids") if isinstance(cfg.get("journal_ids"), dict) else {}
    configured = configured_ids.get(pdf_path.name) or configured_ids.get(journal_name) or configured_ids.get(f"{journal_name}.pdf")
    return configured or safe_name(journal_name, str(cfg.get("unknown_journal_id", "untitled_journal")))


def _issue_fields(name: str) -> dict[str, str]:
    patterns = [
        r"(?P<year>20\d{2})\s*年\s*第?\s*(?P<volume>\d+)\s*卷\s*第?\s*(?P<issue>\d+)\s*期",
        r"(?P<year>20\d{2})[^\d]{0,3}(?P<volume>\d+)[^\d]{0,3}(?P<issue>\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return {key: value or "" for key, value in match.groupdict().items()}
    year_match = re.search(r"(20\d{2})", name)
    return {"year": year_match.group(1) if year_match else "", "volume": "", "issue": ""}


def build_journal_record(
    *,
    pdf_path: Path,
    output_root: Path,
    cfg: dict[str, Any],
    journal_name: str | None = None,
    journal_id: str | None = None,
) -> JournalRecord:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    resolved_name = (journal_name or pdf_path.stem).strip() or pdf_path.stem
    resolved_id = (journal_id or _journal_id_for(pdf_path, resolved_name, cfg)).strip()
    output_dir = output_root / safe_name(resolved_id, str(cfg.get("unknown_journal_id", "untitled_journal")))
    issue = _issue_fields(resolved_name)
    return JournalRecord(
        journal_id=resolved_id,
        journal_name=resolved_name,
        source_pdf=str(pdf_path),
        output_dir=str(output_dir),
        page_count=page_count(pdf_path, cfg),
        file_size=pdf_path.stat().st_size,
        file_hash=file_sha256(pdf_path, int(cfg["hash_chunk_size"])),
        status=cfg["statuses"]["ready"],
        created_at=utc_now(cfg["timestamp_format"]),
        pipeline_version=cfg["pipeline_version"],
        year=issue["year"],
        volume=issue["volume"],
        issue=issue["issue"],
    )


def scan_journals(input_dir: Path, output_root: Path, cfg: dict[str, Any]) -> list[JournalRecord]:
    pattern = f"**/*{cfg['pdf_suffix']}" if bool(cfg["runtime"]["recursive"]) else f"*{cfg['pdf_suffix']}"
    journals: list[JournalRecord] = []
    for pdf_path in sorted(input_dir.glob(pattern)):
        if not pdf_path.is_file():
            continue
        journals.append(build_journal_record(pdf_path=pdf_path, output_root=output_root, cfg=cfg))
    return journals


def scan_input_journals(input_journals: list[dict[str, str]], output_root: Path, cfg: dict[str, Any]) -> list[JournalRecord]:
    journals: list[JournalRecord] = []
    for item in input_journals:
        raw_path = item.get("path") or item.get("source_pdf") or item.get("address") or item.get("url")
        if not raw_path:
            raise ValueError(f"Missing journal path in input journal item: {item}")
        journals.append(
            build_journal_record(
                pdf_path=Path(raw_path),
                output_root=output_root,
                cfg=cfg,
                journal_name=item.get("journal_name") or item.get("name"),
                journal_id=item.get("journal_id"),
            )
        )
    return journals

