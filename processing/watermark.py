from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.io_utils import file_sha256, write_json
from ..core.models import JournalRecord


@dataclass(frozen=True)
class WatermarkCleanResult:
    source_pdf: str
    cleaned_pdf: str
    cleaned: bool
    candidate_names: list[str]
    removed_invocations: int
    page_count: int
    content_hash: str


def _watermark_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("watermark", {})


def watermark_cleaning_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_watermark_cfg(cfg).get("enabled", True))


def _operator_text(operator: Any) -> str:
    if isinstance(operator, bytes):
        return operator.decode("latin1")
    return str(operator)


def _resolve_pdf_object(obj: Any) -> Any:
    for _ in range(8):
        if obj is None:
            return None
        get_object = getattr(obj, "get_object", None)
        if not callable(get_object):
            return obj
        try:
            resolved = get_object()
        except Exception:
            return None
        if resolved is obj:
            return obj
        obj = resolved
    return obj


def _pdf_dict(obj: Any) -> Any:
    resolved = _resolve_pdf_object(obj)
    get_value = getattr(resolved, "get", None)
    return resolved if callable(get_value) else {}


def _xobject_dict(page: Any) -> Any:
    resources = _pdf_dict(page.get("/Resources"))
    return _pdf_dict(resources.get("/XObject"))


def _form_has_configured_image(form: Any, cfg: dict[str, Any]) -> bool:
    expected_sizes = {
        (int(item[0]), int(item[1]))
        for item in _watermark_cfg(cfg).get("image_sizes", [])
        if isinstance(item, (list, tuple)) and len(item) == 2
    }
    if not expected_sizes:
        return False
    resources = _pdf_dict(form.get("/Resources"))
    xobjects = _pdf_dict(resources.get("/XObject"))
    if not xobjects:
        return False
    for nested_ref in xobjects.values():
        nested = _pdf_dict(nested_ref)
        if not nested:
            continue
        if str(nested.get("/Subtype")) != "/Image":
            continue
        size = (int(nested.get("/Width") or 0), int(nested.get("/Height") or 0))
        if size in expected_sizes:
            return True
    return False


def _is_candidate_form(name: str, form: Any, cfg: dict[str, Any]) -> bool:
    configured_names = {str(item) for item in _watermark_cfg(cfg).get("form_names", [])}
    if configured_names and name in configured_names:
        return True
    if form.get("/OC") is not None:
        return True
    return _form_has_configured_image(form, cfg)


def _page_form_invocations(reader: Any, page: Any, cfg: dict[str, Any]) -> list[str]:
    from pypdf.generic import ContentStream  # type: ignore

    xobjects = _xobject_dict(page)
    if not xobjects:
        return []
    content = page.get_contents()
    if content is None:
        return []
    result: list[str] = []
    stream = ContentStream(content, reader)
    for operands, operator in stream.operations:
        if _operator_text(operator) != "Do" or not operands:
            continue
        name = str(operands[0])
        if name not in xobjects:
            continue
        form = _pdf_dict(xobjects[name])
        if not form:
            continue
        if str(form.get("/Subtype")) == "/Form" and _is_candidate_form(name, form, cfg):
            result.append(name)
    return result


def _find_repeated_watermark_forms(reader: Any, cfg: dict[str, Any]) -> list[str]:
    page_count = len(reader.pages)
    if page_count <= 0:
        return []
    counts: dict[str, int] = {}
    for page in reader.pages:
        seen_on_page = set(_page_form_invocations(reader, page, cfg))
        for name in seen_on_page:
            counts[name] = counts.get(name, 0) + 1
    min_page_coverage = float(_watermark_cfg(cfg).get("min_page_coverage", 0.6) or 0.6)
    min_pages = max(1, int(round(page_count * min_page_coverage)))
    return sorted(name for name, count in counts.items() if count >= min_pages)


def _remove_form_invocations(reader: Any, writer: Any, names: set[str]) -> int:
    from pypdf.generic import ContentStream, NameObject  # type: ignore

    removed = 0
    for page in reader.pages:
        content = page.get_contents()
        if content is None:
            writer.add_page(page)
            continue
        stream = ContentStream(content, reader)
        filtered = []
        for operands, operator in stream.operations:
            if _operator_text(operator) == "Do" and operands and str(operands[0]) in names:
                removed += 1
                continue
            filtered.append((operands, operator))
        stream.operations = filtered
        page[NameObject("/Contents")] = stream
        writer.add_page(page)
    return removed


def _output_paths(journal: JournalRecord, cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(journal.output_dir)
    cleaned_template = _watermark_cfg(cfg).get("cleaned_pdf_path", "preprocessed/{journal_id}_cleaned.pdf")
    report_template = _watermark_cfg(cfg).get("report_path", "preprocessed/watermark_cleaning.json")
    return (
        output_dir / str(cleaned_template).format(journal_id=journal.journal_id),
        output_dir / str(report_template).format(journal_id=journal.journal_id),
    )


def clean_pdf_watermarks(journal: JournalRecord, cfg: dict[str, Any]) -> WatermarkCleanResult:
    source_pdf = Path(journal.source_pdf)
    cleaned_pdf, report_path = _output_paths(journal, cfg)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not watermark_cleaning_enabled(cfg):
        result = WatermarkCleanResult(
            source_pdf=str(source_pdf),
            cleaned_pdf=str(source_pdf),
            cleaned=False,
            candidate_names=[],
            removed_invocations=0,
            page_count=journal.page_count,
            content_hash=journal.file_hash,
        )
        write_json(report_path, asdict(result))
        return result

    from pypdf import PdfReader, PdfWriter  # type: ignore

    reader = PdfReader(str(source_pdf))
    candidate_names = _find_repeated_watermark_forms(reader, cfg)
    if not candidate_names:
        result = WatermarkCleanResult(
            source_pdf=str(source_pdf),
            cleaned_pdf=str(source_pdf),
            cleaned=False,
            candidate_names=[],
            removed_invocations=0,
            page_count=len(reader.pages),
            content_hash=journal.file_hash,
        )
        write_json(report_path, asdict(result))
        return result

    cleaned_pdf.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    removed = _remove_form_invocations(reader, writer, set(candidate_names))
    with cleaned_pdf.open("wb") as handle:
        writer.write(handle)

    result = WatermarkCleanResult(
        source_pdf=str(source_pdf),
        cleaned_pdf=str(cleaned_pdf),
        cleaned=removed > 0,
        candidate_names=candidate_names,
        removed_invocations=removed,
        page_count=len(reader.pages),
        content_hash=file_sha256(cleaned_pdf, int(cfg["hash_chunk_size"])) if removed > 0 else journal.file_hash,
    )
    write_json(report_path, asdict(result))
    return result

