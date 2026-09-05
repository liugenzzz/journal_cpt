from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject

WATERMARK_FORM_NAMES = {"/Fm0"}
WATERMARK_IMAGE_SIZES = {(475, 272)}
MIN_PAGE_COVERAGE = 0.60


def operator_text(operator: Any) -> str:
    if isinstance(operator, bytes):
        return operator.decode("latin1")
    return str(operator)


def resolve_pdf_object(obj: Any) -> Any:
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


def pdf_dict(obj: Any) -> Any:
    resolved = resolve_pdf_object(obj)
    get_value = getattr(resolved, "get", None)
    return resolved if callable(get_value) else {}


def xobject_dict(page: Any) -> Any:
    resources = pdf_dict(page.get("/Resources"))
    return pdf_dict(resources.get("/XObject"))


def form_has_watermark_image(form: Any) -> bool:
    resources = pdf_dict(form.get("/Resources"))
    xobjects = pdf_dict(resources.get("/XObject"))
    if not xobjects:
        return False
    for nested_ref in xobjects.values():
        nested = pdf_dict(nested_ref)
        if not nested:
            continue
        if str(nested.get("/Subtype")) != "/Image":
            continue
        size = (int(nested.get("/Width") or 0), int(nested.get("/Height") or 0))
        if size in WATERMARK_IMAGE_SIZES:
            return True
    return False


def is_watermark_form(name: str, form: Any) -> bool:
    if name in WATERMARK_FORM_NAMES:
        return True
    if form.get("/OC") is not None:
        return True
    return form_has_watermark_image(form)


def page_watermark_forms(reader: PdfReader, page: Any) -> list[str]:
    xobjects = xobject_dict(page)
    if not xobjects:
        return []
    content = page.get_contents()
    if content is None:
        return []
    stream = ContentStream(content, reader)
    result: list[str] = []
    for operands, operator in stream.operations:
        if operator_text(operator) != "Do" or not operands:
            continue
        name = str(operands[0])
        if name not in xobjects:
            continue
        form = pdf_dict(xobjects[name])
        if not form:
            continue
        if str(form.get("/Subtype")) == "/Form" and is_watermark_form(name, form):
            result.append(name)
    return result


def repeated_watermark_forms(reader: PdfReader) -> list[str]:
    counts: Counter[str] = Counter()
    for page in reader.pages:
        counts.update(set(page_watermark_forms(reader, page)))
    min_pages = max(1, round(len(reader.pages) * MIN_PAGE_COVERAGE))
    return sorted(name for name, count in counts.items() if count >= min_pages)


def clean_pdf(source_pdf: Path, output_pdf: Path) -> dict[str, Any]:
    reader = PdfReader(str(source_pdf))
    candidates = repeated_watermark_forms(reader)
    writer = PdfWriter()
    removed = 0
    for page in reader.pages:
        content = page.get_contents()
        if content is None or not candidates:
            writer.add_page(page)
            continue
        stream = ContentStream(content, reader)
        filtered = []
        for operands, operator in stream.operations:
            if operator_text(operator) == "Do" and operands and str(operands[0]) in candidates:
                removed += 1
                continue
            filtered.append((operands, operator))
        stream.operations = filtered
        page[NameObject("/Contents")] = stream
        writer.add_page(page)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)
    return {
        "source_pdf": str(source_pdf),
        "output_pdf": str(output_pdf),
        "page_count": len(reader.pages),
        "candidate_forms": candidates,
        "removed_invocations": removed,
        "cleaned": removed > 0,
    }


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python clean_watermarked_pdfs.py OUTPUT_DIR PDF [PDF ...]", file=sys.stderr)
        return 2
    output_dir = Path(sys.argv[1])
    reports = []
    for raw_pdf in sys.argv[2:]:
        source_pdf = Path(raw_pdf)
        target = output_dir / f"{source_pdf.stem}_cleaned.pdf"
        report = clean_pdf(source_pdf, target)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
    report_path = output_dir / "watermark_cleaning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

