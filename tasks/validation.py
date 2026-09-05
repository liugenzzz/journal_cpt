from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text
from ..processing.image_quality import is_meaningful_image_file


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _source_pages(sample: dict[str, Any]) -> list[Any]:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    pages = evidence.get("source_pages")
    return pages if isinstance(pages, list) else []


def _valid_images(sample: dict[str, Any], output_dir: Path) -> bool:
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    return all((output_dir / str(image)).exists() for image in images)


def _meaningful_images(sample: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> bool:
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    for image in images:
        path = output_dir / str(image)
        if is_meaningful_image_file(path, cfg):
            continue
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as opened:
                width, height = opened.size
            if width >= 200 and height >= 200 and path.stat().st_size >= 5000:
                continue
            if width >= 28 and height >= 32 and path.stat().st_size >= 500:
                continue
        except Exception:
            if path.exists() and path.is_file() and path.stat().st_size >= 5000:
                continue
            return False
        return False
    return True


def _required_fields_present(sample: dict[str, Any], cfg: dict[str, Any]) -> bool:
    task_type = str(sample.get("task_type") or "")
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    required = cfg.get("validation", {}).get("required_output_fields", {}).get(task_type, [])
    return all(_has_value(output.get(field)) or _has_value(evidence.get(field)) for field in required)


def _pt_task_types(cfg: dict[str, Any]) -> set[str]:
    return {str(item) for item in cfg.get("export_formats", {}).get("pt", []) if str(item)}


def _alpaca_task_types(cfg: dict[str, Any]) -> set[str]:
    return {str(item) for item in cfg.get("export_formats", {}).get("alpaca", []) if str(item)}


def _sharegpt_task_types(cfg: dict[str, Any]) -> set[str]:
    return {str(item) for item in cfg.get("export_formats", {}).get("sharegpt", []) if str(item)}


def _task_format(task_type: str, cfg: dict[str, Any]) -> str:
    if task_type in _pt_task_types(cfg):
        return "pt"
    if task_type in _sharegpt_task_types(cfg):
        return "sharegpt"
    if task_type in _alpaca_task_types(cfg):
        return "alpaca"
    return "unknown"


def _pt_text(sample: dict[str, Any]) -> str:
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    return str(output.get("text") or sample.get("text") or "").strip()


def _low_value_page(sample: dict[str, Any], cfg: dict[str, Any]) -> bool:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    page_type = str(metadata.get("page_type") or "")
    return page_type in set(cfg.get("validation", {}).get("low_value_page_types", []))


def _figure_context_is_bound(sample: dict[str, Any]) -> bool:
    if str(sample.get("task_type") or "") != "figure_table_formula_to_text":
        return True
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    binding = evidence.get("figure_context_binding") if isinstance(evidence.get("figure_context_binding"), dict) else {}
    target_block_id = str(binding.get("target_block_id") or "")
    metadata_block_id = str(metadata.get("block_id") or "")
    if not target_block_id or target_block_id != metadata_block_id:
        return False
    evidence_ids = evidence.get("evidence_block_ids")
    if not isinstance(evidence_ids, list) or target_block_id not in {str(item) for item in evidence_ids}:
        return False
    rule = str(evidence.get("context_binding_rule") or "")
    if "target" not in rule and "same" not in rule:
        return False
    return True


def _caption_title_fragments(sample: dict[str, Any]) -> list[str]:
    payload = sample.get("input_payload") if isinstance(sample.get("input_payload"), dict) else {}
    fragments: list[str] = []
    candidates = [payload.get("caption_text")]
    target_block = payload.get("target_block")
    if isinstance(target_block, dict):
        candidates.append(target_block.get("text"))
    for value in candidates:
        text = clean_text(value)
        if not text:
            continue
        text = re.sub(
            r"^(图\s*[0-9一二三四五六七八九十]+|表\s*[0-9一二三四五六七八九十]+|"
            r"式\s*[（(]?\s*[0-9]+[）)]?|Fig\.?\s*[0-9]+|Figure\s*[0-9]+|Table\s*[0-9]+)"
            r"\s*[-—:：.．、]?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip(" -—:：.．、")
        if len(text) >= 4 and text not in fragments:
            fragments.append(text)
    return fragments


def _figure_instruction_is_generic(sample: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if str(sample.get("task_type") or "") != "figure_table_formula_to_text":
        return True
    instruction = clean_text(sample.get("instruction") or sample.get("question") or "")
    if not instruction:
        return False
    patterns = cfg.get("validation", {}).get("figure_instruction_forbidden_patterns", [])
    for pattern in patterns:
        if re.search(str(pattern), instruction, flags=re.IGNORECASE):
            return False
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    article_title = clean_text(metadata.get("article_title"))
    if article_title and article_title in instruction:
        return False
    for fragment in _caption_title_fragments(sample):
        if fragment and fragment in instruction:
            return False
    return True


def _cross_page_images_ok(sample: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if str(sample.get("task_type") or "") != "cross_page_article_context":
        return True
    routing = cfg.get("routing", {})
    min_images = max(2, int(routing.get("cross_page_min_images", 2)))
    max_images = max(min_images, int(routing.get("cross_page_max_images", 3)))
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    return min_images <= len(images) <= max_images


def _pt_payload_is_pure(sample: dict[str, Any]) -> bool:
    if str(sample.get("task_type") or "") != "domain_knowledge_corpus":
        return True
    record = {"text": _pt_text(sample)}
    return set(record.keys()) == {"text"} and bool(record["text"])


def _quality_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    quality = cfg.get("validation", {}).get("quality_dimensions", {})
    return quality if isinstance(quality, dict) else {}


def _quality_result(passed: bool, score: float, reasons: list[str]) -> dict[str, Any]:
    return {"passed": bool(passed), "score": round(max(0.0, min(1.0, float(score))), 3), "reasons": reasons}


def _question_text(sample: dict[str, Any]) -> str:
    return clean_text(sample.get("instruction") or sample.get("question") or "")


def _flatten_text(value: Any, limit: int = 80) -> list[str]:
    texts: list[str] = []
    if isinstance(value, str):
        text = clean_text(value)
        if text:
            texts.append(text)
    elif isinstance(value, dict):
        for child in value.values():
            texts.extend(_flatten_text(child, limit))
            if len(texts) >= limit:
                break
    elif isinstance(value, list):
        for child in value:
            texts.extend(_flatten_text(child, limit))
            if len(texts) >= limit:
                break
    return texts[:limit]


def _answer_text(sample: dict[str, Any]) -> str:
    if str(sample.get("task_type") or "") == "domain_knowledge_corpus":
        return _pt_text(sample)
    answer = clean_text(sample.get("answer"))
    if answer:
        return answer
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    output_text = clean_text(" ".join(_flatten_text(output)))
    if output_text:
        return output_text
    return ""


def _contains_mojibake(text: str, cfg: dict[str, Any]) -> bool:
    quality = _quality_cfg(cfg)
    for pattern in quality.get("mojibake_patterns", []):
        if re.search(str(pattern), text):
            return True
    return False


def _symbol_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    normal = re.findall(r"[\u4e00-\u9fffA-Za-z0-9，。；：？！、,.!?;:()\[\]{}<>《》“”\"'/%+\-=—_·&~]", compact)
    return 1.0 - (len(normal) / len(compact))


def _has_repeated_char_run(text: str, cfg: dict[str, Any]) -> bool:
    quality = _quality_cfg(cfg)
    max_run = int(quality.get("max_repeated_char_run", 8))
    return bool(re.search(r"(.)\1{" + str(max_run) + r",}", text))


def _extract_terms(text: str, limit: int = 120) -> list[str]:
    text = clean_text(text)
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_/\-]{1,30}|[\u4e00-\u9fffA-Za-z0-9]{2,18}", text)
    stopwords = {
        "请根据",
        "根据",
        "进行",
        "内容",
        "信息",
        "回答",
        "问题",
        "图片",
        "图像",
        "页面",
        "该页面",
        "该论文",
        "期刊",
        "论文",
        "任务",
        "输出",
        "字段",
        "说明",
        "分析",
        "描述",
    }
    terms: list[str] = []
    for term in raw_terms:
        term = term.strip(" ：:;；,，。、()（）[]【】")
        if len(term) < 2 or term in stopwords:
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _overlap_count(first: str, second: str) -> int:
    first_terms = set(_extract_terms(first))
    second_terms = set(_extract_terms(second))
    return len(first_terms & second_terms)


def _text_format_quality(sample: dict[str, Any], task_format: str, cfg: dict[str, Any]) -> dict[str, Any]:
    quality = _quality_cfg(cfg)
    reasons: list[str] = []
    texts = [_answer_text(sample)] if task_format == "pt" else [_question_text(sample), _answer_text(sample)]
    labels = ["text"] if task_format == "pt" else ["question", "answer"]
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    min_question = int(cfg.get("validation", {}).get("min_question_chars", 4))
    min_answer = int(cfg.get("validation", {}).get("min_answer_chars", 8))
    score = 1.0
    for label, text in zip(labels, texts):
        if not text:
            reasons.append(f"{label} is empty")
            score -= 0.45
            continue
        if label == "question" and len(text) < min_question:
            reasons.append("question is too short")
            score -= 0.25
        if label in {"answer", "text"} and len(text) < min_answer:
            reasons.append(f"{label} is too short")
            score -= 0.25
        if _contains_mojibake(text, cfg):
            reasons.append(f"{label} contains mojibake")
            score -= 0.45
        if _has_repeated_char_run(text, cfg):
            reasons.append(f"{label} contains repeated-character noise")
            score -= 0.25
        symbol_ratio = _symbol_ratio(text)
        if symbol_ratio > float(quality.get("max_symbol_ratio", 0.32)):
            reasons.append(f"{label} symbol ratio is high ({symbol_ratio:.2f})")
            score -= 0.25
        if "```" in text:
            reasons.append(f"{label} contains markdown code fence")
            score -= 0.15
        if label == "question" and text.lstrip().startswith(("{", "[")):
            reasons.append("question looks like raw JSON")
            score -= 0.25
        if task_format in {"sharegpt", "alpaca"} and label == "answer":
            if text.lstrip().startswith(("{", "[")):
                reasons.append(f"{task_format} answer looks like raw JSON")
                score -= 0.35
            if re.search(r"\b(input_payload|output_payload|metadata|export_input|layout_blocks)\b\s*[:：]", text):
                reasons.append(f"{task_format} answer contains internal data fields")
                score -= 0.35
            raw_field_names = {str(key) for key in output if str(key)}
            if raw_field_names:
                raw_field_pattern = r"\b(" + "|".join(re.escape(name) for name in sorted(raw_field_names)) + r")\b\s*[:：]"
                if re.search(raw_field_pattern, text, flags=re.IGNORECASE):
                    reasons.append(f"{task_format} answer contains raw output field names")
                    score -= 0.35
    if task_format == "pt":
        forbidden = {"instruction", "input", "output", "messages", "conversations", "images", "metadata"}
        text = texts[0]
        if any(re.search(rf"\b{re.escape(key)}\b\s*[:：]", text, flags=re.IGNORECASE) for key in forbidden):
            reasons.append("pt text contains training-format fields")
            score -= 0.35
    threshold = float(quality.get("min_text_format_score", 0.72))
    return _quality_result(score >= threshold, score, reasons)


def _pt_source_coverage_quality(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    quality = _quality_cfg(cfg)
    reasons: list[str] = []
    output_text = clean_text(_pt_text(sample))
    input_payload = sample.get("input_payload") if isinstance(sample.get("input_payload"), dict) else {}
    source_text = clean_text(input_payload.get("raw_text"))
    if not source_text:
        return _quality_result(bool(output_text), 1.0 if output_text else 0.0, ["pt source text is empty"] if not output_text else [])

    score = 1.0
    ratio = len(output_text) / max(1, len(source_text))
    min_ratio = float(quality.get("min_pt_output_source_ratio", 0.65))
    if ratio < min_ratio:
        reasons.append(f"pt output/source length ratio is low ({ratio:.2f})")
        score -= 0.45

    headings = re.findall(r"(?:^|\s)(\d+(?:\.\d+)*\s+[\u4e00-\u9fffA-Za-z][^\s。；:：]{1,40})", source_text)
    missing = [heading for heading in dict.fromkeys(headings) if heading not in output_text]
    max_missing = int(quality.get("max_missing_pt_section_headings", 0))
    if len(missing) > max_missing:
        reasons.append("pt output misses section headings: " + "；".join(missing[:5]))
        score -= 0.45

    threshold = float(quality.get("min_pt_source_coverage_score", 0.70))
    return _quality_result(score >= threshold, score, reasons)


def _availability_only_answer(answer: str, cfg: dict[str, Any]) -> bool:
    quality = _quality_cfg(cfg)
    unusable = any(re.search(str(pattern), answer, flags=re.IGNORECASE) for pattern in quality.get("unusable_answer_patterns", []))
    availability = any(re.search(str(pattern), answer, flags=re.IGNORECASE) for pattern in quality.get("availability_only_patterns", []))
    if not (unusable or availability):
        return False
    min_terms = int(quality.get("min_answer_specific_terms", 2))
    return len(_extract_terms(answer, limit=20)) < min_terms + 2


def _qa_relevance_quality(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    quality = _quality_cfg(cfg)
    reasons: list[str] = []
    question = _question_text(sample)
    answer = _answer_text(sample)
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    score = 1.0
    if not question or not answer:
        reasons.append("question or answer is empty")
        score -= 0.55
    if not _required_fields_present(sample, cfg):
        reasons.append("required answer fields are missing")
        score -= 0.45
    output_texts = [text for text in _flatten_text(output) if text]
    if not output_texts:
        reasons.append("answer has no substantive output text")
        score -= 0.35
    if _availability_only_answer(answer, cfg):
        reasons.append("answer only states availability/insufficient data instead of answering")
        score -= 0.45
    min_terms = int(quality.get("min_answer_specific_terms", 2))
    if len(_extract_terms(answer, limit=40)) < min_terms:
        reasons.append("answer lacks task-specific content")
        score -= 0.25
    task_type = str(sample.get("task_type") or "")
    if task_type == "section_heading_scope_alignment":
        judgement = clean_text(output.get("alignment_judgement"))
        summary = clean_text(output.get("scope_summary"))
        if judgement and summary and ("不对齐" in judgement and "对齐" in summary[:20]):
            reasons.append("alignment judgement and explanation look inconsistent")
            score -= 0.2
    if task_type == "article_metadata_extraction":
        if not any(_has_value(output.get(key)) for key in ("title", "authors", "abstract", "keywords")):
            reasons.append("metadata answer does not extract concrete metadata")
            score -= 0.3
    threshold = float(quality.get("min_qa_relevance_score", 0.70))
    return _quality_result(score >= threshold, score, reasons)


def _visual_dependency_quality(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    quality = _quality_cfg(cfg)
    reasons: list[str] = []
    score = 1.0
    question = _question_text(sample)
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    if not images:
        reasons.append("visual task has no image")
        score -= 0.55
    visual_terms = [str(term) for term in quality.get("visual_dependency_terms", []) if str(term)]
    if not any(term.lower() in question.lower() for term in visual_terms):
        reasons.append("question does not require concrete visual/page information")
        score -= 0.35
    task_type = str(sample.get("task_type") or "")
    if task_type == "figure_table_formula_to_text" and not re.search(r"(图片|图像|表格|公式|可见|显示|视觉)", question):
        reasons.append("figure/table/formula question is not grounded in the current visual object")
        score -= 0.25
    if task_type in {"page_to_journal_layout_description", "two_column_reading_order_reconstruction"}:
        if not re.search(r"(页面|版面|布局|双栏|通栏|阅读顺序|结构|区域)", question):
            reasons.append("page-layout question can be answered without the current page image")
            score -= 0.25
    if task_type == "cross_page_article_context" and not re.search(r"(连续|跨页|多页|页面|页)", question):
        reasons.append("cross-page question does not depend on the supplied page images")
        score -= 0.25
    threshold = float(quality.get("min_visual_dependency_score", 0.70))
    return _quality_result(score >= threshold, score, reasons)


def _page_no_from_image(image: str) -> int | None:
    match = re.search(r"p(\d{1,4})", image.replace("\\", "/"))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _context_support_text(sample: dict[str, Any]) -> str:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    input_payload = sample.get("input_payload") if isinstance(sample.get("input_payload"), dict) else {}
    parts: list[str] = []
    parts.extend(_flatten_text(evidence.get("evidence_text"), 60))
    parts.extend(_flatten_text(input_payload, 80))
    return clean_text(" ".join(parts))


def _image_question_correspondence_quality(sample: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    quality = _quality_cfg(cfg)
    reasons: list[str] = []
    score = 1.0
    task_type = str(sample.get("task_type") or "")
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), dict) else {}
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    if not images:
        reasons.append("no current image is attached")
        score -= 0.55
    elif not _valid_images(sample, output_dir):
        reasons.append("one or more image paths do not exist")
        score -= 0.55
    elif not _meaningful_images(sample, output_dir, cfg):
        reasons.append("one or more images are blank or low-quality")
        score -= 0.35
    visual_evidence = evidence.get("visual_evidence")
    if not isinstance(visual_evidence, list) or not visual_evidence:
        reasons.append("visual_evidence is missing")
        score -= 0.2
    source_pages = _source_pages(sample)
    image_pages = [page for image in images if (page := _page_no_from_image(str(image))) is not None]
    if image_pages and source_pages:
        source_page_set = {int(page) for page in source_pages if isinstance(page, int) or str(page).isdigit()}
        if source_page_set and not set(image_pages).issubset(source_page_set):
            reasons.append("image page numbers do not match source_pages")
            score -= 0.3
    support_text = _context_support_text(sample)
    answer = _answer_text(sample)
    min_overlap = int(quality.get("min_visual_support_overlap_terms", 1))
    overlap = _overlap_count(answer, support_text)
    support_ok = overlap >= min_overlap
    if task_type in {"page_to_journal_layout_description", "two_column_reading_order_reconstruction"}:
        payload = sample.get("input_payload") if isinstance(sample.get("input_payload"), dict) else {}
        support_ok = support_ok or bool(payload.get("layout_blocks"))
    if task_type == "figure_table_formula_to_text":
        payload = sample.get("input_payload") if isinstance(sample.get("input_payload"), dict) else {}
        support_ok = support_ok or (
            _figure_context_is_bound(sample)
            and bool(clean_text(payload.get("caption_text")) or clean_text(payload.get("related_context_text")) or payload.get("target_block"))
        )
    if task_type == "cross_page_article_context":
        support_ok = support_ok or (
            2 <= len(images) <= 3
            and (bool(output.get("page_roles")) or bool(output.get("key_points_by_page")))
        )
    if task_type == "evidence_to_claim_chain":
        support_ok = support_ok or (bool(output.get("evidence_chain")) and bool(evidence.get("evidence_text")))
    if not support_ok:
        reasons.append("answer is not supported by current image-bound OCR/layout evidence")
        score -= 0.35
    threshold = float(quality.get("min_image_question_correspondence_score", 0.70))
    return _quality_result(score >= threshold, score, reasons)


def _run_quality_checks(sample: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    quality = _quality_cfg(cfg)
    if quality and not bool(quality.get("enabled", True)):
        return {"passed": True, "dimensions": {}}
    task_type = str(sample.get("task_type") or "")
    task_format = _task_format(task_type, cfg)
    dimensions: dict[str, dict[str, Any]] = {}
    dimensions["text_format_quality"] = _text_format_quality(sample, task_format, cfg)
    if task_format == "pt":
        dimensions["pt_source_coverage"] = _pt_source_coverage_quality(sample, cfg)
    if task_format in {"alpaca", "sharegpt"}:
        dimensions["qa_relevance"] = _qa_relevance_quality(sample, cfg)
    if task_format == "sharegpt":
        dimensions["visual_dependency"] = _visual_dependency_quality(sample, cfg)
        dimensions["image_question_correspondence"] = _image_question_correspondence_quality(sample, output_dir, cfg)
    passed = all(item.get("passed") for item in dimensions.values())
    scores = [float(item.get("score", 0.0)) for item in dimensions.values()]
    return {
        "passed": passed,
        "score": round(sum(scores) / len(scores), 3) if scores else 1.0,
        "format": task_format,
        "dimensions": dimensions,
    }


def validate_sample(sample: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    task_type = str(sample.get("task_type") or "")
    if task_type not in set(cfg.get("task_types", [])):
        return None
    is_pt_task = task_type in _pt_task_types(cfg)
    if is_pt_task:
        min_chars = int(cfg.get("validation", {}).get("min_pt_text_chars", 80))
        if not _has_value(sample.get("id")) or len(clean_text(_pt_text(sample))) < min_chars:
            return None
        if _low_value_page(sample, cfg) or not _pt_payload_is_pure(sample):
            return None
    else:
        if not _has_value(sample.get("id")) or not _has_value(sample.get("instruction")) or not _has_value(sample.get("output_payload")):
            return None
        if _low_value_page(sample, cfg) and task_type != "page_to_journal_layout_description":
            return None
    if not _required_fields_present(sample, cfg):
        return None
    pages = _source_pages(sample)
    if not pages:
        return None
    multi_page_tasks = set(cfg.get("validation", {}).get("multi_page_tasks", []))
    if task_type in multi_page_tasks and len(set(pages)) < 2:
        return None
    image_required_tasks = set(cfg.get("validation", {}).get("image_required_tasks", []))
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    if task_type in image_required_tasks and not images:
        return None
    if images and not _valid_images(sample, output_dir):
        return None
    if not _figure_context_is_bound(sample):
        return None
    if not _figure_instruction_is_generic(sample, cfg):
        return None
    if not _cross_page_images_ok(sample, cfg):
        return None
    quality_checks = _run_quality_checks(sample, output_dir, cfg)
    if not quality_checks.get("passed"):
        return None
    # Ensure output can always be serialized before export.
    try:
        json.dumps(sample.get("output_payload", {}), ensure_ascii=False)
    except TypeError:
        return None
    metadata = sample.setdefault("metadata", {})
    metadata["validator"] = "rule"
    metadata["quality_checks"] = quality_checks
    metadata["quality_score"] = float(quality_checks.get("score") or cfg["validation"]["default_quality_score"])
    return sample
