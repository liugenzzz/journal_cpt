from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..core.io_utils import write_jsonl

_INTERNAL_FIELD_KEYS = {
    "block_id",
    "block_ids",
    "controlled_block_ids",
    "evidence_block_ids",
    "filtered_noise_blocks",
    "filtered_content",
    "source_pages",
    "visual_evidence",
    "visual_evidence_by_page",
    "bbox",
    "confidence",
    "score",
    "quality_score",
    "quality_checks",
    "metadata",
    "generator",
    "generator_model",
    "generator_provider",
    "image_path",
    "images",
    "page_image",
    "target_image",
    "normalized_page_path",
    "mineru_parse_path",
    "task_type",
    "object_label",
}

_FIELD_LABELS = {
    "page_type": "页面类型",
    "article_role": "页面作用",
    "column_mode": "栏式结构",
    "main_topic": "主要主题",
    "page_visual_description": "页面视觉描述",
    "page_visual_descriptions": "页面视觉描述",
    "layout_regions": "版面区域",
    "content_blocks": "主要内容块",
    "figures_tables_formulas": "图表公式",
    "trainable_content": "可训练内容",
    "reading_order": "阅读顺序",
    "reconstructed_text": "恢复正文",
    "uncertainty_notes": "不确定处",
    "object_type": "对象类型",
    "visual_description": "视觉描述",
    "visible_structure": "可见结构",
    "caption_information": "上下文说明",
    "text_explanation": "正文解释",
    "evidence_to_claim": "证据结论",
    "claim": "论点",
    "evidence_chain": "关键证据",
    "chain_steps": "推理链条",
    "reasoning_scope": "适用范围",
    "context_topic": "跨页主题",
    "page_roles": "页面作用",
    "cross_page_summary": "跨页概述",
    "continuity_relations": "衔接关系",
    "key_points_by_page": "各页要点",
    "text_excerpt": "内容摘录",
    "description": "说明",
    "evidence": "依据",
    "key_point": "要点",
    "role": "作用",
    "summary": "摘要",
    "title": "题名",
    "authors": "作者",
    "affiliations": "作者单位",
    "journal_name": "期刊",
    "year": "年份",
    "volume": "卷",
    "issue": "期",
    "pages": "页码",
    "doi": "DOI",
    "abstract": "摘要",
    "keywords": "关键词",
    "classification_no": "分类号",
    "document_code": "文献标识码",
    "funding": "基金信息",
    "heading": "小节标题",
    "heading_level": "标题层级",
    "scope_summary": "正文范围",
    "alignment_judgement": "匹配判断",
    "mismatch_risk": "不确定性",
    "section_title": "小节标题",
    "key_terms": "关键术语",
    "method_or_condition_points": "方法与条件要点",
    "result_or_claim_points": "结果与结论要点",
    "research_object": "研究对象",
    "method_steps": "方法流程",
    "experimental_or_simulation_conditions": "实验或仿真条件",
    "variables_and_parameters": "变量和参数",
    "evaluation_metrics": "评价指标",
    "assumptions_or_constraints": "假设与约束",
    "research_problem": "研究问题",
    "contributions": "主要贡献",
    "key_findings": "关键发现",
    "conclusions": "结论",
    "conditions_or_limitations": "适用条件与局限",
    "supporting_evidence": "支撑证据",
}

_BLOCK_TYPE_LABELS = {
    "title": "标题",
    "article_title": "论文题名",
    "section": "小节标题",
    "section_title": "小节标题",
    "text": "正文",
    "paragraph": "正文",
    "figure": "图示",
    "image": "图示",
    "table": "表格",
    "formula": "公式",
    "equation": "公式",
    "figure_caption": "图注",
    "table_caption": "表注",
    "caption": "图表说明",
    "journal_header": "期刊栏眉",
    "header": "页眉",
    "footer": "页脚",
    "page_number": "页码",
    "reference": "参考文献",
    "unknown": "其他",
}

_COLUMN_LABELS = {
    "left": "左栏",
    "right": "右栏",
    "center": "中部",
    "full": "通栏",
    "full_width": "通栏",
    "single": "单栏",
    "single_column": "单栏",
    "two_column": "双栏",
    "multi_column": "多栏",
    "mixed": "混排",
    "unknown": "",
}

_SCOPE_LABELS = {
    "page_fragment": "当前页片段",
    "article_fragment": "论文片段",
    "article": "整篇论文",
}

_RAW_FIELD_NAMES = {
    "title",
    "authors",
    "affiliations",
    "journal_name",
    "year",
    "volume",
    "issue",
    "pages",
    "doi",
    "abstract",
    "keywords",
    "classification_no",
    "document_code",
    "funding",
    "heading",
    "heading_level",
    "controlled_block_ids",
    "scope_summary",
    "alignment_judgement",
    "mismatch_risk",
    "section_title",
    "summary",
    "key_terms",
    "method_or_condition_points",
    "result_or_claim_points",
    "research_object",
    "method_steps",
    "experimental_or_simulation_conditions",
    "variables_and_parameters",
    "evaluation_metrics",
    "assumptions_or_constraints",
    "research_problem",
    "contributions",
    "key_findings",
    "conclusions",
    "conditions_or_limitations",
    "supporting_evidence",
}


def _answer(sample: dict[str, Any]) -> str:
    generated = _generated_answer(sample)
    if generated:
        return generated
    payload = sample.get("output_payload")
    if isinstance(payload, dict):
        text = _format_payload_summary(payload)
        if text:
            return text
    return json.dumps(sample.get("output_payload", {}), ensure_ascii=False)


def _input(sample: dict[str, Any]) -> str:
    value = sample.get("export_input")
    if isinstance(value, str) and value.strip():
        return value
    return json.dumps(sample.get("input_payload", {}), ensure_ascii=False)


def _metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _clean_scalar(value: Any, limit: int = 900) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        text = "是" if value else "否"
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bbbox\s*=\s*\[[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bblock_id\s*[:=]\s*[\w.-]+", "", text, flags=re.IGNORECASE)
    text = text.strip(" ,;，；")
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _looks_like_json_text(value: Any) -> bool:
    text = _clean_scalar(value, 0)
    if not text or text[:1] not in ("{", "["):
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return bool(re.match(r"^[{\[]\s*['\"]?(instruction|answer|output_payload|input_payload|metadata)", text))


def _generated_answer(sample: dict[str, Any]) -> str:
    answer = _clean_scalar(sample.get("answer"), 5000)
    if not answer or _looks_like_json_text(answer):
        return ""
    if re.search(r"\b(input_payload|output_payload|metadata|export_input|layout_blocks)\b\s*[:：]", answer):
        return ""
    raw_field_pattern = r"\b(" + "|".join(re.escape(name) for name in sorted(_RAW_FIELD_NAMES)) + r")\b\s*[:：]"
    if re.search(raw_field_pattern, answer, flags=re.IGNORECASE):
        return ""
    return answer


def _label_block_type(value: Any) -> str:
    text = _clean_scalar(value, 80)
    return _BLOCK_TYPE_LABELS.get(text, text)


def _label_column(value: Any) -> str:
    text = _clean_scalar(value, 80)
    return _COLUMN_LABELS.get(text, text)


def _page_prefix(value: Any) -> str:
    if not _has_value(value):
        return ""
    return f"第 {_clean_scalar(value, 30)} 页"


def _count_items(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if _has_value(item)])
    return 1 if _has_value(value) else 0


def _format_block(value: dict[str, Any], limit: int = 260) -> str:
    excerpt = _clean_scalar(value.get("text_excerpt") or value.get("text") or value.get("content"), limit)
    role = _label_block_type(value.get("semantic_role") or value.get("block_type"))
    column = _label_column(value.get("column"))
    prefix = " / ".join(item for item in (column, role) if item)
    if excerpt:
        return f"{prefix}：{excerpt}" if prefix else excerpt
    return prefix


def _format_step(value: dict[str, Any], limit: int = 520) -> str:
    if not _has_value(value.get("step")) and not _has_value(value.get("description")):
        return ""
    step = _clean_scalar(value.get("step"), 30)
    description = _format_value(value.get("description"), limit=limit, item_limit=3)
    evidence = _format_value(value.get("evidence"), limit=limit, item_limit=3)
    prefix = f"第 {step} 步" if step else "步骤"
    if description and evidence:
        return f"{prefix}：{description}；依据：{evidence}"
    if description:
        return f"{prefix}：{description}"
    if evidence:
        return f"{prefix}依据：{evidence}"
    return ""


def _format_page_item(value: dict[str, Any], limit: int = 520) -> str:
    page = _page_prefix(value.get("page_index") or value.get("page") or value.get("page_no"))
    content = ""
    for key in ("key_point", "summary", "role", "description", "text_excerpt", "text", "content"):
        if _has_value(value.get(key)):
            content = _format_value(value.get(key), limit=limit, item_limit=3)
            break
    if not content:
        return ""
    return f"{page}：{content}" if page else content


def _format_dict(value: dict[str, Any], limit: int = 900, item_limit: int = 6) -> str:
    if "step" in value or ("description" in value and "evidence" in value):
        step_text = _format_step(value, limit)
        if step_text:
            return step_text
    if any(key in value for key in ("text_excerpt", "block_type", "semantic_role")):
        block_text = _format_block(value, min(limit, 320))
        if block_text:
            return block_text
    if any(key in value for key in ("page_index", "page", "page_no")):
        page_text = _format_page_item(value, min(limit, 520))
        if page_text:
            return page_text

    parts: list[str] = []
    preferred_keys = (
        "summary",
        "description",
        "role",
        "key_point",
        "text",
        "content",
        "evidence",
        "claim",
        "evidence_to_claim",
    )
    for key in preferred_keys:
        if key in value and key not in _INTERNAL_FIELD_KEYS:
            formatted = _format_value(value.get(key), limit=min(limit, 360), item_limit=3)
            if formatted:
                label = _FIELD_LABELS.get(key)
                parts.append(f"{label}：{formatted}" if label else formatted)
        if len(parts) >= item_limit:
            break
    if not parts:
        for key, child in value.items():
            if key in _INTERNAL_FIELD_KEYS:
                continue
            label = _FIELD_LABELS.get(key)
            if not label:
                continue
            formatted = _format_value(child, limit=min(limit, 360), item_limit=3)
            if formatted:
                parts.append(f"{label}：{formatted}")
            if len(parts) >= item_limit:
                break
    return _clean_scalar("；".join(parts), limit)


def _format_list(value: Any, limit: int = 900, item_limit: int = 6) -> str:
    if not isinstance(value, (list, tuple, set)):
        return ""
    parts: list[str] = []
    for item in list(value)[:item_limit]:
        formatted = _format_value(item, limit=min(limit, 420), item_limit=3)
        if formatted:
            parts.append(formatted)
    return _clean_scalar("；".join(parts), limit)


def _format_value(value: Any, limit: int = 900, item_limit: int = 6) -> str:
    if not _has_value(value):
        return ""
    if isinstance(value, dict):
        return _format_dict(value, limit=limit, item_limit=item_limit)
    if isinstance(value, (list, tuple, set)):
        return _format_list(value, limit=limit, item_limit=item_limit)
    return _clean_scalar(value, limit)


def _numbered_lines(value: Any, limit: int = 8) -> str:
    if not isinstance(value, (list, tuple, set)):
        return _format_value(value, limit=1200)
    lines: list[str] = []
    for index, item in enumerate(list(value)[:limit], start=1):
        formatted = _format_value(item, limit=520, item_limit=4)
        if formatted:
            lines.append(f"{index}. {formatted}")
    return "\n".join(lines)


def _format_regions(value: Any) -> str:
    if not isinstance(value, list):
        return _format_value(value, limit=700)
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            formatted = _format_value(item, limit=120)
            if formatted:
                parts.append(formatted)
            continue
        region = _label_column(item.get("region") or item.get("region_type") or item.get("type"))
        count = item.get("block_count")
        description = _format_value(item.get("description") or item.get("content"), limit=180, item_limit=2)
        if region and _has_value(count):
            suffix = f"，{description}" if description else ""
            parts.append(f"{region}约 {count} 个内容块{suffix}")
        elif region:
            parts.append(f"{region}：{description}" if description else region)
        elif description:
            parts.append(description)
    return "；".join(parts)


def _append_line(lines: list[str], label: str, value: Any, *, limit: int = 900, item_limit: int = 6) -> None:
    formatted = _format_value(value, limit=limit, item_limit=item_limit)
    if formatted:
        lines.append(f"{label}：{formatted}")


def _append_numbered(lines: list[str], label: str, value: Any, *, limit: int = 8) -> None:
    formatted = _numbered_lines(value, limit=limit)
    if formatted:
        lines.append(f"{label}：\n{formatted}")


def _format_two_column_answer(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _append_line(lines, "页面视觉描述", payload.get("page_visual_description"), limit=1000)
    _append_line(lines, "栏式结构", _label_column(payload.get("column_mode")) or payload.get("column_mode"), limit=120)
    noise_count = _count_items(payload.get("filtered_noise_blocks"))
    if noise_count:
        lines.append(f"版面噪声：已排除 {noise_count} 个页眉、页脚、页码或其他非正文内容块。")
    _append_numbered(lines, "阅读顺序", payload.get("reading_order"), limit=12)
    _append_line(lines, "恢复正文", payload.get("reconstructed_text"), limit=1800)
    _append_line(lines, "图表公式", payload.get("figures_tables_formulas"), limit=900, item_limit=5)
    _append_line(lines, "不确定处", payload.get("uncertainty_notes"), limit=500, item_limit=4)
    return "\n".join(lines)


def _format_layout_answer(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _append_line(lines, "页面视觉描述", payload.get("page_visual_description"), limit=1000)
    _append_line(lines, "页面类型", payload.get("page_type"), limit=120)
    _append_line(lines, "页面作用", payload.get("article_role"), limit=180)
    _append_line(lines, "栏式结构", _label_column(payload.get("column_mode")) or payload.get("column_mode"), limit=120)
    _append_line(lines, "主要主题", payload.get("main_topic"), limit=260)
    regions = _format_regions(payload.get("layout_regions"))
    if regions:
        lines.append(f"版面区域：{regions}")
    _append_numbered(lines, "主要内容块", payload.get("content_blocks"), limit=8)
    _append_line(lines, "图表公式", payload.get("figures_tables_formulas"), limit=900, item_limit=5)
    _append_line(lines, "可训练内容", payload.get("trainable_content"), limit=1200)
    filtered_count = _count_items(payload.get("filtered_content"))
    if filtered_count:
        lines.append(f"过滤建议：页眉、页脚、页码、参考文献编号等 {filtered_count} 个版面噪声不作为正文训练内容。")
    return "\n".join(lines)


def _format_visual_answer(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _append_line(lines, "对象类型", _label_block_type(payload.get("object_type")) or payload.get("object_type"), limit=120)
    _append_line(lines, "视觉描述", payload.get("visual_description"), limit=900)
    _append_line(lines, "可见结构", payload.get("visible_structure"), limit=700)
    _append_line(lines, "上下文说明", payload.get("caption_information"), limit=700)
    _append_line(lines, "正文解释", payload.get("text_explanation"), limit=900)
    _append_line(lines, "证据结论", payload.get("evidence_to_claim"), limit=700)
    return "\n".join(lines)


def _format_evidence_answer(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _append_line(lines, "页面视觉描述", payload.get("page_visual_description"), limit=1000)
    _append_line(lines, "论点", payload.get("claim"), limit=500)
    _append_line(lines, "关键证据", payload.get("evidence_chain"), limit=900, item_limit=5)
    _append_numbered(lines, "推理链条", payload.get("chain_steps"), limit=5)
    if _has_value(payload.get("supporting_figures_tables_formulas")):
        lines.append("视觉证据：页面中相关图表、表格或公式用于支撑上述论点。")
    _append_line(lines, "图表公式", payload.get("figures_tables_formulas"), limit=900, item_limit=5)
    scope = _SCOPE_LABELS.get(str(payload.get("reasoning_scope") or ""), payload.get("reasoning_scope"))
    _append_line(lines, "适用范围", scope, limit=180)
    return "\n".join(lines)


def _format_cross_page_answer(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    _append_numbered(lines, "页面视觉描述", payload.get("page_visual_descriptions"), limit=4)
    _append_line(lines, "跨页主题", payload.get("context_topic"), limit=420)
    _append_numbered(lines, "页面作用", payload.get("page_roles"), limit=4)
    _append_line(lines, "跨页概述", payload.get("cross_page_summary"), limit=1500)
    _append_line(lines, "衔接关系", payload.get("continuity_relations"), limit=900, item_limit=5)
    _append_line(lines, "图表公式", payload.get("figures_tables_formulas"), limit=900, item_limit=5)
    _append_numbered(lines, "各页要点", payload.get("key_points_by_page"), limit=4)
    return "\n".join(lines)


def _format_payload_summary(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if key in _INTERNAL_FIELD_KEYS:
            continue
        label = _FIELD_LABELS.get(key)
        if not label:
            continue
        _append_line(lines, label, value, limit=1000)
    if lines:
        return "\n".join(lines)
    values = [_format_value(value, limit=700) for key, value in payload.items() if key not in _INTERNAL_FIELD_KEYS]
    return "\n".join(value for value in values if value)


_SHAREGPT_ANSWER_FORMATTERS = {
    "two_column_reading_order_reconstruction": _format_two_column_answer,
    "page_to_journal_layout_description": _format_layout_answer,
    "figure_table_formula_to_text": _format_visual_answer,
    "evidence_to_claim_chain": _format_evidence_answer,
    "cross_page_article_context": _format_cross_page_answer,
}


def _sharegpt_answer(sample: dict[str, Any]) -> str:
    generated = _generated_answer(sample)
    if generated:
        return generated
    payload = sample.get("output_payload")
    if not isinstance(payload, dict):
        return _clean_scalar(sample.get("answer"), 1800) or "当前样本未提供可用的结构化回答。"
    task_type = str(sample.get("task_type") or "")
    formatter = _SHAREGPT_ANSWER_FORMATTERS.get(task_type)
    text = formatter(payload) if formatter else _format_payload_summary(payload)
    return text.strip() or "当前样本未提供可用的结构化回答。"


def to_sharegpt(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    image_tokens = "".join(cfg["sharegpt_image_token"] for _ in images)
    instruction = str(sample.get("instruction") or "").replace(cfg["sharegpt_image_token"], "").strip()
    record: dict[str, Any] = {
        "id": sample["id"],
        "messages": [
            {"role": "user", "content": f"{image_tokens}{instruction}".strip()},
            {"role": "assistant", "content": _sharegpt_answer(sample)},
        ],
        "metadata": _metadata(sample),
    }
    if images:
        record["images"] = images
    return record


def to_alpaca(sample: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "instruction": sample.get("instruction", ""),
        "input": _input(sample),
        "output": _answer(sample),
        "metadata": _metadata(sample),
    }
    images = sample.get("images") if isinstance(sample.get("images"), list) else []
    if images:
        record["images"] = images
    return record


def to_pt(sample: dict[str, Any]) -> dict[str, Any]:
    output = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    text = str(output.get("text") or sample.get("text") or "").strip()
    return {"text": text}


def _visual_sample_to_pt(sample: dict[str, Any]) -> dict[str, str]:
    if str(sample.get("task_type") or "") != "figure_table_formula_to_text":
        return {"text": ""}
    payload = sample.get("output_payload") if isinstance(sample.get("output_payload"), dict) else {}
    if not payload:
        return {"text": ""}
    metadata = _metadata(sample)
    lines = [
        f"文章：{metadata.get('article_title', '')}".strip(),
        f"页码：{metadata.get('page_label') or metadata.get('page_index', '')}".strip(),
        "图表公式知识：",
    ]
    object_type = _label_block_type(payload.get("object_type"))
    if object_type:
        lines.append(f"对象类型：{object_type}")
    _append_line(lines, "视觉内容", payload.get("visual_description"), limit=1400)
    _append_line(lines, "可见结构", payload.get("visible_structure"), limit=1200)
    _append_line(lines, "图注表注或公式说明", payload.get("caption_information"), limit=1000)
    _append_line(lines, "正文解释", payload.get("text_explanation"), limit=1400)
    _append_line(lines, "证据结论", payload.get("evidence_to_claim"), limit=1000)
    text = "\n".join(line for line in lines if line and not line.endswith("："))
    return {"text": text.strip()}


def _pt_records_for_task(task_type: str, task_samples: list[dict[str, Any]], all_samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = [record for sample in task_samples if (record := to_pt(sample))["text"]]
    if task_type == "domain_knowledge_corpus":
        visual_records = [
            record
            for sample in all_samples
            if (record := _visual_sample_to_pt(sample))["text"]
        ]
        records.extend(visual_records)
    return records


def export_task_files(output_dir: Path, samples: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    sharegpt_task_types = set(cfg["export_formats"]["sharegpt"])
    alpaca_task_types = set(cfg["export_formats"]["alpaca"])
    pt_task_types = set(cfg["export_formats"].get("pt", []))
    for task_type in cfg["task_types"]:
        task_samples = [sample for sample in samples if sample.get("task_type") == task_type]
        if task_type in sharegpt_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_sharegpt"].format(task_type=task_type),
                (to_sharegpt(sample, cfg) for sample in task_samples),
            )
        if task_type in alpaca_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_alpaca"].format(task_type=task_type),
                (to_alpaca(sample) for sample in task_samples),
            )
        if task_type in pt_task_types:
            write_jsonl(
                output_dir / cfg["paths"]["export_pt"].format(task_type=task_type),
                _pt_records_for_task(task_type, task_samples, samples),
            )
