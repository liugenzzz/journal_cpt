from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.io_utils import clean_text, safe_name, truncate_text, utc_now
from ..core.models import JournalSampleJob
from ..services.clients import VlmPool, parse_json_array, parse_jsonl_objects


JSON_OUTPUT_INSTRUCTION = (
    "For valid JSON, every literal backslash in string values must be escaped as two backslash characters, especially LaTeX examples like \\\\alpha, \\\\frac, \\\\%, or \\\\_. "
    "只返回合法 JSON 数组，不要使用 markdown 代码块。"
    "每条样本必须包含 instruction、answer 字段和该任务要求的全部输出字段。"
    "instruction 必须由你根据当前输入证据重新生成，保持表达多样，不得复制固定模板句。"
    "answer 必须是忠实于当前期刊上下文、图片、OCR、版面块和绑定上下文的自然语言回答，"
    "可以用中文小标题分段组织，但不要输出 JSON、字段清单、input_payload、output_payload、metadata 或内部 block_id/bbox。"
    "证据不足、不适合该任务或上下文与目标图表不匹配时返回 []。"
)

ALPACA_QA_TASK_TYPES = {
    "article_metadata_extraction",
    "section_heading_scope_alignment",
    "section_keypoint_summary",
    "method_experiment_condition_extraction",
    "article_contribution_conclusion",
}

SHAREGPT_TASK_TYPES = {
    "two_column_reading_order_reconstruction",
    "page_to_journal_layout_description",
    "figure_table_formula_to_text",
    "evidence_to_claim_chain",
    "cross_page_article_context",
}

PT_JSONL_OUTPUT_INSTRUCTION = (
    "只返回 JSONL，不要使用 markdown 代码块。"
    "每一行必须是一个完整 JSON 对象，并且只包含 text 字段。"
    "不要生成 instruction、input、output、messages、conversations 或问答格式。"
    "如果输入没有可用于继续预训练的论文正文知识，请返回空内容。"
)

REFERENCE_MARK_RE = re.compile(
    r"\s*(?:\$?\s*\^\s*\{\s*)?[\[［]\s*\d+(?:\s*[-–—,，、]\s*\d+)*\s*[\]］](?:\s*\}\s*\$?)?"
)

TASK_SPECS: dict[str, dict[str, Any]] = {
    "article_metadata_extraction": {
        "format": "alpaca",
        "input_fields": ["source_text", "layout_blocks", "page_image"],
        "output_fields": [
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
        ],
        "quality_rules": ["只能抽取首页可见字段；目录、封面、编委页返回 []；answer 要用自然语言完整说明可见元数据，不要输出字段变量名。"],
    },
    "two_column_reading_order_reconstruction": {
        "format": "sharegpt",
        "input_fields": ["page_image", "layout_blocks", "ocr_text", "cross_column_barriers"],
        "output_fields": [
            "page_visual_description",
            "column_mode",
            "filtered_noise_blocks",
            "reading_order",
            "cross_column_barriers",
            "reconstructed_text",
            "figures_tables_formulas",
            "uncertainty_notes",
        ],
        "quality_rules": [
            "必须区分通栏、左栏、右栏和跨栏图表；跨栏图表是双栏阅读的纵向分隔点，读取左栏时一旦到达跨栏图表，应先转读同一纵向带内的右栏内容，不得继续读跨栏图表下方的左栏内容；读完该带右栏后再读跨栏图表，再进入图表下方的新双栏区域。",
            "回答要覆盖页面主要正文、图表/表格/公式和应过滤内容，不得把页眉页脚混入正文读序。",
        ],
    },
    "page_to_journal_layout_description": {
        "format": "sharegpt",
        "input_fields": ["page_image", "page_type", "layout_blocks", "ocr_text"],
        "output_fields": [
            "page_visual_description",
            "page_type",
            "article_role",
            "column_mode",
            "main_topic",
            "layout_regions",
            "content_blocks",
            "figures_tables_formulas",
            "trainable_content",
            "filtered_content",
        ],
        "quality_rules": ["页面类型必须来自期刊页面类型集合；回答要完整覆盖页面文字、版面结构、图表/表格/公式、可训练内容和应过滤内容。"],
    },
    "section_heading_scope_alignment": {
        "format": "alpaca",
        "input_fields": ["heading", "controlled_text", "controlled_blocks"],
        "output_fields": ["heading", "heading_level", "controlled_block_ids", "scope_summary", "alignment_judgement", "mismatch_risk"],
        "quality_rules": ["不要把论文题名当作小节标题；判断必须依赖标题和受控正文；answer 要说明标题、控制范围、匹配判断和风险，不要输出字段变量名。"],
    },
    "section_keypoint_summary": {
        "format": "alpaca",
        "input_fields": ["section_title", "section_text"],
        "output_fields": ["section_title", "summary", "key_terms", "method_or_condition_points", "result_or_claim_points"],
        "quality_rules": ["摘要必须来自输入正文；answer 要尽量覆盖小节中的研究目的、方法、参数条件、结果、结论和关键术语，不要输出字段变量名。"],
    },
    "figure_table_formula_to_text": {
        "format": "sharegpt",
        "input_fields": ["target_image", "target_block", "caption_text", "related_context_text"],
        "output_fields": ["object_type", "object_label", "visual_description", "visible_structure", "caption_information", "text_explanation", "evidence_to_claim"],
        "quality_rules": [
            "必须先对目标图片/表格/公式的可见内容做详细视觉描述，再结合图注/表注、绑定正文和领域上下文生成问题与答案。",
            "只能使用 target_block、caption_text 和 related_context_text 中与目标对象绑定的证据。",
            "不得引用页面上其他图表、其他公式或无标签邻近正文。",
            "instruction 只能使用“这张图片/该表格/该公式”等泛称，不得出现图号、表号、公式号、图表题名、论文题名或小节标题。",
        ],
    },
    "method_experiment_condition_extraction": {
        "format": "alpaca",
        "input_fields": ["article_title", "source_text"],
        "output_fields": ["research_object", "method_steps", "experimental_or_simulation_conditions", "variables_and_parameters", "evaluation_metrics", "assumptions_or_constraints"],
        "quality_rules": ["只抽取明确出现的条件、参数、工况、设备、变量和指标；answer 要完整说明研究对象、方法流程、试验/仿真条件、参数和评价指标，不要输出字段变量名。"],
    },
    "evidence_to_claim_chain": {
        "format": "sharegpt",
        "input_fields": ["page_image", "source_text"],
        "output_fields": ["page_visual_description", "claim", "evidence_chain", "chain_steps", "supporting_figures_tables_formulas", "reasoning_scope"],
        "quality_rules": ["问题和答案必须忠于当前页面内容；证据链每一步必须能回溯到正文、图表、公式、摘要或结论，并描述页面中可见的图表/表格/公式证据。"],
    },
    "article_contribution_conclusion": {
        "format": "alpaca",
        "input_fields": ["article_title", "article_text"],
        "output_fields": ["research_problem", "contributions", "key_findings", "conclusions", "conditions_or_limitations", "supporting_evidence"],
        "quality_rules": ["说明范围是整篇还是片段；贡献和结论必须由输入证据支持；answer 要覆盖研究问题、贡献、发现、结论、适用条件和证据范围，不要输出字段变量名。"],
    },
    "cross_page_article_context": {
        "format": "sharegpt",
        "input_fields": ["page_images", "page_texts"],
        "output_fields": ["page_visual_descriptions", "context_topic", "page_roles", "cross_page_summary", "continuity_relations", "figures_tables_formulas", "key_points_by_page"],
        "quality_rules": ["必须依赖至少两页证据；输入图片应为 2-3 张连续页面图；回答要覆盖跨页正文、图表/表格/公式和页面间承接关系，无清晰关联时返回 []。"],
    },
    "domain_knowledge_corpus": {
        "format": "pt",
        "input_fields": ["journal_name", "article_title", "page_range", "raw_text"],
        "output_fields": ["text"],
        "response_format": "jsonl",
        "requires_instruction": False,
        "quality_rules": ["PT 每行只包含 text 字段；过滤封面、目录、编委、参考文献列表和广告页。"],
    },
}

EVIDENCE_KEYS = {
    "source_pages",
    "evidence_block_ids",
    "evidence_text",
    "visual_evidence",
    "evidence_text_by_page",
    "visual_evidence_by_page",
    "supporting_evidence",
}


def _metadata(job: JournalSampleJob, cfg: dict[str, Any], sample_no: int) -> dict[str, Any]:
    block = job.block
    article = job.article
    return {
        "source_pdf": job.journal.source_pdf,
        "journal_name": job.journal.journal_name,
        "journal_id": job.journal.journal_id,
        "article_id": article.article_id if article else job.page.article_id,
        "article_title": article.article_title if article else job.page.article_title,
        "year": article.year if article else job.journal.year,
        "volume": article.volume if article else job.journal.volume,
        "issue": article.issue if article else job.journal.issue,
        "page_index": job.page.page_index,
        "page_label": job.page.page_label,
        "page_type": job.page.page_type,
        "column_mode": job.page.column_mode,
        "block_id": block.block_id if block else cfg["default_block_id"],
        "block_type": block.block_type if block else cfg["default_block_type"],
        "semantic_role": block.semantic_role if block else cfg["default_semantic_role"],
        "bbox": block.bbox if block else [],
        "image_path": job.images[0] if job.images else "",
        "mineru_parse_path": job.page.mineru_parse_path,
        "normalized_page_path": job.page.normalized_page_path,
        "task_type": job.task_type,
        "created_at": utc_now(cfg["timestamp_format"]),
        "pipeline_version": cfg["pipeline_version"],
        "sample_no": sample_no,
        "generator": "llm",
    }


def _sample_id(metadata: dict[str, Any]) -> str:
    journal_id = safe_name(str(metadata.get("journal_id") or "journal"), "journal")
    article_id = safe_name(str(metadata.get("article_id") or "article_unknown"), "article_unknown")
    page_no = int(metadata.get("page_index") or 0)
    block_id = safe_name(str(metadata.get("block_id") or "page"), "page")
    task_type = str(metadata.get("task_type") or "task")
    sample_no = int(metadata.get("sample_no") or 0)
    return f"{journal_id}_{article_id}_p{page_no:03d}_{block_id}_{task_type}_{sample_no:06d}"


def _source_pages(job: JournalSampleJob) -> list[int]:
    pages = job.page_window or [job.page]
    return [page.page_index for page in pages]


def _evidence_block_ids(job: JournalSampleJob) -> list[str]:
    if job.blocks:
        return [block.block_id for block in job.blocks]
    if job.block:
        return [job.block.block_id]
    ids: list[str] = []
    for page in job.page_window or [job.page]:
        ids.extend(block.block_id for block in page.blocks if clean_text(block.text))
    return ids[:32]


def _evidence_text(job: JournalSampleJob) -> list[str]:
    if job.blocks:
        return [block.text for block in job.blocks if clean_text(block.text)]
    if job.block and clean_text(job.block.text):
        return [job.block.text]
    texts: list[str] = []
    for page in job.page_window or [job.page]:
        for block in page.blocks:
            text = clean_text(block.text)
            if text:
                texts.append(text)
    return texts[:16]


def _window_text(job: JournalSampleJob, limit: int) -> str:
    pages = job.page_window or [job.page]
    text = "\n\n".join(page.full_text for page in pages if clean_text(page.full_text))
    return truncate_text(text, limit)


def _page_visible_text(page: Any) -> str:
    return clean_text(getattr(page, "visible_text", "")) or clean_text(getattr(page, "full_text", ""))


def _block_text(blocks: list[Any], limit: int) -> str:
    texts: list[str] = []
    for block in blocks:
        if isinstance(block, dict):
            texts.append(clean_text(block.get("markdown") or block.get("text")))
        else:
            texts.append(clean_text(getattr(block, "markdown", "") or getattr(block, "text", "")))
    return truncate_text("\n".join(text for text in texts if text), limit)


def _page_texts(job: JournalSampleJob, limit: int) -> list[dict[str, Any]]:
    pages = job.page_window or [job.page]
    return [
        {
            "page_index": page.page_index,
            "page_type": page.page_type,
            "column_mode": page.column_mode,
            "text": truncate_text(page.full_text, limit),
            "visible_text": truncate_text(_page_visible_text(page), limit),
            "paragraphs": list(getattr(page, "paragraphs", [])),
        }
        for page in pages
    ]


def _visual_context_text(job: JournalSampleJob, limit: int) -> tuple[str, str]:
    caption_texts: list[str] = []
    related_texts: list[str] = []
    target_id = job.block.block_id if job.block else ""
    for block in job.blocks:
        text = clean_text(block.text)
        if not text or block.block_id == target_id:
            continue
        if block.semantic_role in {"figure_caption", "table_caption"}:
            caption_texts.append(text)
        else:
            related_texts.append(text)
    return truncate_text("\n".join(caption_texts), limit), truncate_text("\n".join(related_texts), limit)


def _page_range(job: JournalSampleJob) -> str:
    pages = job.page_window or [job.page]
    start = clean_text(pages[0].page_label) or str(pages[0].page_index)
    end = clean_text(pages[-1].page_label) or str(pages[-1].page_index)
    return start if start == end else f"{start}-{end}"


def _truncate_raw_text(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    breakpoints = [cut.rfind("\n\n"), cut.rfind("\n"), cut.rfind("。")]
    breakpoint = max(breakpoints)
    if breakpoint > limit // 2:
        return cut[: breakpoint + 1].strip()
    return cut


def _clean_pt_corpus_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        line = REFERENCE_MARK_RE.sub("", line)
        line = re.sub(r"\s*(LI\s+Y\s+D\.\s+Application of safety analysis.*?Civil Aircraft Design & Research,\s*\d{4})", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line.upper() == "OSID:":
            continue
        line = re.sub(r"\s+([，。；：！？、,.!?;:])", r"\1", line)
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _corpus_raw_text(job: JournalSampleJob, limit: int) -> str:
    raw = job.source.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return _truncate_raw_text(raw, limit)
    return _truncate_raw_text(_window_text(job, limit), limit)


def _template_input(job: JournalSampleJob, cfg: dict[str, Any]) -> dict[str, Any]:
    max_page_chars = int(cfg["generation"]["max_page_context_chars"])
    max_block_chars = int(cfg["generation"]["max_block_context_chars"])
    max_article_chars = int(cfg["generation"].get("max_article_context_chars", max_page_chars))
    if job.task_type == "domain_knowledge_corpus":
        max_pt_chars = int(cfg["generation"].get("max_pt_context_chars", max_page_chars))
        return {
            "journal_name": job.journal.journal_name,
            "article_title": job.page.article_title,
            "page_range": _page_range(job),
            "raw_text": _corpus_raw_text(job, max_pt_chars),
        }
    if job.task_type == "article_metadata_extraction":
        return {
            "source_text": truncate_text(job.page.full_text or clean_text(job.source), max_page_chars),
            "page_image": job.images[0] if job.images else job.page.page_image,
            "layout_blocks": job.source.get("page", {}).get("blocks", []),
        }
    if job.task_type == "two_column_reading_order_reconstruction":
        return {
            "page_image": job.images[0] if job.images else job.page.page_image,
            "ocr_text": truncate_text(_page_visible_text(job.page), max_page_chars),
            "paragraph_text": truncate_text(job.page.full_text, max_page_chars),
            "layout_blocks": [asdict(block) for block in job.page.blocks],
            "current_column_mode": job.page.column_mode,
            "current_reading_order_blocks": job.page.reading_order_blocks,
            "cross_column_barriers": _cross_column_barriers(job),
            "cross_column_barrier_policy": (
                "在两栏正文中，跨栏图表是纵向分隔点；读取左栏时若到达跨栏图表位置，"
                "应先转读同一纵向带内的右栏内容，再读取该跨栏图表，然后再进入图表下方的新双栏区域。"
            ),
        }
    if job.task_type == "page_to_journal_layout_description":
        return {
            "page_image": job.images[0] if job.images else job.page.page_image,
            "page_type": job.page.page_type,
            "column_mode": job.page.column_mode,
            "ocr_text": truncate_text(_page_visible_text(job.page), max_page_chars),
            "paragraph_text": truncate_text(job.page.full_text, max_page_chars),
            "layout_blocks": [asdict(block) for block in job.page.blocks],
        }
    if job.task_type in {"section_heading_scope_alignment", "section_keypoint_summary"}:
        heading = job.block.text if job.block else str(job.source.get("section_title") or job.page.article_title)
        section_text = _block_text(job.blocks[1:] if job.block and job.blocks else job.blocks, max_page_chars)
        return {
            "heading": clean_text(heading),
            "section_title": clean_text(heading),
            "controlled_text": section_text,
            "controlled_blocks": [asdict(block) for block in job.blocks],
            "section_text": section_text,
        }
    if job.task_type == "figure_table_formula_to_text":
        caption_text, related_context_text = _visual_context_text(job, max_block_chars)
        link = job.source.get("figure_link") if isinstance(job.source.get("figure_link"), dict) else {}
        caption_note = clean_text(link.get("caption_note")) if isinstance(link, dict) else ""
        if caption_note:
            caption_text = "\n".join(text for text in [caption_text, caption_note] if text)
        return {
            "target_image": job.images[0] if job.images else "",
            "target_block": asdict(job.block) if job.block else {},
            "caption_text": caption_text,
            "related_context_text": related_context_text,
            "context_binding_rule": job.source.get("context_binding_rule", ""),
        }
    if job.task_type == "method_experiment_condition_extraction":
        return {
            "article_title": job.page.article_title,
            "source_text": truncate_text(job.page.full_text, max_page_chars),
        }
    if job.task_type == "evidence_to_claim_chain":
        return {
            "page_image": job.images[0] if job.images else job.page.page_image,
            "article_title": job.page.article_title,
            "source_text": truncate_text(job.page.full_text, max_page_chars),
            "figure_links": job.page.figure_links,
        }
    if job.task_type == "article_contribution_conclusion":
        return {
            "article_title": job.page.article_title,
            "article_text": _window_text(job, max_article_chars),
            "scope": "article" if job.article and set(_source_pages(job)) >= set(job.article.page_indices[: len(_source_pages(job))]) else "article_fragment",
        }
    if job.task_type == "cross_page_article_context":
        return {
            "page_images": list(job.images[:3]),
            "page_texts": _page_texts(job, int(cfg["generation"]["max_neighbor_context_chars"])),
            "article_title": job.page.article_title,
            "image_count_policy": "Use 2-3 page images for this task.",
        }
    return {"source_text": truncate_text(job.page.full_text, max_page_chars)}


def _model_context(job: JournalSampleJob, cfg: dict[str, Any]) -> dict[str, Any]:
    max_neighbor_chars = int(cfg["generation"]["max_neighbor_context_chars"])
    page_window = [
        {
            "page_index": page.page_index,
            "page_type": page.page_type,
            "page_image": page.page_image,
            "full_text": truncate_text(page.full_text, max_neighbor_chars),
            "visible_text": truncate_text(_page_visible_text(page), max_neighbor_chars),
            "paragraphs": list(getattr(page, "paragraphs", [])),
            "column_mode": page.column_mode,
            "blocks": [asdict(block) for block in page.blocks],
            "figure_links": page.figure_links,
        }
        for page in job.page_window
    ]
    return {
        "template_input": _template_input(job, cfg),
        "page_ocr": truncate_text(_page_visible_text(job.page), int(cfg["generation"]["max_page_context_chars"])),
        "page_paragraph_text": truncate_text(job.page.full_text, int(cfg["generation"]["max_page_context_chars"])),
        "layout_blocks": [asdict(block) for block in job.page.blocks],
        "block": asdict(job.block) if job.block else {},
        "job_blocks": [asdict(block) for block in job.blocks],
        "page_window": page_window,
        "article": asdict(job.article) if job.article else {},
        "source": job.source,
    }


_DEDUPE_MIN_CHARS = 200


def _dedupe_payload(value: Any, seen: dict[str, str], path: str) -> Any:
    """把逐字节完全相同的子树替换成指向首次出现位置的短引用。

    同一份版面块会以 context.template_input.layout_blocks、context.layout_blocks、
    context.source.page.blocks 等多个键重复出现，整页正文也会以 ocr_text /
    paragraph_text / page_ocr / page_paragraph_text / full_text 反复出现。
    实测单页任务里 105K 字符有 51K 是纯重复，跨页任务直接把 prompt 顶到
    max_model_len 之外被服务端拒掉。

    这里只合并「序列化后逐字节相同」的子树，键本身保留、指向首次出现的位置，
    所以模型能看到的信息一条不少，只是不再重复看。任务提示词不受影响。
    """
    if isinstance(value, dict):
        return {key: _dedupe_payload(item, seen, f"{path}.{key}" if path else str(key)) for key, item in value.items()}
    if isinstance(value, list):
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if len(serialized) >= _DEDUPE_MIN_CHARS:
            first = seen.get(serialized)
            if first is not None:
                return f"<同 {first}，内容不再重复>"
            seen[serialized] = path
        return [_dedupe_payload(item, seen, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, str) and len(value) >= _DEDUPE_MIN_CHARS:
        first = seen.get(value)
        if first is not None:
            return f"<同 {first}，内容不再重复>"
        seen[value] = path
    return value


def _payload_chars(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, indent=2))


def _largest_shrinkable(context: dict[str, Any]) -> str:
    """挑 context 里最大的可裁剪键；template_input 是任务主输入，最后才动。"""
    sizes = {
        key: len(json.dumps(item, ensure_ascii=False, default=str))
        for key, item in context.items()
        if key != "template_input" and not str(item).startswith("<同 ")
    }
    return max(sizes, key=lambda key: sizes[key]) if sizes else ""


def _enforce_prompt_budget(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """超预算时按体积从大到小省略 context 下的辅助字段，直到装得下。

    不裁 template_input：那是任务本身的输入。辅助字段（source、page_window、
    layout_blocks 之类）省略后会留一句说明，而不是静默消失。
    """
    if max_chars <= 0:
        return payload
    context = payload.get("context")
    if not isinstance(context, dict):
        return payload
    for _ in range(len(context)):
        if _payload_chars(payload) <= max_chars:
            return payload
        key = _largest_shrinkable(context)
        if not key:
            break
        context[key] = f"<超出 prompt 预算 {max_chars} 字符，本字段已省略>"
    return payload


def _prompt(job: JournalSampleJob, cfg: dict[str, Any]) -> str:
    spec = TASK_SPECS[job.task_type]
    if spec["format"] == "pt":
        template_input = _template_input(job, cfg)
        prompt = str(cfg["prompts"][job.task_type]).strip()
        return (
            f"{prompt}\n\n{PT_JSONL_OUTPUT_INSTRUCTION}\n\n"
            "输入信息：\n"
            f"期刊：{template_input.get('journal_name', '')}\n"
            f"文章：{template_input.get('article_title', '')}\n"
            f"页码范围：{template_input.get('page_range', '')}\n"
            "恢复读序后的原始文本：\n"
            f"{template_input.get('raw_text', '')}"
        )

    payload = {
        "task_type": job.task_type,
        "target_format": spec["format"],
        "input_fields": spec["input_fields"],
        "output_fields": spec["output_fields"],
        "quality_rules": spec["quality_rules"],
        "source_info": _metadata(job, cfg, 0),
        "context": _model_context(job, cfg),
        "expected_count": cfg["generation"]["samples_per_job"].get(job.task_type, 1),
    }
    payload = _dedupe_payload(payload, {}, "")
    payload = _enforce_prompt_budget(payload, int(cfg["generation"].get("max_prompt_chars", 0)))
    prompt = str(cfg["prompts"][job.task_type]).strip()
    instruction_rule = (
        "请生成训练样本而不是直接解释任务。每条样本的 instruction 必须围绕当前任务和当前证据自然提问；"
        "输出对象中除 instruction、answer 和证据字段外，只保留 output_fields 指定字段。"
    )
    if job.task_type in SHAREGPT_TASK_TYPES:
        instruction_rule += (
            " 本任务是图文 ShareGPT 样本，必须由你根据当前图片和输入上下文生成 question/instruction 与 answer。"
            "answer 要用自然中文正面回答 instruction，尽量完整覆盖当前页面或目标图像中的主要信息，"
            "包括正文主题、版面/读序、表格、图、公式、图表中的可见内容和与论文上下文的关系；"
            "不要只评价图片是否可用，也不要输出字段名堆叠、JSON 或内部 block_id/bbox。"
            "如果图片与 OCR/上下文存在冲突，应在 answer 中说明可确认内容和不确定处。"
        )
    elif job.task_type in ALPACA_QA_TASK_TYPES:
        instruction_rule += (
            " 本任务是纯文本或文本为主的 Alpaca 问答样本，必须由你根据当前期刊论文上下文生成自然问题和自然回答。"
            "instruction 要忠于输入中的论文题名、小节、摘要、正文片段、方法条件或结论证据，不能泛泛写成“抽取字段”。"
            "answer 要正面回答 instruction，尽量完整覆盖输入上下文中的全部关键信息；"
            "可以按“主题、依据、结论、限制”等中文小标题结构化组织，但不要输出 JSON、字段清单、"
            "也不要出现 title/authors/summary/key_terms/research_object 等变量字段名。"
        )
        if job.task_type == "article_metadata_extraction":
            instruction_rule += (
                " 对元数据任务，answer 应用自然语言说明页面中可见的题名、作者、机构、期刊、年份卷期页码、摘要、关键词、基金或编号等信息；"
                "缺失信息只说明页面未见，不要编造。"
            )
        elif job.task_type == "section_heading_scope_alignment":
            instruction_rule += (
                " 对小节标题范围任务，answer 应说明标题控制了哪些正文内容、为什么匹配或不匹配、是否存在跨栏/图表/跨页造成的不确定性。"
            )
        elif job.task_type == "section_keypoint_summary":
            instruction_rule += (
                " 对小节要点任务，answer 应覆盖研究目的、方法步骤、参数/工况、结果性陈述、术语和结论，不要只给一句摘要。"
            )
        elif job.task_type == "method_experiment_condition_extraction":
            instruction_rule += (
                " 对方法与条件任务，answer 应说明研究对象、方法流程、实验或仿真条件、设备/变量/参数、评价指标和约束假设。"
            )
        elif job.task_type == "article_contribution_conclusion":
            instruction_rule += (
                " 对贡献结论任务，answer 应说明研究问题、主要贡献、关键发现、结论、适用条件/局限和支撑证据范围。"
            )
    if job.task_type == "figure_table_formula_to_text":
        instruction_rule += (
            " 特别注意：related_context_text 只代表与 target_block 绑定的上下文，不要使用页面上其他图表或公式的信息。"
            " 必须先依据当前目标图片/表格/公式生成 visual_description，描述可见结构、文字、表格行列、公式符号、曲线/箭头/标注等可见内容；"
            " answer 必须结合 visual_description、caption_text、related_context_text 和与目标对象相关的领域上下文进行证据化解读。"
            " 生成 instruction 时只能说“这张图片”“该表格”“该公式”等泛称，"
            "不得出现图号、表号、公式号、object_label、caption_text 中的题名、论文题名、小节标题或任何编号标题信息；"
            "这些编号和标题即使出现在上下文里，也只能用于 answer/output 字段的证据化解读，不能写进 instruction。"
        )
    elif job.task_type in {
        "two_column_reading_order_reconstruction",
        "page_to_journal_layout_description",
        "evidence_to_claim_chain",
    }:
        instruction_rule += (
            " 必须先观察当前页面图像，生成 page_visual_description，描述页面可见的标题、正文区域、栏式结构、表格、图、公式、页眉页脚等；"
            "问题要指向当前页面内容，答案要尽量覆盖该页所有主要信息，尤其不能遗漏页面中的表、图或公式。"
        )
        if job.task_type == "two_column_reading_order_reconstruction":
            instruction_rule += (
                " 对双栏正文，阅读顺序必须按纵向带恢复；跨栏图表会截断当前带，读左栏时到达跨栏图表位置后，"
                "先读同一带右栏内容，再读跨栏图表，然后再读图表下方的新带，不能把跨栏图表下方左栏提前到右栏之前。"
            )
    elif job.task_type == "cross_page_article_context":
        instruction_rule += (
            " 必须逐页观察 2-3 张连续页面图像，生成 page_visual_descriptions；"
            "问题要指向这些连续页面的上下文衔接，答案要覆盖各页主要正文、图表/表格/公式和跨页承接关系。"
        )
    return f"{prompt}\n\n{instruction_rule}\n\n{JSON_OUTPUT_INSTRUCTION}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def _normalize_instruction(raw: Any) -> str:
    return clean_text(raw).replace("<image>", "").strip()


def _looks_like_raw_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[:1] not in ("{", "["):
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return bool(re.match(r"^[{\[]\s*['\"]?(instruction|answer|output_payload|input_payload|metadata)", stripped))


def _normalize_answer(raw: Any) -> str:
    text = clean_text(raw).replace("<image>", "").strip()
    if not text or _looks_like_raw_json(text):
        return ""
    if re.search(r"\b(input_payload|output_payload|metadata|export_input|layout_blocks)\b\s*[:：]", text):
        return ""
    raw_field_names = {
        str(field)
        for spec in TASK_SPECS.values()
        for field in spec.get("output_fields", [])
        if str(field)
    }
    raw_field_pattern = r"\b(" + "|".join(re.escape(name) for name in sorted(raw_field_names)) + r")\b\s*[:：]"
    if re.search(raw_field_pattern, text, flags=re.IGNORECASE):
        return ""
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _block_lookup(job: JournalSampleJob) -> dict[str, Any]:
    pages = job.page_window or [job.page]
    result: dict[str, Any] = {}
    for page in pages:
        for block in page.blocks:
            result[block.block_id] = block
    return result


def _block_id(value: Any) -> str:
    text = clean_text(value)
    return text if re.fullmatch(r"p\d{1,4}_b\d{1,4}", text) else ""


def _block_to_content(block: Any) -> dict[str, Any]:
    content = truncate_text(clean_text(getattr(block, "markdown", "") or getattr(block, "text", "")), 520)
    role = clean_text(getattr(block, "semantic_role", "") or getattr(block, "block_type", ""))
    column = clean_text(getattr(block, "column", ""))
    item: dict[str, Any] = {"content": content}
    if role:
        item["role"] = role
    if column:
        item["column"] = column
    return item


def _content_from_block_id(value: Any, by_id: dict[str, Any]) -> str:
    block_id = _block_id(value)
    if not block_id or block_id not in by_id:
        return ""
    return clean_text(getattr(by_id[block_id], "markdown", "") or getattr(by_id[block_id], "text", ""))


def _dict_text(value: dict[str, Any]) -> str:
    for key in ("content", "text", "text_excerpt", "description", "summary", "key_point", "caption", "visible_structure"):
        text = clean_text(value.get(key))
        if text:
            return text
    return clean_text({key: item for key, item in value.items() if key not in {"block_id", "bbox", "blocks", "confidence"}})


def _normalize_text_field(value: Any, by_id: dict[str, Any], limit: int = 3200) -> str:
    if isinstance(value, str):
        return truncate_text(_content_from_block_id(value, by_id) or value, limit)
    if isinstance(value, dict):
        return truncate_text(_dict_text(value), limit)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(_content_from_block_id(item, by_id) or clean_text(item))
            elif isinstance(item, dict):
                block_text = _content_from_block_id(item.get("block_id"), by_id)
                parts.append(block_text or _dict_text(item))
            else:
                parts.append(clean_text(item))
        return truncate_text("\n".join(part for part in parts if part), limit)
    return truncate_text(clean_text(value), limit)


def _clean_output_item(item: Any, by_id: dict[str, Any]) -> Any:
    if isinstance(item, str):
        block_text = _content_from_block_id(item, by_id)
        return {"content": truncate_text(block_text, 520)} if block_text else item
    if isinstance(item, list):
        cleaned_items: list[Any] = []
        for child in item:
            cleaned = _clean_output_item(child, by_id)
            if _has_output_value(cleaned):
                cleaned_items.append(cleaned)
        return cleaned_items
    if not isinstance(item, dict):
        return item
    block_text = _content_from_block_id(item.get("block_id"), by_id)
    cleaned: dict[str, Any] = {}
    if block_text:
        cleaned["content"] = truncate_text(block_text, 520)
    for key, value in item.items():
        if key in {"block_id", "bbox", "blocks", "confidence", "image_path", "extracted_image_path", "markdown"}:
            continue
        target_key = "content" if key in {"text", "text_excerpt"} else key
        if target_key == "content" and cleaned.get("content"):
            continue
        if target_key in {"block_type", "semantic_role", "type"}:
            cleaned.setdefault("role", value)
            continue
        cleaned[target_key] = _clean_output_item(value, by_id) if isinstance(value, (dict, list)) else value
    return {key: value for key, value in cleaned.items() if _has_output_value(value)}


def _has_output_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _normalize_item_list(value: Any, by_id: dict[str, Any], limit: int = 18) -> list[Any]:
    values = value if isinstance(value, list) else [value]
    result: list[Any] = []
    for item in values[:limit]:
        cleaned = _clean_output_item(item, by_id)
        if _has_output_value(cleaned):
            result.append(cleaned)
    return result


def _visual_objects(job: JournalSampleJob) -> list[dict[str, Any]]:
    visual_roles = {"figure", "image", "table", "formula", "equation", "figure_caption", "table_caption", "caption"}
    pages = job.page_window or [job.page]
    result: list[dict[str, Any]] = []
    for page in pages:
        for block in page.blocks:
            if block.block_type not in visual_roles and block.semantic_role not in visual_roles:
                continue
            text = clean_text(block.markdown or block.text)
            item = {
                "page_index": page.page_index,
                "type": block.block_type,
                "description": truncate_text(text, 420) if text else f"页面中可见 {block.block_type} 区域",
            }
            result.append(item)
    return result[:12]


def _block_box(block: Any) -> tuple[float, float, float, float] | None:
    bbox = getattr(block, "bbox", None)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    left, top, right, bottom = [float(value) for value in bbox]
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _block_top(block: Any) -> float:
    box = _block_box(block)
    return box[1] if box else float(getattr(block, "reading_order", 9999) or 9999)


def _block_bottom(block: Any) -> float:
    box = _block_box(block)
    return box[3] if box else _block_top(block)


def _cross_column_barriers(job: JournalSampleJob) -> list[dict[str, Any]]:
    visual_types = {"figure", "table", "formula"}
    noise_roles = {"journal_header", "footer", "page_number", "reference", "unknown"}
    barriers = [
        block
        for block in job.page.blocks
        if block.block_type in visual_types
        and block.column == "full_width"
        and block.span_kind in {"full_width", "cross_column"}
        and _block_box(block) is not None
    ]
    barriers = sorted(barriers, key=lambda item: (_block_top(item), getattr(item, "reading_order", 9999) or 9999))
    result: list[dict[str, Any]] = []
    band_top = float("-inf")
    for barrier in barriers:
        barrier_top = _block_top(barrier)
        left_before = [
            block.block_id
            for block in job.page.blocks
            if block.column == "left" and block.semantic_role not in noise_roles and band_top <= _block_top(block) < barrier_top
        ]
        right_before = [
            block.block_id
            for block in job.page.blocks
            if block.column == "right" and block.semantic_role not in noise_roles and band_top <= _block_top(block) < barrier_top
        ]
        result.append(
            {
                "block_id": barrier.block_id,
                "block_type": barrier.block_type,
                "text_excerpt": truncate_text(clean_text(barrier.markdown or barrier.text), 180),
                "bbox": barrier.bbox,
                "left_column_blocks_before_barrier": left_before,
                "right_column_blocks_before_barrier": right_before,
                "reading_policy": "When scanning the left column, stop at this cross-column visual barrier; read the right-column blocks in the same vertical band before reading the barrier and before any left-column blocks below it.",
            }
        )
        band_top = _block_bottom(barrier)
    return result


def _page_visual_description(job: JournalSampleJob) -> str:
    page = job.page
    section_titles = [
        clean_text(block.text)
        for block in page.blocks
        if block.semantic_role == "section_heading" and clean_text(block.text)
    ][:4]
    visual_objects = _visual_objects(job)
    visual_text = "；".join(item["description"] for item in visual_objects[:5] if item.get("description"))
    parts = [
        f"该页为{page.page_type}，栏式结构为{page.column_mode}。",
        f"页面正文主题围绕{truncate_text(page.article_title or _first_sentence(page.full_text, 120), 160)}。",
    ]
    if section_titles:
        parts.append(f"可见小节包括：{'；'.join(section_titles)}。")
    if visual_text:
        parts.append(f"页面中可见的图表/公式信息包括：{visual_text}。")
    return clean_text(" ".join(parts))


def _page_visual_descriptions(job: JournalSampleJob) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []
    for page in job.page_window or [job.page]:
        proxy = JournalSampleJob(
            job.task_type,
            job.journal,
            page,
            article=job.article,
            page_window=[page],
            source=job.source,
            images=[page.page_image],
        )
        descriptions.append({"page_index": page.page_index, "description": _page_visual_description(proxy)})
    return descriptions


def _fallback_answer_text(task_type: str, output_payload: dict[str, Any]) -> str:
    if not output_payload:
        return ""
    ordered_keys = TASK_SPECS.get(task_type, {}).get("output_fields", list(output_payload.keys()))
    parts: list[str] = []
    for key in ordered_keys:
        if key in {"object_label", "filtered_noise_blocks"}:
            continue
        text = clean_text(output_payload.get(key))
        if text:
            parts.append(text)
    return truncate_text("；".join(parts), 2600)


def _normalize_output_payload(job: JournalSampleJob, output: dict[str, Any]) -> dict[str, Any]:
    by_id = _block_lookup(job)
    normalized = dict(output)
    task_type = job.task_type
    if task_type == "figure_table_formula_to_text":
        visual_description = _normalize_text_field(normalized.get("visual_description"), by_id, 1200)
        if not visual_description:
            visual_description = _normalize_text_field(normalized.get("visible_structure"), by_id, 1200)
        if not visual_description and job.block:
            visual_description = truncate_text(clean_text(job.block.markdown or job.block.text), 1200)
        normalized["visual_description"] = visual_description or "目标图表/公式区域可见，但模型未给出足够细节。"
        normalized["visible_structure"] = _normalize_text_field(normalized.get("visible_structure"), by_id, 1000) or normalized["visual_description"]
        normalized["caption_information"] = _normalize_text_field(normalized.get("caption_information"), by_id, 1000)
        normalized["text_explanation"] = _normalize_text_field(normalized.get("text_explanation"), by_id, 1200)
        normalized["evidence_to_claim"] = _normalize_text_field(normalized.get("evidence_to_claim"), by_id, 1000)
    elif task_type in {"two_column_reading_order_reconstruction", "page_to_journal_layout_description", "evidence_to_claim_chain"}:
        normalized["page_visual_description"] = _normalize_text_field(normalized.get("page_visual_description"), by_id, 1400) or _page_visual_description(job)
    elif task_type == "cross_page_article_context":
        if not _has_output_value(normalized.get("page_visual_descriptions")):
            normalized["page_visual_descriptions"] = _page_visual_descriptions(job)

    if task_type in {"two_column_reading_order_reconstruction", "page_to_journal_layout_description", "evidence_to_claim_chain", "cross_page_article_context"}:
        if not _has_output_value(normalized.get("figures_tables_formulas")):
            normalized["figures_tables_formulas"] = _visual_objects(job)
        else:
            normalized["figures_tables_formulas"] = _normalize_item_list(normalized.get("figures_tables_formulas"), by_id, 12)

    if task_type == "page_to_journal_layout_description":
        normalized["trainable_content"] = _normalize_text_field(normalized.get("trainable_content"), by_id, 4200)
        normalized["filtered_content"] = _normalize_text_field(normalized.get("filtered_content"), by_id, 1600)
        normalized["content_blocks"] = _normalize_item_list(normalized.get("content_blocks"), by_id, 24)
        normalized["layout_regions"] = _normalize_item_list(normalized.get("layout_regions"), by_id, 12)
    elif task_type == "two_column_reading_order_reconstruction":
        normalized["reading_order"] = _blocks_for_output(job)
        normalized["cross_column_barriers"] = _normalize_item_list(normalized.get("cross_column_barriers"), by_id, 12) or _cross_column_barriers(job)
        normalized["reconstructed_text"] = _normalize_text_field(normalized.get("reconstructed_text"), by_id, 4200) or truncate_text(job.page.full_text, 4200)
    elif task_type == "cross_page_article_context":
        normalized["page_roles"] = _normalize_item_list(normalized.get("page_roles"), by_id, 8)
        normalized["key_points_by_page"] = _normalize_item_list(normalized.get("key_points_by_page"), by_id, 8)
    return {key: value for key, value in normalized.items() if _has_output_value(value)}


def _split_output_and_evidence(row: dict[str, Any], job: JournalSampleJob) -> tuple[dict[str, Any], dict[str, Any]]:
    output_fields = set(TASK_SPECS[job.task_type]["output_fields"])
    evidence = {key: row.get(key) for key in EVIDENCE_KEYS if key in row}
    evidence.setdefault("source_pages", _source_pages(job))
    evidence.setdefault("evidence_block_ids", _evidence_block_ids(job))
    evidence.setdefault("evidence_text", _evidence_text(job))
    evidence.setdefault("visual_evidence", list(job.images))
    if job.task_type == "figure_table_formula_to_text":
        evidence.setdefault("figure_context_binding", job.source.get("figure_link", {}))
        evidence.setdefault("context_binding_rule", job.source.get("context_binding_rule", ""))
    output = {key: row.get(key) for key in output_fields if key in row}
    return _normalize_output_payload(job, output), evidence


def _fallback_enabled(cfg: dict[str, Any], reason: str) -> bool:
    fallback_cfg = cfg.get("generation", {}).get("heuristic_fallback", {})
    if not isinstance(fallback_cfg, dict) or not bool(fallback_cfg.get("enabled", True)):
        return False
    if reason == "skip_vlm":
        return bool(fallback_cfg.get("on_skip_vlm", False))
    return bool(fallback_cfg.get("on_vlm_error", False))


def _sentences(text: str, limit: int = 6) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。！？；;.!?])\s*", text) if part.strip()]
    if not parts:
        parts = [text]
    return parts[:limit]


def _first_sentence(text: str, limit: int = 220) -> str:
    sentences = _sentences(text, 1)
    return truncate_text(sentences[0] if sentences else text, limit)


def _keyword_terms(text: str, limit: int = 8) -> list[str]:
    text = clean_text(text)
    explicit = re.search(r"(关键词|Key\s*words?)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if explicit:
        candidates = re.split(r"[;；,，、\s]+", explicit.group(2))
    else:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-_/]{1,24}|[\u4e00-\u9fffA-Za-z0-9]{2,16}", text)
    stopwords = {"摘要", "关键词", "本文", "研究", "进行", "采用", "结果", "表明", "方法", "分析", "影响"}
    terms: list[str] = []
    for item in candidates:
        term = item.strip(" ：:;；,，。、()（）[]【】")
        if len(term) < 2 or term in stopwords:
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _sentences_with_keywords(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    result = [sentence for sentence in _sentences(text, 16) if any(keyword in sentence for keyword in keywords)]
    if not result:
        result = _sentences(text, limit)
    return [truncate_text(sentence, 260) for sentence in result[:limit]]


def _blocks_for_output(job: JournalSampleJob) -> list[dict[str, Any]]:
    blocks = job.blocks or ([job.block] if job.block else job.page.blocks)
    result = []
    for block in blocks:
        if block is None:
            continue
        result.append(
            {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "semantic_role": block.semantic_role,
                "column": block.column,
                "span_kind": block.span_kind,
                "text_excerpt": truncate_text(block.text, 180),
            }
        )
    return result


def _noise_block_ids(job: JournalSampleJob) -> list[str]:
    return [
        block.block_id
        for block in job.page.blocks
        if block.semantic_role in {"journal_header", "footer", "page_number", "reference", "unknown"}
    ]


def _layout_regions(job: JournalSampleJob) -> list[dict[str, Any]]:
    regions: dict[str, int] = {}
    for block in job.page.blocks:
        key = block.column or "unknown"
        regions[key] = regions.get(key, 0) + 1
    return [{"region": key, "block_count": count} for key, count in sorted(regions.items())]


def _visual_label(job: JournalSampleJob) -> str:
    link = job.source.get("figure_link") if isinstance(job.source.get("figure_link"), dict) else {}
    label = clean_text(link.get("object_label"))
    if label:
        return label
    target_text = clean_text(job.block.text if job.block else "")
    match = re.search(r"(图\s*[0-9一二三四五六七八九十]+|表\s*[0-9一二三四五六七八九十]+|式\s*[（(]?\s*[0-9]+|Fig\.?\s*[0-9]+|Table\s*[0-9]+)", target_text, flags=re.IGNORECASE)
    return clean_text(match.group(1)) if match else ""


def _base_evidence(job: JournalSampleJob) -> dict[str, Any]:
    evidence = {
        "source_pages": _source_pages(job),
        "evidence_block_ids": _evidence_block_ids(job),
        "evidence_text": _evidence_text(job),
        "visual_evidence": list(job.images),
    }
    if job.task_type == "figure_table_formula_to_text":
        evidence["figure_context_binding"] = job.source.get("figure_link", {})
        evidence["context_binding_rule"] = job.source.get("context_binding_rule", "")
    return evidence


def _article_metadata_output(job: JournalSampleJob) -> dict[str, Any]:
    article = job.article
    page_text = job.page.full_text
    title = clean_text((article.article_title if article else "") or job.page.article_title or _first_sentence(page_text, 80))
    return {
        "title": title,
        "authors": list(article.authors if article else []),
        "affiliations": list(article.affiliations if article else []),
        "journal_name": (article.journal_name if article else "") or job.journal.journal_name,
        "year": (article.year if article else "") or job.journal.year,
        "volume": (article.volume if article else "") or job.journal.volume,
        "issue": (article.issue if article else "") or job.journal.issue,
        "pages": article.pages if article else job.page.page_label,
        "doi": article.doi if article else "",
        "abstract": article.abstract if article else "",
        "keywords": list(article.keywords if article else _keyword_terms(page_text)),
        "classification_no": "",
        "document_code": "",
        "funding": "",
    }


def _heuristic_output(job: JournalSampleJob, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    text = _template_input(job, cfg)
    page_text = clean_text(job.page.full_text)
    if job.task_type == "article_metadata_extraction":
        return "请从该期刊论文首页内容中抽取文章元数据。", _article_metadata_output(job)
    if job.task_type == "two_column_reading_order_reconstruction":
        return "请根据该期刊页面的 OCR 块和坐标恢复双栏阅读顺序。", {
            "page_visual_description": _page_visual_description(job),
            "column_mode": job.page.column_mode,
            "filtered_noise_blocks": _noise_block_ids(job),
            "reading_order": _blocks_for_output(job),
            "cross_column_barriers": _cross_column_barriers(job),
            "reconstructed_text": truncate_text(job.page.full_text, 2600),
            "figures_tables_formulas": _visual_objects(job),
            "uncertainty_notes": list(job.page.uncertainty_notes),
        }
    if job.task_type == "page_to_journal_layout_description":
        trainable = "" if job.page.page_type in set(cfg.get("page_types", {}).get("low_value", [])) else truncate_text(job.page.full_text, 900)
        return "请描述该期刊页面的版面结构、页面类型和可训练内容。", {
            "page_visual_description": _page_visual_description(job),
            "page_type": job.page.page_type,
            "article_role": job.page.article_role,
            "column_mode": job.page.column_mode,
            "main_topic": job.page.article_title or _first_sentence(job.page.full_text, 120),
            "layout_regions": _layout_regions(job),
            "content_blocks": _blocks_for_output(job)[:18],
            "figures_tables_formulas": _visual_objects(job),
            "trainable_content": trainable,
            "filtered_content": _noise_block_ids(job),
        }
    if job.task_type == "section_heading_scope_alignment":
        controlled_text = clean_text(text.get("controlled_text"))
        heading = clean_text(text.get("heading"))
        return "请判断该论文小节标题与其后正文范围是否匹配。", {
            "heading": heading,
            "heading_level": "section",
            "controlled_block_ids": [block.block_id for block in job.blocks if block.block_id != (job.block.block_id if job.block else "")],
            "scope_summary": _first_sentence(controlled_text, 260),
            "alignment_judgement": "对齐",
            "mismatch_risk": "未发现明显不匹配风险" if "two_column" not in job.page.column_mode else "双栏或图表插入可能造成正文范围边界不确定",
        }
    if job.task_type == "section_keypoint_summary":
        section_text = clean_text(text.get("section_text"))
        return "请概括该期刊论文小节的研究要点。", {
            "section_title": clean_text(text.get("section_title")),
            "summary": _first_sentence(section_text, 320),
            "key_terms": _keyword_terms(section_text),
            "method_or_condition_points": _sentences_with_keywords(section_text, ["方法", "模型", "试验", "实验", "仿真", "参数", "工况", "条件"]),
            "result_or_claim_points": _sentences_with_keywords(section_text, ["结果", "表明", "结论", "影响", "验证", "提高", "降低"]),
        }
    if job.task_type == "figure_table_formula_to_text":
        caption_text = clean_text(text.get("caption_text"))
        related_text = clean_text(text.get("related_context_text"))
        target = job.block
        target_text = clean_text(target.text if target else "")
        caption_info = caption_text or target_text or "输入未提供明确图注、表注或公式说明。"
        explanation = related_text or "输入未提供与该对象同标签绑定的正文解释。"
        return "请基于这张图片及其绑定上下文进行证据化解读。", {
            "object_type": target.block_type if target else "unknown",
            "object_label": _visual_label(job),
            "visual_description": target_text or caption_info,
            "visible_structure": target_text or f"目标对象类型为 {target.block_type if target else 'unknown'}，bbox={target.bbox if target else []}",
            "caption_information": caption_info,
            "text_explanation": explanation,
            "evidence_to_claim": _first_sentence(related_text or caption_info, 320),
        }
    if job.task_type == "method_experiment_condition_extraction":
        source_text = clean_text(text.get("source_text"))
        return "请从该论文片段中抽取方法、实验或仿真条件。", {
            "research_object": job.page.article_title or _first_sentence(source_text, 120),
            "method_steps": _sentences_with_keywords(source_text, ["方法", "流程", "采用", "建立", "试验", "实验", "仿真", "计算"]),
            "experimental_or_simulation_conditions": _sentences_with_keywords(source_text, ["条件", "工况", "参数", "边界", "载荷", "速度", "温度", "压力"]),
            "variables_and_parameters": _keyword_terms(source_text, 12),
            "evaluation_metrics": _sentences_with_keywords(source_text, ["指标", "评价", "误差", "结果", "性能", "效率", "安全"]),
            "assumptions_or_constraints": _sentences_with_keywords(source_text, ["假设", "约束", "限制", "满足", "要求"], 3),
        }
    if job.task_type == "evidence_to_claim_chain":
        source_text = clean_text(text.get("source_text"))
        claim = _sentences_with_keywords(source_text, ["结果", "表明", "说明", "证明", "可见", "因此", "结论", "验证"], 1)[0]
        evidence_items = _evidence_text(job)[:4]
        return "请结合当前页面图像和论文片段构建证据到论点的链条。", {
            "page_visual_description": _page_visual_description(job),
            "claim": claim,
            "evidence_chain": evidence_items,
            "chain_steps": [
                {"step": 1, "description": "提取页面中的正文、图表或公式证据。", "evidence": evidence_items[:2]},
                {"step": 2, "description": "依据证据中出现的结果性表述归纳论点。", "evidence": [claim]},
            ],
            "supporting_figures_tables_formulas": [link.get("object_label") or link.get("target_block_id") for link in job.page.figure_links],
            "figures_tables_formulas": _visual_objects(job),
            "reasoning_scope": "page_fragment",
        }
    if job.task_type == "article_contribution_conclusion":
        article_text = clean_text(text.get("article_text"))
        base = _first_sentence(article_text, 220)
        return "请提炼该期刊论文片段的研究问题、贡献和结论。", {
            "research_problem": job.page.article_title or base,
            "contributions": _sentences_with_keywords(article_text, ["提出", "建立", "构建", "设计", "研究", "实现"], 4),
            "key_findings": _sentences_with_keywords(article_text, ["发现", "结果", "表明", "影响", "提高", "降低"], 4),
            "conclusions": _sentences_with_keywords(article_text, ["结论", "表明", "说明", "验证", "因此"], 4),
            "conditions_or_limitations": _sentences_with_keywords(article_text, ["条件", "限制", "范围", "假设", "工况", "参数"], 3),
            "supporting_evidence": _evidence_text(job)[:6],
        }
    if job.task_type == "cross_page_article_context":
        page_texts = text.get("page_texts") if isinstance(text.get("page_texts"), list) else []
        key_points = [
            {"page_index": item.get("page_index"), "key_point": _first_sentence(str(item.get("text") or ""), 220)}
            for item in page_texts
            if isinstance(item, dict)
        ]
        return "请归纳连续多页同一篇论文之间的上下文衔接。", {
            "page_visual_descriptions": _page_visual_descriptions(job),
            "context_topic": job.page.article_title or (key_points[0]["key_point"] if key_points else ""),
            "page_roles": [
                {"page_index": item.get("page_index"), "role": "延续同一篇论文的论述片段"}
                for item in page_texts
                if isinstance(item, dict)
            ],
            "cross_page_summary": " ".join(item["key_point"] for item in key_points if item.get("key_point")),
            "continuity_relations": ["连续页面属于同一 article_id，按页序承接正文、图表或结论信息。"],
            "figures_tables_formulas": _visual_objects(job),
            "key_points_by_page": key_points,
        }
    return None


def _heuristic_sample(
    job: JournalSampleJob,
    cfg: dict[str, Any],
    instruction: str,
    output_payload: dict[str, Any],
    fallback_error: str = "",
) -> dict[str, Any]:
    metadata = _metadata(job, cfg, 1)
    metadata["generator"] = "heuristic_vlm_fallback"
    metadata["generator_provider"] = "heuristic"
    metadata["generator_model"] = "rule"
    if fallback_error:
        metadata["fallback_error"] = truncate_text(fallback_error, 500)
    template_input = _template_input(job, cfg)
    sample = {
        "id": _sample_id(metadata),
        "task_type": job.task_type,
        "instruction": instruction,
        "input_payload": template_input,
        "output_payload": output_payload,
        "evidence": _base_evidence(job),
        "metadata": metadata,
        "images": list(job.images),
    }
    sample["question"] = instruction
    sample["answer"] = _fallback_answer_text(job.task_type, output_payload) or json.dumps(output_payload, ensure_ascii=False, indent=2)
    if TASK_SPECS[job.task_type]["format"] == "alpaca":
        sample["export_input"] = _format_alpaca_input(job.task_type, template_input)
    return sample


def generate_heuristic_for_job(
    job: JournalSampleJob,
    cfg: dict[str, Any],
    fallback_error: str = "",
) -> list[dict[str, Any]]:
    if job.task_type == "domain_knowledge_corpus":
        template_input = _template_input(job, cfg)
        raw_text = _clean_pt_corpus_text(str(template_input.get("raw_text") or ""))
        if not raw_text:
            return []
        article_title = clean_text(template_input.get("article_title"))
        header = (
            f"期刊：{template_input.get('journal_name', '')}\n"
            f"文章：{article_title or job.page.article_title or job.page.article_id}\n"
            f"页码：{template_input.get('page_range', '')}\n\n"
        )
        text = header + raw_text
        rows = [{"text": text}]
        samples = _pt_samples_from_rows(rows, job, cfg, "heuristic", "rule")
        for sample in samples:
            metadata = sample.setdefault("metadata", {})
            metadata["generator"] = "rule_based_pt_cleaner"
            if fallback_error:
                metadata["generator"] = "heuristic_vlm_fallback"
                metadata["fallback_error"] = truncate_text(fallback_error, 500)
        return samples

    generated = _heuristic_output(job, cfg)
    if generated is None:
        return []
    instruction, output_payload = generated
    return [_heuristic_sample(job, cfg, instruction, output_payload, fallback_error)]


def _pt_samples_from_rows(
    rows: list[dict[str, Any]],
    job: JournalSampleJob,
    cfg: dict[str, Any],
    response_provider: str,
    response_model: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    template_input = _template_input(job, cfg)
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        metadata = _metadata(job, cfg, index)
        metadata["generator_provider"] = response_provider
        metadata["generator_model"] = response_model
        metadata["target_format"] = "pt"
        metadata["llamafactory_stage"] = "pt"
        evidence = {
            "source_pages": _source_pages(job),
            "evidence_block_ids": _evidence_block_ids(job),
            "evidence_text": _evidence_text(job),
            "visual_evidence": [],
        }
        sample = {
            "id": _sample_id(metadata),
            "task_type": job.task_type,
            "input_payload": template_input,
            "output_payload": {"text": text},
            "evidence": evidence,
            "metadata": metadata,
            "images": [],
            "text": text,
            "question": "",
            "answer": text,
            "export_input": template_input.get("raw_text", ""),
        }
        samples.append(sample)
    return samples


def generate_for_job(job: JournalSampleJob, output_dir: Path, cfg: dict[str, Any], vlm: VlmPool | None) -> list[dict[str, Any]]:
    if job.task_type == "domain_knowledge_corpus":
        return generate_heuristic_for_job(job, cfg)

    if vlm is None or bool(cfg["runtime"]["skip_vlm"]):
        if _fallback_enabled(cfg, "skip_vlm"):
            return generate_heuristic_for_job(job, cfg, "skip_vlm")
        return []

    image_paths = [output_dir / image for image in job.images if (output_dir / image).exists()]
    try:
        response = vlm.chat(task_type=job.task_type, prompt=_prompt(job, cfg), images=image_paths)
    except Exception as exc:
        if _fallback_enabled(cfg, "vlm_error"):
            return generate_heuristic_for_job(job, cfg, str(exc))
        raise
    if TASK_SPECS[job.task_type]["format"] == "pt":
        try:
            rows = parse_jsonl_objects(response.text)
        except Exception as exc:
            if _fallback_enabled(cfg, "vlm_error"):
                return generate_heuristic_for_job(job, cfg, str(exc))
            raise
        return _pt_samples_from_rows(rows, job, cfg, response.provider_name, response.model)

    try:
        rows = parse_json_array(response.text)
    except Exception as exc:
        if _fallback_enabled(cfg, "vlm_error"):
            return generate_heuristic_for_job(job, cfg, str(exc))
        raise
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        output_payload, evidence = _split_output_and_evidence(row, job)
        if not output_payload:
            continue
        metadata = _metadata(job, cfg, index)
        metadata["generator_provider"] = response.provider_name
        metadata["generator_model"] = response.model
        metadata["target_format"] = TASK_SPECS[job.task_type]["format"]
        instruction = _normalize_instruction(row.get("instruction"))
        if not instruction:
            continue
        answer = _normalize_answer(row.get("answer"))
        if not answer:
            answer = _fallback_answer_text(job.task_type, output_payload)
        template_input = _template_input(job, cfg)
        sample = {
            "id": _sample_id(metadata),
            "task_type": job.task_type,
            "instruction": instruction,
            "input_payload": template_input,
            "output_payload": output_payload,
            "evidence": evidence,
            "metadata": metadata,
            "images": list(job.images),
        }
        sample["question"] = instruction
        sample["answer"] = answer or json.dumps(output_payload, ensure_ascii=False, indent=2)
        if TASK_SPECS[job.task_type]["format"] == "alpaca":
            sample["export_input"] = _format_alpaca_input(job.task_type, template_input)
        samples.append(sample)
    return samples


def _format_alpaca_input(task_type: str, payload: dict[str, Any]) -> str:
    if task_type == "article_metadata_extraction":
        return str(payload.get("source_text") or "")
    if task_type == "section_heading_scope_alignment":
        return f"小节标题：{payload.get('heading', '')}\n\n正文范围：{payload.get('controlled_text', '')}".strip()
    if task_type == "section_keypoint_summary":
        return f"小节标题：{payload.get('section_title', '')}\n\n小节正文：{payload.get('section_text', '')}".strip()
    if task_type == "method_experiment_condition_extraction":
        return str(payload.get("source_text") or "")
    if task_type == "article_contribution_conclusion":
        return f"文章题名：{payload.get('article_title', '')}\n\n正文片段：{payload.get('article_text', '')}".strip()
    return json.dumps(payload, ensure_ascii=False, indent=2)
