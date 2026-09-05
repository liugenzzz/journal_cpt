from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _normalized_text(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or "").lower())
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _char_ngrams(value: str, n: int = 3) -> set[str]:
    text = _normalized_text(value)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _near_duplicate(text: str, seen_texts: list[str], threshold: float) -> bool:
    grams = _char_ngrams(text)
    if not grams:
        return False
    for seen in seen_texts:
        seen_grams = _char_ngrams(seen)
        if not seen_grams:
            continue
        overlap = len(grams & seen_grams)
        union = len(grams | seen_grams)
        jaccard = overlap / max(1, union)
        containment = overlap / max(1, min(len(grams), len(seen_grams)))
        if max(jaccard, containment) >= threshold:
            return True
    return False


def deduplicate(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    seen_pt_texts: list[str] = []
    per_page_task: dict[tuple[str, str, int, str], int] = {}
    max_per_page = int(cfg["validation"]["max_samples_per_page_task"])
    similarity_threshold = float(cfg.get("validation", {}).get("dedup_similarity_threshold", 0.92))
    result: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id"))
        if sample_id in seen_ids:
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        key = (
            str(metadata.get("journal_id")),
            str(metadata.get("article_id")),
            int(metadata.get("page_index") or 0),
            str(metadata.get("task_type")),
        )
        if per_page_task.get(key, 0) >= max_per_page:
            continue
        payload = sample.get("output_payload")
        if isinstance(payload, dict) and payload.get("text"):
            text_basis = str(payload.get("text"))
        else:
            text_basis = (
                f"{sample.get('question')}||{sample.get('answer')}||"
                f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
            )
        text_key = hashlib.sha256(text_basis.encode("utf-8")).hexdigest()
        if text_key in seen_text:
            continue
        if str(sample.get("task_type") or "") == "domain_knowledge_corpus":
            if _near_duplicate(text_basis, seen_pt_texts, similarity_threshold):
                continue
            seen_pt_texts.append(text_basis)
        seen_ids.add(sample_id)
        seen_text.add(text_key)
        per_page_task[key] = per_page_task.get(key, 0) + 1
        result.append(sample)
    return result
