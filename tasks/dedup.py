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


def _is_near(overlap: int, size: int, seen_size: int, threshold: float) -> bool:
    union = size + seen_size - overlap
    jaccard = overlap / max(1, union)
    containment = overlap / max(1, min(size, seen_size))
    return max(jaccard, containment) >= threshold


def _near_duplicate(text: str, seen_texts: list[str], threshold: float) -> bool:
    """逐条比较版本，保留给外部调用方；deduplicate 内部走倒排索引。"""
    grams = _char_ngrams(text)
    if not grams:
        return False
    for seen in seen_texts:
        seen_grams = _char_ngrams(seen)
        if not seen_grams:
            continue
        if _is_near(len(grams & seen_grams), len(grams), len(seen_grams), threshold):
            return True
    return False


class _NearDuplicateIndex:
    """n-gram 倒排索引。

    逐条比较要对每个已见文本重算 gram set 再做集合交并，是 O(n²) 次集合运算，
    1200 条 PT 样本要跑四分多钟。这里维护 gram -> 已见文档号 的倒排表，
    扫一遍新样本自己的 n-gram 就能同时得到它与**所有**已见文本的精确交集大小，
    再套同一个 jaccard/containment 判据。

    是精确等价，不是近似：交集大小逐个都对得上，判定结果与逐条比较完全一致。
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._postings: dict[str, list[int]] = {}
        self._sizes: list[int] = []

    def add_if_new(self, text: str) -> bool:
        """不是近重则收录并返回 True。"""
        grams = _char_ngrams(text)
        if not grams:
            # 与逐条比较一致：空文本不参与近重判定，也不进索引（逐条版会跳过空 gram 的已见项）。
            return True
        overlaps: dict[int, int] = {}
        for gram in grams:
            for doc in self._postings.get(gram, ()):
                overlaps[doc] = overlaps.get(doc, 0) + 1
        size = len(grams)
        for doc, overlap in overlaps.items():
            if _is_near(overlap, size, self._sizes[doc], self.threshold):
                return False
        doc = len(self._sizes)
        self._sizes.append(size)
        for gram in grams:
            self._postings.setdefault(gram, []).append(doc)
        return True


def deduplicate(samples: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    per_page_task: dict[tuple[str, str, int, str], int] = {}
    max_per_page = int(cfg["validation"]["max_samples_per_page_task"])
    similarity_threshold = float(cfg.get("validation", {}).get("dedup_similarity_threshold", 0.92))
    pt_index = _NearDuplicateIndex(similarity_threshold)
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
            if not pt_index.add_if_new(text_basis):
                continue
        seen_ids.add(sample_id)
        seen_text.add(text_key)
        per_page_task[key] = per_page_task.get(key, 0) + 1
        result.append(sample)
    return result
