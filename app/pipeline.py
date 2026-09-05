from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..core.config_loader import deep_merge, load_config, package_root
from ..core.io_utils import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from ..core.logging_utils import RunState, configure_logger
from ..core.progress import ProgressBar
from ..core.models import PipelineOptions
from ..processing.crop import crop_blocks
from ..processing.ingest import scan_input_journals, scan_journals
from ..processing.normalize import normalize_pages
from ..processing.render import render_pages
from ..processing.watermark import UnreadablePdfError, clean_pdf_watermarks
from ..services.clients import VlmPool
from ..services.mineru import mineru_cache_is_valid, parse_journal_with_mineru
from ..tasks.dedup import deduplicate
from ..tasks.exporters import export_task_files
from ..tasks.generation import generate_for_job
from ..tasks.routing import build_sample_jobs
from ..tasks.validation import validate_sample


class GenerationBatchError(RuntimeError):
    """Raised when every model generation job in a batch failed."""


def _apply_options(cfg: dict[str, Any], options: PipelineOptions) -> dict[str, Any]:
    override: dict[str, Any] = {"runtime": {}}
    if options.input_dir is not None:
        override["runtime"]["input_dir"] = str(options.input_dir)
    if options.input_journals is not None:
        override["runtime"]["input_journals"] = options.input_journals
    if options.output_root is not None:
        override["runtime"]["output_root"] = str(options.output_root)
    if options.reuse_mineru is not None:
        override["runtime"]["reuse_mineru"] = bool(options.reuse_mineru)
    if options.skip_vlm is not None:
        override["runtime"]["skip_vlm"] = bool(options.skip_vlm)
    if options.journal_workers is not None:
        override["runtime"]["journal_workers"] = int(options.journal_workers)
    if options.max_workers is not None:
        override["runtime"]["max_workers"] = int(options.max_workers)
    if options.page_workers is not None:
        override["runtime"]["page_workers"] = int(options.page_workers)
    if options.crop_workers is not None:
        override["runtime"]["crop_workers"] = int(options.crop_workers)
    if options.reuse_normalized is not None:
        override["runtime"]["reuse_normalized"] = bool(options.reuse_normalized)
    if options.reuse_samples is not None:
        override["runtime"]["reuse_samples"] = bool(options.reuse_samples)
    if options.reuse_exports is not None:
        override["runtime"]["reuse_exports"] = bool(options.reuse_exports)
    if options.recursive is not None:
        override["runtime"]["recursive"] = bool(options.recursive)
    if options.log_level is not None:
        override["runtime"]["log_level"] = str(options.log_level)
    if options.progress is not None:
        override["runtime"]["progress"] = bool(options.progress)
    if options.force_rebuild is not None:
        override["runtime"]["force_rebuild"] = bool(options.force_rebuild)
    if options.mineru_workers is not None:
        override.setdefault("mineru", {})["max_concurrency"] = int(options.mineru_workers)
    if options.mineru_retry_count is not None:
        override.setdefault("mineru", {})["retry_count"] = int(options.mineru_retry_count)
    if options.mineru_min_page_coverage is not None:
        override.setdefault("mineru", {})["min_page_coverage"] = float(options.mineru_min_page_coverage)
    if options.min_page_text_chars is not None:
        override.setdefault("routing", {})["min_page_text_chars"] = int(options.min_page_text_chars)
    if options.min_block_text_chars is not None:
        override.setdefault("routing", {})["min_block_text_chars"] = int(options.min_block_text_chars)
    cfg = deep_merge(cfg, override)
    if options.mineru_workers is not None:
        # providers 里的 max_concurrency 会盖掉 mineru 顶层默认值，命令行必须逐个写进去。
        for provider in cfg.get("mineru", {}).get("providers") or []:
            if isinstance(provider, dict):
                provider["max_concurrency"] = int(options.mineru_workers)
    if bool(cfg["runtime"].get("force_rebuild")):
        cfg["runtime"]["reuse_mineru"] = False
        cfg["runtime"]["reuse_normalized"] = False
        cfg["runtime"]["reuse_samples"] = False
        cfg["runtime"]["reuse_exports"] = False
    if options.task_types is not None:
        cfg = _select_task_types(cfg, options.task_types)
    return cfg


def _select_task_types(cfg: dict[str, Any], task_types: list[str]) -> dict[str, Any]:
    requested = [str(task_type).strip() for task_type in task_types if str(task_type).strip()]
    known_tasks = list(cfg.get("task_types", []))
    known_task_set = {str(task_type) for task_type in known_tasks}
    unknown = [task_type for task_type in requested if task_type not in known_task_set]
    if unknown:
        raise ValueError(f"Unknown task type(s): {', '.join(unknown)}")
    selected = list(dict.fromkeys(requested))
    if not selected:
        raise ValueError("At least one task type must be selected.")

    selected_set = set(selected)
    cfg["task_types"] = [task_type for task_type in known_tasks if task_type in selected_set]
    for key, task_list in list(cfg.get("export_formats", {}).items()):
        if isinstance(task_list, list):
            cfg["export_formats"][key] = [task_type for task_type in task_list if task_type in selected_set]
    enabled_tasks = cfg.setdefault("routing", {}).setdefault("enabled_tasks", {})
    for task_type in known_tasks:
        enabled_tasks[task_type] = task_type in selected_set
    for provider in cfg.get("vlm_pool", {}).get("providers", []):
        if not isinstance(provider, dict):
            continue
        provider_tasks = provider.get("task_types")
        if isinstance(provider_tasks, list):
            provider["task_types"] = [task_type for task_type in provider_tasks if task_type in selected_set]
    return cfg


def _resolve_config_path(value: str, base_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _resolve_config_runtime_paths(cfg: dict[str, Any], options: PipelineOptions) -> dict[str, Any]:
    base_dir = Path(options.config_path).resolve().parent if options.config_path else package_root().parent
    if options.input_dir is None:
        cfg["runtime"]["input_dir"] = _resolve_config_path(str(cfg["runtime"]["input_dir"]), base_dir)
    if options.output_root is None:
        cfg["runtime"]["output_root"] = _resolve_config_path(str(cfg["runtime"]["output_root"]), base_dir)
    if options.input_journals is None:
        input_journals = cfg["runtime"].get("input_journals")
        if isinstance(input_journals, list):
            for item in input_journals:
                if not isinstance(item, dict):
                    continue
                raw_path = item.get("path") or item.get("source_pdf") or item.get("address") or item.get("url")
                if raw_path and not Path(str(raw_path)).is_absolute():
                    item["path"] = _resolve_config_path(str(raw_path), base_dir)
    return cfg


def _generation_state_path(output_dir: Path, cfg: dict[str, Any]) -> Path:
    return output_dir / cfg["paths"].get("sample_generation_state", "samples/generation_state.json")


def _job_resume_key_from_values(
    *,
    task_type: str,
    journal_id: str,
    article_id: str,
    page_index: int,
    block_id: str,
) -> str:
    return f"{task_type}|{journal_id}|{article_id}|{page_index}|{block_id}"


def _job_resume_key(job: Any, cfg: dict[str, Any]) -> str:
    article = getattr(job, "article", None)
    page = getattr(job, "page", None)
    block = getattr(job, "block", None)
    article_id = getattr(article, "article_id", "") or getattr(page, "article_id", "")
    page_index = int(getattr(page, "page_index", 0) or 0)
    block_id = getattr(block, "block_id", "") or cfg.get("default_block_id", "page")
    return _job_resume_key_from_values(
        task_type=str(getattr(job, "task_type", "")),
        journal_id=str(getattr(getattr(job, "journal", None), "journal_id", "")),
        article_id=str(article_id),
        page_index=page_index,
        block_id=str(block_id),
    )


def _sample_resume_key(sample: dict[str, Any], cfg: dict[str, Any]) -> str:
    metadata = sample.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    task_type = str(sample.get("task_type") or metadata.get("task_type") or "")
    page_index = int(metadata.get("page_index") or 0)
    return _job_resume_key_from_values(
        task_type=task_type,
        journal_id=str(metadata.get("journal_id") or ""),
        article_id=str(metadata.get("article_id") or ""),
        page_index=page_index,
        block_id=str(metadata.get("block_id") or cfg.get("default_block_id", "page")),
    )


def _state_logger(cfg: dict[str, Any]) -> logging.Logger:
    return logging.getLogger(str(cfg.get("logger_name") or __name__))


def _read_generation_state(output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = _generation_state_path(output_dir, cfg)
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception as exc:
        # 读不动就等于"没有断点"，整批样本会重新生成。必须说出来，
        # 否则用户只会看到莫名其妙的全量重跑。
        _state_logger(cfg).warning("generation checkpoint unreadable, regenerating from scratch path=%s error=%s", path, exc)
        return {}
    if not isinstance(payload, dict):
        _state_logger(cfg).warning("generation checkpoint is not an object, regenerating from scratch path=%s", path)
        return {}
    return payload


def _generation_state_matches(payload: dict[str, Any], cfg: dict[str, Any], mineru_content_hash: str) -> bool:
    return (
        bool(mineru_content_hash)
        and payload.get("mineru_content_hash") == mineru_content_hash
        and payload.get("pipeline_version") == cfg.get("pipeline_version", "")
    )


def _completed_generation_jobs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    completed = payload.get("completed_jobs")
    if not isinstance(completed, dict):
        return {}
    return {str(key): value for key, value in completed.items() if isinstance(value, dict)}


def _write_generation_state(
    output_dir: Path,
    cfg: dict[str, Any],
    mineru_content_hash: str,
    completed_jobs: dict[str, dict[str, Any]],
) -> None:
    write_json(
        _generation_state_path(output_dir, cfg),
        {
            "mineru_content_hash": mineru_content_hash,
            "pipeline_version": cfg.get("pipeline_version", ""),
            "completed_jobs": completed_jobs,
        },
    )


def _load_resume_raw_rows(
    output_dir: Path,
    cfg: dict[str, Any],
    job_index_by_key: dict[str, int],
) -> tuple[list[tuple[int, int, dict[str, Any]]], dict[str, int]]:
    rows: list[tuple[int, int, dict[str, Any]]] = []
    counts_by_key: dict[str, int] = {}
    for task_type in cfg["task_types"]:
        path = output_dir / cfg["paths"]["sample_raw"].format(task_type=task_type)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    sample = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(sample, dict):
                    continue
                key = _sample_resume_key(sample, cfg)
                if key not in job_index_by_key:
                    continue
                sample_index = counts_by_key.get(key, 0)
                counts_by_key[key] = sample_index + 1
                rows.append((job_index_by_key[key], sample_index, sample))
    return rows, counts_by_key


def _generate_samples_for_jobs(
    *,
    jobs: list[Any],
    output_dir: Path,
    cfg: dict[str, Any],
    vlm: VlmPool | None,
    state: RunState,
    mineru_content_hash: str = "",
) -> list[dict[str, Any]]:
    raw_paths = {
        str(task_type): output_dir / cfg["paths"]["sample_raw"].format(task_type=task_type)
        for task_type in cfg["task_types"]
    }

    raw_sample_rows: list[tuple[int, int, dict[str, Any]]] = []
    failures: list[dict[str, str]] = []
    max_workers = max(1, int(cfg["runtime"]["max_workers"]))
    max_pending = max(max_workers, int(cfg["runtime"].get("vlm_max_pending", max_workers)))
    resume_enabled = bool(cfg["runtime"].get("reuse_samples")) and bool(mineru_content_hash)
    job_keys = [_job_resume_key(job, cfg) for job in jobs]
    job_index_by_key = {key: index for index, key in enumerate(job_keys)}
    state_payload = _read_generation_state(output_dir, cfg) if resume_enabled else {}
    state_matches = resume_enabled and _generation_state_matches(state_payload, cfg, mineru_content_hash)
    completed_jobs = _completed_generation_jobs(state_payload) if state_matches else {}
    raw_counts_by_key: dict[str, int] = {}

    if state_matches:
        raw_sample_rows, raw_counts_by_key = _load_resume_raw_rows(output_dir, cfg, job_index_by_key)
    else:
        for task_type in cfg["task_types"]:
            write_jsonl(raw_paths[str(task_type)], [])
        if mineru_content_hash:
            _write_generation_state(output_dir, cfg, mineru_content_hash, {})
        completed_jobs = {}

    def add_samples(job_index: int, samples: list[dict[str, Any]]) -> None:
        for sample_index, sample in enumerate(samples):
            raw_sample_rows.append((job_index, sample_index, sample))
            task_type = str(sample.get("task_type") or "")
            raw_path = raw_paths.setdefault(
                task_type,
                output_dir / cfg["paths"]["sample_raw"].format(task_type=task_type),
            )
            append_jsonl(raw_path, sample)

    def sample_count_is_resumable(key: str) -> bool:
        raw_count = raw_counts_by_key.get(key, 0)
        checkpoint = completed_jobs.get(key, {})
        sample_count = checkpoint.get("sample_count")
        if isinstance(sample_count, int):
            return sample_count == 0 or raw_count >= sample_count
        return raw_count > 0

    flush_every = max(1, int(cfg["runtime"].get("generation_state_flush_every", 20)))
    flush_seconds = max(0.0, float(cfg["runtime"].get("generation_state_flush_seconds", 10.0)))
    unflushed = 0
    last_flush_at = time.monotonic()

    def flush_generation_state(force: bool = False) -> None:
        """攒批落盘。

        原来每完成一个 job 就整份重写一次 checkpoint，n 个 job 要序列化 n²/2 个条目，
        而且把"崩在写文件中间"的窗口放到最大。丢掉最后几条 checkpoint 是安全的：
        样本本身已经追加进 samples/raw，sample_count_is_resumable 在没有 checkpoint
        条目时会退回按 raw 行数判断，只有"产出 0 条样本"的 job 会被重跑。
        """
        nonlocal unflushed, last_flush_at
        if not mineru_content_hash or unflushed == 0:
            return
        if not force and unflushed < flush_every and time.monotonic() - last_flush_at < flush_seconds:
            return
        _write_generation_state(output_dir, cfg, mineru_content_hash, completed_jobs)
        unflushed = 0
        last_flush_at = time.monotonic()

    def mark_job_completed(job: Any, job_key: str, samples: list[dict[str, Any]]) -> None:
        nonlocal unflushed
        if not mineru_content_hash:
            return
        completed_jobs[job_key] = {
            "task_type": str(getattr(job, "task_type", "")),
            "journal_id": str(getattr(getattr(job, "journal", None), "journal_id", "")),
            "sample_count": len(samples),
        }
        unflushed += 1
        flush_generation_state()

    def ordered_raw_samples() -> list[dict[str, Any]]:
        ordered_rows = sorted(raw_sample_rows, key=lambda item: (item[0], item[1]))
        return [sample for _, _, sample in ordered_rows]

    def record_failure(job: Any, exc: Exception) -> None:
        failures.append(
            {
                "journal_id": str(job.journal.journal_id),
                "task_type": str(job.task_type),
                "error": str(exc),
            }
        )
        state.error(
            {
                "stage": "generate",
                "journal_id": job.journal.journal_id,
                "task_type": job.task_type,
                "error": str(exc),
            }
        )

    def raise_if_all_failed() -> None:
        if not jobs or raw_sample_rows or len(failures) < len(jobs):
            return
        task_counts: dict[str, int] = {}
        for failure in failures:
            task_type = failure["task_type"]
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        task_summary = ", ".join(f"{task}:{count}" for task, count in sorted(task_counts.items()))
        unique_errors = list(dict.fromkeys(failure["error"] for failure in failures))
        error_summary = " | ".join(unique_errors[:5])
        if len(unique_errors) > 5:
            error_summary += f" | ... {len(unique_errors) - 5} more"
        journal_id = failures[0]["journal_id"]
        raise GenerationBatchError(
            "all VLM generation jobs failed "
            f"journal_id={journal_id} jobs={len(jobs)} tasks=[{task_summary}] errors=[{error_summary}]"
        )

    if max_workers == 1 or len(jobs) <= 1:
        for job_index, job in enumerate(jobs):
            job_key = job_keys[job_index]
            if sample_count_is_resumable(job_key):
                continue
            try:
                samples = generate_for_job(job, output_dir, cfg, vlm)
                add_samples(job_index, samples)
                mark_job_completed(job, job_key, samples)
            except Exception as exc:
                record_failure(job, exc)
        flush_generation_state(force=True)
        raise_if_all_failed()
        return ordered_raw_samples()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Any, Any] = {}
        job_iter = iter(enumerate(jobs))
        exhausted = False
        while futures or not exhausted:
            while not exhausted and len(futures) < max_pending:
                try:
                    job_index, job = next(job_iter)
                except StopIteration:
                    exhausted = True
                    break
                if sample_count_is_resumable(job_keys[job_index]):
                    continue
                futures[executor.submit(generate_for_job, job, output_dir, cfg, vlm)] = (job_index, job)
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job_index, job = futures.pop(future)
                job_key = job_keys[job_index]
                try:
                    samples = future.result()
                except Exception as exc:
                    record_failure(job, exc)
                    continue
                add_samples(job_index, samples)
                mark_job_completed(job, job_key, samples)
    flush_generation_state(force=True)
    raise_if_all_failed()
    return ordered_raw_samples()


def _expected_export_paths(output_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for task_type in cfg["task_types"]:
        paths.extend(_expected_export_paths_for_task(output_dir, cfg, task_type))
    return paths


def _expected_export_paths_for_task(output_dir: Path, cfg: dict[str, Any], task_type: str) -> list[Path]:
    paths: list[Path] = []
    format_paths = {
        "sharegpt": "export_sharegpt",
        "alpaca": "export_alpaca",
        "pt": "export_pt",
    }
    for format_name, path_key in format_paths.items():
        if path_key not in cfg["paths"]:
            continue
        if task_type in cfg.get("export_formats", {}).get(format_name, []):
            paths.append(output_dir / cfg["paths"][path_key].format(task_type=task_type))
    return paths


def _deduped_path_for_task(output_dir: Path, cfg: dict[str, Any], task_type: str) -> Path:
    return output_dir / cfg["paths"]["sample_deduped"].format(task_type=task_type)


def _load_deduped_samples(output_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for task_type in cfg["task_types"]:
        path = _deduped_path_for_task(output_dir, cfg, task_type)
        samples.extend(item for item in iter_jsonl(path) if isinstance(item, dict))
    return samples


def _deduped_paths_exist(output_dir: Path, cfg: dict[str, Any]) -> bool:
    paths = [_deduped_path_for_task(output_dir, cfg, task_type) for task_type in cfg["task_types"]]
    return all(path.exists() for path in paths) and any(path.stat().st_size > 0 for path in paths)


def _read_sample_cache_state(output_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / cfg["paths"].get("sample_cache_state", "samples/cache_state.json")
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception as exc:
        _state_logger(cfg).warning("sample cache state unreadable, treating as absent path=%s error=%s", path, exc)
        return {}
    if not isinstance(payload, dict):
        _state_logger(cfg).warning("sample cache state is not an object, treating as absent path=%s", path)
        return {}
    return payload


def _cached_completed_task_types(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> set[str]:
    if not mineru_content_hash:
        return set()
    payload = _read_sample_cache_state(output_dir, cfg)
    if payload.get("mineru_content_hash") != mineru_content_hash:
        return set()
    if payload.get("pipeline_version") != cfg.get("pipeline_version", ""):
        return set()
    completed = payload.get("completed_task_types")
    if not isinstance(completed, list):
        completed = payload.get("task_types")
    if not isinstance(completed, list):
        return set()
    return {str(task_type) for task_type in completed if str(task_type)}


def _sample_cache_state_matches(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> bool:
    if not mineru_content_hash:
        return False
    payload = _read_sample_cache_state(output_dir, cfg)
    return (
        payload.get("mineru_content_hash") == mineru_content_hash
        and payload.get("pipeline_version") == cfg.get("pipeline_version", "")
    )


def _task_outputs_exist(output_dir: Path, cfg: dict[str, Any], task_type: str) -> bool:
    export_paths = _expected_export_paths_for_task(output_dir, cfg, task_type)
    return (
        bool(export_paths)
        and _deduped_path_for_task(output_dir, cfg, task_type).exists()
        and all(path.exists() for path in export_paths)
    )


def _task_outputs_have_content(output_dir: Path, cfg: dict[str, Any], task_type: str) -> bool:
    if not _task_outputs_exist(output_dir, cfg, task_type):
        return False
    export_paths = _expected_export_paths_for_task(output_dir, cfg, task_type)
    paths = [_deduped_path_for_task(output_dir, cfg, task_type), *export_paths]
    return any(path.stat().st_size > 0 for path in paths)


def _completed_task_types(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> set[str]:
    cached = _cached_completed_task_types(output_dir, cfg, mineru_content_hash)
    selected = {str(task_type) for task_type in cfg["task_types"]}
    completed = {
        task_type
        for task_type in selected
        if task_type in cached and _task_outputs_exist(output_dir, cfg, task_type)
    }
    if _sample_cache_state_matches(output_dir, cfg, mineru_content_hash):
        completed.update(task_type for task_type in selected if _task_outputs_have_content(output_dir, cfg, task_type))
    return completed


def _sample_cache_matches(output_dir: Path, cfg: dict[str, Any], mineru_content_hash: str) -> bool:
    completed = _completed_task_types(output_dir, cfg, mineru_content_hash)
    return set(cfg["task_types"]).issubset(completed)


def _ordered_task_types(cfg: dict[str, Any], task_types: set[str]) -> list[str]:
    ordered = [task_type for task_type in cfg.get("task_types", []) if task_type in task_types]
    extras = sorted(task_type for task_type in task_types if task_type not in ordered)
    return ordered + extras


def _write_sample_cache_state(
    output_dir: Path,
    cfg: dict[str, Any],
    mineru_content_hash: str,
    completed_task_types: set[str] | None = None,
) -> None:
    previous = _cached_completed_task_types(output_dir, cfg, mineru_content_hash)
    current = set(cfg["task_types"]) if completed_task_types is None else set(completed_task_types)
    completed = current | previous
    ordered_completed = _ordered_task_types(cfg, completed)
    write_json(
        output_dir / cfg["paths"].get("sample_cache_state", "samples/cache_state.json"),
        {
            "mineru_content_hash": mineru_content_hash,
            "task_types": ordered_completed,
            "completed_task_types": ordered_completed,
            "pipeline_version": cfg.get("pipeline_version", ""),
        },
    )


def _exports_exist(output_dir: Path, cfg: dict[str, Any]) -> bool:
    expected = _expected_export_paths(output_dir, cfg)
    return bool(expected) and all(path.exists() for path in expected) and any(path.stat().st_size > 0 for path in expected)


def _export_reuse_ready(output_dir: Path, cfg: dict[str, Any]) -> bool:
    if not _exports_exist(output_dir, cfg):
        return False
    payload = _read_sample_cache_state(output_dir, cfg)
    if not payload:
        return False
    if payload.get("pipeline_version") != cfg.get("pipeline_version", ""):
        return False
    completed = payload.get("completed_task_types")
    if not isinstance(completed, list):
        completed = payload.get("task_types")
    if not isinstance(completed, list):
        return False
    completed_set = {str(task_type) for task_type in completed if str(task_type)}
    return all(
        task_type in completed_set or _task_outputs_have_content(output_dir, cfg, task_type)
        for task_type in cfg["task_types"]
    )


def _cfg_for_task_types(cfg: dict[str, Any], task_types: list[str]) -> dict[str, Any]:
    return _select_task_types(deepcopy(cfg), task_types)


def _scale_provider_quota(cfg: dict[str, Any], divisor: int) -> None:
    """按 journal 进程数摊薄 provider 配额。

    journal 级用的是 ProcessPoolExecutor，VlmPool 里的信号量和限流状态都是进程内的，
    每个子进程各建一份。不摊薄的话 provider 实际承受的并发是配置值的 journal_workers 倍。
    """
    if divisor <= 1:
        return
    providers = cfg.get("vlm_pool", {}).get("providers")
    if not isinstance(providers, list):
        return
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        declared = max(1, int(provider.get("max_concurrency") or 1))
        provider["max_concurrency"] = max(1, declared // divisor)
        interval = float(provider.get("min_interval_seconds") or 0.0)
        if interval > 0:
            provider["min_interval_seconds"] = interval * divisor


def _build_vlm_pool(cfg: dict[str, Any]) -> VlmPool:
    journal_workers = max(1, int(cfg["runtime"].get("journal_workers", 1)))
    cfg = deep_merge(
        cfg,
        {
            "vlm_pool": {
                "provider_defaults": {
                    "min_interval_seconds": float(cfg["runtime"].get("vlm_min_interval_seconds", 0.0)) * journal_workers,
                    "max_image_bytes": int(cfg["generation"].get("max_image_bytes", 0)),
                    "max_image_side": int(cfg["generation"].get("max_image_side", 1600)),
                    "image_jpeg_quality": int(cfg["generation"].get("image_jpeg_quality", 85)),
                }
            }
        },
    )
    _scale_provider_quota(cfg, journal_workers)
    return VlmPool.from_config(cfg, cfg["prompts"])


def _process_journal(journal: Any, cfg: dict[str, Any], vlm: VlmPool | None = None) -> dict[str, Any]:
    output_dir = Path(journal.output_dir)
    logger = configure_logger(f"{cfg['logger_name']}.{journal.journal_id}", cfg["runtime"].get("log_level"))
    state = RunState(logger)
    if bool(cfg["runtime"]["skip_vlm"]):
        vlm = None
    elif vlm is None:
        # 多进程路径：VlmPool 含线程锁不可 pickle，只能在子进程里现建。
        vlm = _build_vlm_pool(cfg)

    logger.info("start journal_id=%s pdf=%s", journal.journal_id, journal.source_pdf)
    append_jsonl(output_dir / cfg["paths"]["manifest"], asdict(journal))
    try:
        watermark_result = clean_pdf_watermarks(journal, cfg)
        working_journal = replace(journal, source_pdf=watermark_result.cleaned_pdf) if watermark_result.cleaned else journal
        working_cfg = cfg
        if watermark_result.cleaned:
            working_cfg = deepcopy(cfg)
            working_cfg["runtime"]["reuse_mineru"] = False
            working_cfg["runtime"]["reuse_normalized"] = False
            working_cfg["runtime"]["reuse_samples"] = False
            working_cfg["runtime"]["reuse_exports"] = False
            working_cfg["runtime"]["rerender_pages"] = True
        state.checkpoint("watermark_cleaning", asdict(watermark_result))

        if (
            bool(working_cfg["runtime"].get("reuse_exports"))
            and _export_reuse_ready(output_dir, working_cfg)
            and mineru_cache_is_valid(working_journal, working_cfg)
        ):
            logger.info("reuse exports journal_id=%s", journal.journal_id)
            return {"journal_id": journal.journal_id, "output_dir": str(output_dir), "sample_count": 0, "reused": "exports"}

        render_pages(working_journal, working_cfg)
        state.checkpoint("render_pages", working_cfg["statuses"]["done"])
        mineru_result = parse_journal_with_mineru(working_journal, working_cfg)
        content = mineru_result.content
        parsed_path = mineru_result.parsed_path
        image_map = mineru_result.image_map
        state.checkpoint("mineru", mineru_result.status)
        normalized = normalize_pages(working_journal, content, parsed_path, image_map, working_cfg)
        pages = normalized.pages
        articles = normalized.articles
        cfg = working_cfg
        crop_blocks(pages, output_dir, cfg)
        state.checkpoint("normalize_and_crop", len(pages))
        export_all_tasks = True
        if (
            bool(cfg["runtime"].get("reuse_samples"))
            and _deduped_paths_exist(output_dir, cfg)
            and _sample_cache_matches(output_dir, cfg, mineru_result.content_hash)
        ):
            deduped = _load_deduped_samples(output_dir, cfg)
            logger.info("reuse deduped samples journal_id=%s samples=%s", journal.journal_id, len(deduped))
        else:
            completed_tasks = (
                _completed_task_types(output_dir, cfg, mineru_result.content_hash)
                if bool(cfg["runtime"].get("reuse_samples"))
                else set()
            )
            pending_tasks = [task_type for task_type in cfg["task_types"] if task_type not in completed_tasks]
            existing_deduped = []
            if completed_tasks:
                completed_cfg = _cfg_for_task_types(cfg, _ordered_task_types(cfg, completed_tasks))
                existing_deduped = _load_deduped_samples(output_dir, completed_cfg)
                logger.info(
                    "reuse completed task samples journal_id=%s tasks=%s samples=%s",
                    journal.journal_id,
                    ",".join(completed_cfg["task_types"]),
                    len(existing_deduped),
                )
            if not pending_tasks:
                deduped = existing_deduped
                _write_sample_cache_state(output_dir, cfg, mineru_result.content_hash, completed_tasks)
                export_all_tasks = not bool(cfg["runtime"].get("reuse_exports"))
                logger.info("all selected tasks already completed journal_id=%s", journal.journal_id)
            else:
                pending_task_set = set(pending_tasks)
                jobs = [job for job in build_sample_jobs(journal, pages, articles, cfg) if job.task_type in pending_task_set]
                job_counts = {task_type: 0 for task_type in pending_tasks}
                for job in jobs:
                    job_counts[job.task_type] = job_counts.get(job.task_type, 0) + 1
                logger.info(
                    "generate pending tasks journal_id=%s tasks=%s jobs=%s",
                    journal.journal_id,
                    ",".join(pending_tasks),
                    len(jobs),
                )
                pending_cfg = _cfg_for_task_types(cfg, pending_tasks)
                raw_samples = _generate_samples_for_jobs(
                    jobs=jobs,
                    output_dir=output_dir,
                    cfg=pending_cfg,
                    vlm=vlm,
                    state=state,
                    mineru_content_hash=mineru_result.content_hash,
                )
                validated = [
                    valid
                    for sample in raw_samples
                    if (valid := validate_sample(sample, output_dir, pending_cfg)) is not None
                ]
                for task_type in pending_cfg["task_types"]:
                    write_jsonl(
                        output_dir / pending_cfg["paths"]["sample_validated"].format(task_type=task_type),
                        (sample for sample in validated if sample.get("task_type") == task_type),
                    )
                new_deduped = deduplicate(validated, pending_cfg)
                for task_type in pending_cfg["task_types"]:
                    write_jsonl(
                        output_dir / pending_cfg["paths"]["sample_deduped"].format(task_type=task_type),
                        (sample for sample in new_deduped if sample.get("task_type") == task_type),
                    )
                sample_counts = {task_type: 0 for task_type in pending_tasks}
                for sample in new_deduped:
                    task_type = str(sample.get("task_type") or "")
                    sample_counts[task_type] = sample_counts.get(task_type, 0) + 1
                completed_pending = {
                    task_type
                    for task_type in pending_tasks
                    if job_counts.get(task_type, 0) == 0 or sample_counts.get(task_type, 0) > 0
                }
                if completed_pending:
                    completed_pending_cfg = _cfg_for_task_types(cfg, _ordered_task_types(cfg, completed_pending))
                    export_task_files(
                        output_dir,
                        [sample for sample in new_deduped if sample.get("task_type") in completed_pending],
                        completed_pending_cfg,
                    )
                incomplete_pending = [task_type for task_type in pending_tasks if task_type not in completed_pending]
                if incomplete_pending:
                    logger.info("pending tasks produced no valid samples journal_id=%s tasks=%s", journal.journal_id, ",".join(incomplete_pending))
                completed_tasks.update(completed_pending)
                _write_sample_cache_state(output_dir, cfg, mineru_result.content_hash, completed_tasks)
                deduped = existing_deduped + new_deduped
                export_all_tasks = not bool(cfg["runtime"].get("reuse_exports"))
        if export_all_tasks:
            export_task_files(output_dir, deduped, cfg)
        state.increment("journals_completed")
        state.increment("samples_exported", len(deduped))
        state.checkpoint("export", cfg["statuses"]["done"])
        result = {"journal_id": journal.journal_id, "output_dir": str(output_dir), "sample_count": len(deduped)}
        logger.info("completed journal_id=%s samples=%s", journal.journal_id, len(deduped))
        return result
    except UnreadablePdfError as exc:
        # 文件本身坏了，重试也没用，不算失败，直接跳过下一篇。
        logger.warning("skip unreadable pdf journal_id=%s pdf=%s reason=%s", journal.journal_id, journal.source_pdf, exc)
        return {
            "journal_id": journal.journal_id,
            "output_dir": str(output_dir),
            "sample_count": 0,
            "skipped": "unreadable_pdf",
            "skip_reason": str(exc),
            "source_pdf": str(journal.source_pdf),
        }
    except Exception as exc:
        state.error({"stage": "journal", "journal_id": journal.journal_id, "error": str(exc)})
        logger.exception("failed journal_id=%s", journal.journal_id)
        return {"journal_id": journal.journal_id, "output_dir": str(output_dir), "sample_count": 0, "error": str(exc)}


def run_pipeline(options: PipelineOptions | None = None) -> list[dict[str, Any]]:
    options = options or PipelineOptions()
    cfg = _apply_options(load_config(options.config_path), options)
    cfg = _resolve_config_runtime_paths(cfg, options)
    input_dir = Path(cfg["runtime"]["input_dir"])
    output_root = Path(cfg["runtime"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    input_journals = cfg["runtime"].get("input_journals")
    if isinstance(input_journals, list) and input_journals:
        journals = scan_input_journals(input_journals, output_root, cfg)
    else:
        journals = scan_journals(input_dir, output_root, cfg)
    if options.limit_journals:
        journals = journals[: options.limit_journals]

    journal_workers = max(1, int(cfg["runtime"]["journal_workers"]))
    show_progress = bool(cfg["runtime"].get("progress", True))
    total = len(journals)
    recursive = bool(cfg["runtime"].get("recursive", True))
    print(
        f"扫描到 {total} 个 PDF（{'递归' if recursive else '仅一级'}：{input_dir}），"
        f"并发 {journal_workers}，输出 {output_root}",
        flush=True,
    )
    if not journals:
        return []

    def _summarize(result: dict[str, Any]) -> tuple[bool, str]:
        ok = not result.get("error")
        label = str(result.get("journal_id", ""))
        if result.get("skipped"):
            label = f"{label} 跳过"
        elif not ok:
            label = f"{label} 失败"
        return ok, label

    skip_log = output_root / str(cfg["paths"].get("skipped_journals", "skipped_journals.jsonl"))

    def _record_skip(result: dict[str, Any]) -> None:
        if not result.get("skipped"):
            return
        append_jsonl(
            skip_log,
            {
                "journal_id": result.get("journal_id"),
                "source_pdf": result.get("source_pdf"),
                "skipped": result.get("skipped"),
                "reason": result.get("skip_reason"),
            },
        )

    if journal_workers == 1 or total <= 1:
        results = []
        # 单进程时整批共用一个 pool，provider 的冷却和轮询状态才能跨 journal 延续。
        shared_vlm = None if bool(cfg["runtime"]["skip_vlm"]) else _build_vlm_pool(cfg)
        with ProgressBar(total, desc="期刊", enabled=show_progress) as bar:
            for journal in journals:
                bar.set_current(journal.journal_id)
                result = _process_journal(journal, cfg, shared_vlm)
                _record_skip(result)
                ok, label = _summarize(result)
                bar.advance(ok=ok, current=label)
                results.append(result)
        return results

    with ProgressBar(total, desc="期刊", enabled=show_progress) as bar:
        with ProcessPoolExecutor(max_workers=journal_workers) as executor:
            futures = {executor.submit(_process_journal, journal, cfg): index for index, journal in enumerate(journals)}
            ordered_results: list[dict[str, Any] | None] = [None] * total
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # 单个期刊崩溃不该带崩整批
                    result = {
                        "journal_id": journals[index].journal_id,
                        "output_dir": journals[index].output_dir,
                        "sample_count": 0,
                        "error": str(exc),
                    }
                ordered_results[index] = result
                _record_skip(result)
                ok, label = _summarize(result)
                bar.advance(ok=ok, current=label)
    return [result for result in ordered_results if result is not None]



