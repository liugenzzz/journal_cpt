from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from .clients import MinerUClient
from .cooldown import SharedCooldown, runtime_dir
from ..core.io_utils import clean_text, maybe_json, read_json, stable_json_hash, write_json
from ..core.models import JournalRecord


@dataclass(frozen=True)
class MinerUParseResult:
    payload: Any
    content: list[dict[str, Any]]
    parsed_path: str
    image_map: dict[str, str]
    status: dict[str, Any]
    content_hash: str
    refreshed: bool


class MinerUParseIncompleteError(RuntimeError):
    pass


def strip_embedded_images(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: strip_embedded_images(value) for key, value in payload.items() if key != "images"}
    if isinstance(payload, list):
        return [strip_embedded_images(item) for item in payload]
    return payload


def iter_image_maps(payload: Any):
    payload = maybe_json(payload)
    if isinstance(payload, dict):
        for key in ("images", "image_dict", "img_dict"):
            image_map = maybe_json(payload.get(key))
            if isinstance(image_map, dict):
                yield image_map
        for value in payload.values():
            yield from iter_image_maps(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from iter_image_maps(value)


def decode_image(value: Any) -> tuple[bytes, str] | None:
    value = maybe_json(value)
    if isinstance(value, dict):
        for key in ("data", "base64", "content", "image"):
            decoded = decode_image(value.get(key))
            if decoded:
                return decoded
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    mime = "image/png"
    if text.startswith("data:image/"):
        header, _, text = text.partition(",")
        match = re.search(r"data:([^;]+)", header)
        if match:
            mime = match.group(1)
    try:
        return base64.b64decode(text, validate=False), mime
    except Exception:
        return None


def _item_page(item: dict[str, Any], fallback: int) -> int:
    for key in ("page_idx", "page_index", "page", "page_no"):
        value = item.get(key)
        if isinstance(value, int):
            return value + 1 if key == "page_idx" else value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            return number + 1 if key == "page_idx" else number
    return fallback


def _item_image_refs(item: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key, raw_value in item.items():
        value = maybe_json(raw_value)
        if key in {"img_path", "image_path", "table_img_path", "figure_path", "path", "file_path"}:
            if isinstance(value, str) and value.strip():
                refs.append(value.strip())
            elif isinstance(value, list):
                refs.extend(str(part).strip() for part in value if str(part).strip())
        elif isinstance(value, dict):
            refs.extend(_item_image_refs(value))
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, dict):
                    refs.extend(_item_image_refs(part))
    return refs


def _normalize_image_ref(ref: str) -> str:
    return ref.replace("\\", "/")


def collect_image_ref_pages(content: list[dict[str, Any]]) -> dict[str, int]:
    ref_pages: dict[str, int] = {}
    fallback_page = 1
    for item in content:
        page_no = _item_page(item, fallback_page)
        fallback_page = page_no
        for ref in _item_image_refs(item):
            normalized = _normalize_image_ref(ref)
            ref_pages.setdefault(ref, page_no)
            ref_pages.setdefault(normalized, page_no)
            ref_pages.setdefault(Path(normalized).name, page_no)
    return ref_pages


def _image_suffix(name: str, mime: str, cfg: dict[str, Any]) -> str:
    suffix = Path(name).suffix.lower()
    allowed_suffixes = set(cfg["extracted_image_naming"]["allowed_suffixes"])
    if suffix not in allowed_suffixes:
        suffix = mimetypes.guess_extension(mime) or ".png"
    return suffix


def safe_image_name(name: str, image_no: int, page_no: int | None, mime: str, cfg: dict[str, Any]) -> str:
    suffix = _image_suffix(name, mime, cfg)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name.replace("\\", "/")).stem)
    source_stem = stem.strip("._-") or "image"
    naming = cfg["extracted_image_naming"]
    page_label = str(page_no).zfill(3) if page_no else str(naming["unknown_page_label"])
    return naming["template"].format(
        page_label=page_label,
        page_no=page_no or 0,
        image_no=image_no,
        source_stem=source_stem,
        suffix=suffix,
    )


def save_extracted_images(
    payload: Any,
    content: list[dict[str, Any]],
    output_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, str]:
    images_dir = output_dir / cfg["paths"]["extracted_images"]
    images_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, str] = {}
    used_names: set[str] = set()
    ref_pages = collect_image_ref_pages(content)
    page_counts: dict[int | None, int] = {}
    for mineru_images in iter_image_maps(payload):
        for original_name, encoded in mineru_images.items():
            decoded = decode_image(encoded)
            if not decoded:
                continue
            image_bytes, mime = decoded
            normalized_name = _normalize_image_ref(str(original_name))
            page_no = (
                ref_pages.get(str(original_name))
                or ref_pages.get(normalized_name)
                or ref_pages.get(Path(normalized_name).name)
            )
            page_counts[page_no] = page_counts.get(page_no, 0) + 1
            image_no = page_counts[page_no]
            filename = safe_image_name(str(original_name), image_no, page_no, mime, cfg)
            while filename in used_names:
                image_no += 1
                page_counts[page_no] = image_no
                filename = safe_image_name(str(original_name), image_no, page_no, mime, cfg)
            used_names.add(filename)
            target = images_dir / filename
            if not target.exists():
                target.write_bytes(image_bytes)
            image_map[str(original_name)] = str(target.relative_to(output_dir)).replace("\\", "/")
    return image_map


def find_content_list(payload: Any) -> list[dict[str, Any]]:
    candidates: list[list[dict[str, Any]]] = []

    def visit(value: Any) -> None:
        value = maybe_json(value)
        if isinstance(value, dict):
            content = maybe_json(value.get("content_list"))
            if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                candidates.append(content)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value) and any("type" in item for item in value):
                candidates.append(value)
            for child in value:
                visit(child)

    visit(payload)
    return max(candidates, key=len) if candidates else []


def _mineru_paths(journal: JournalRecord, cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(journal.output_dir)
    raw_path = output_dir / cfg["paths"]["mineru_raw"].format(journal_id=journal.journal_id)
    parsed_path = output_dir / cfg["paths"]["mineru_parsed"].format(journal_id=journal.journal_id)
    image_map_path = output_dir / cfg["paths"]["image_map"]
    status_path = output_dir / cfg["paths"].get("mineru_status", "mineru/status/{journal_id}.json").format(journal_id=journal.journal_id)
    return raw_path, parsed_path, image_map_path, status_path


def _content_page_numbers(content: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    fallback_page = 1
    for item in content:
        page_no = _item_page(item, fallback_page)
        fallback_page = page_no
        if page_no > 0:
            pages.add(page_no)
    return pages


def _content_text_chars(content: list[dict[str, Any]]) -> int:
    keys = (
        "text",
        "content",
        "md_content",
        "markdown",
        "caption",
        "img_caption",
        "image_caption",
        "table_caption",
        "image_footnote",
        "latex",
        "html",
    )
    return sum(len(clean_text(" ".join(clean_text(item.get(key)) for key in keys))) for item in content)


def _build_mineru_status(
    journal: JournalRecord,
    content: list[dict[str, Any]],
    image_map: dict[str, str],
    cfg: dict[str, Any],
    source: str,
    error: str = "",
) -> dict[str, Any]:
    mineru_cfg = cfg.get("mineru", {})
    page_count = max(0, int(journal.page_count or 0))
    pages = _content_page_numbers(content)
    text_chars = _content_text_chars(content)
    min_items = max(0, int(mineru_cfg.get("min_content_items", 1)))
    min_page_coverage = max(0.0, float(mineru_cfg.get("min_page_coverage", 0.0)))
    min_text_chars = max(0, int(mineru_cfg.get("min_text_chars", 0)))

    errors: list[str] = []
    if len(content) < min_items:
        errors.append(f"content_items {len(content)} < {min_items}")
    if page_count and min_page_coverage:
        required_pages = max(1, ceil(page_count * min_page_coverage))
        if len(pages) < required_pages:
            errors.append(f"page_coverage {len(pages)}/{page_count} < {min_page_coverage:.2f}")
    if text_chars < min_text_chars:
        errors.append(f"text_chars {text_chars} < {min_text_chars}")
    if error:
        errors.append(error)

    return {
        "valid": not errors,
        "errors": errors,
        "journal_id": journal.journal_id,
        "source": source,
        "content_hash": stable_json_hash(content),
        "content_items": len(content),
        "text_chars": text_chars,
        "pages_with_content": len(pages),
        "page_count": page_count,
        "page_coverage": (len(pages) / page_count) if page_count else 1.0,
        "image_count": len(image_map),
    }


def _validate_mineru_result(
    journal: JournalRecord,
    content: list[dict[str, Any]],
    image_map: dict[str, str],
    cfg: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    status = _build_mineru_status(journal, content, image_map, cfg, source)
    if not status["valid"]:
        raise MinerUParseIncompleteError("; ".join(status["errors"]))
    return status


def mineru_cache_is_valid(journal: JournalRecord, cfg: dict[str, Any]) -> bool:
    _, parsed_path, image_map_path, status_path = _mineru_paths(journal, cfg)
    if not parsed_path.exists() or not image_map_path.exists():
        return False
    try:
        content = read_json(parsed_path)
        image_map = read_json(image_map_path)
    except Exception:
        return False
    if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
        return False
    if not isinstance(image_map, dict):
        return False
    status = _build_mineru_status(journal, content, image_map, cfg, "cache")
    try:
        write_json(status_path, status)
    except Exception:
        pass
    return bool(status["valid"])


@dataclass(frozen=True)
class MinerUProvider:
    name: str
    cfg: dict[str, Any]
    max_concurrency: int
    weight: int

    @property
    def url(self) -> str:
        return str(self.cfg.get("url") or "")


def mineru_providers(cfg: dict[str, Any]) -> list[MinerUProvider]:
    """把 mineru 配置摊成 provider 列表。

    没配 providers 时退回单实例，行为与改造前一致。
    provider 条目继承 mineru 顶层的通用配置（parse_method / timeout / 阈值等），
    只覆盖 url、server_url、max_concurrency 这类实例相关的字段。
    """
    mineru_cfg = dict(cfg.get("mineru", {}))
    raw_providers = mineru_cfg.pop("providers", None)
    if not isinstance(raw_providers, list) or not raw_providers:
        raw_providers = [{}]
    providers: list[MinerUProvider] = []
    for index, entry in enumerate(raw_providers, start=1):
        if not isinstance(entry, dict) or entry.get("enabled") is False:
            continue
        provider_cfg = dict(mineru_cfg)
        provider_cfg.update(entry)
        name = str(provider_cfg.get("name") or f"mineru_{index}")
        provider_cfg["name"] = name
        providers.append(
            MinerUProvider(
                name=name,
                cfg=provider_cfg,
                max_concurrency=max(1, int(provider_cfg.get("max_concurrency", 1))),
                weight=max(1, int(provider_cfg.get("weight", 1))),
            )
        )
    if not providers:
        raise RuntimeError("No enabled MinerU provider is configured.")
    missing = [provider.name for provider in providers if not provider.url]
    if missing:
        raise RuntimeError(f"MinerU provider(s) without url: {', '.join(missing)}")
    return providers


def _mineru_slot_dir(cfg: dict[str, Any], journal: JournalRecord) -> Path:
    directory = runtime_dir(cfg, "mineru_slots", fallback=Path(journal.output_dir).parent)
    assert directory is not None
    return directory


def _reap_stale_slot(candidate: Path, stale_seconds: float, now: float) -> None:
    if not stale_seconds:
        return
    try:
        if candidate.exists() and now - candidate.stat().st_mtime > stale_seconds:
            candidate.unlink()
    except OSError:
        pass


def _try_take_slot(slot_dir: Path, provider: MinerUProvider, journal: JournalRecord, stale_seconds: float, now: float) -> Path | None:
    provider_dir = slot_dir / provider.name
    provider_dir.mkdir(parents=True, exist_ok=True)
    for index in range(provider.max_concurrency):
        candidate = provider_dir / f"slot_{index}.lock"
        _reap_stale_slot(candidate, stale_seconds, now)
        try:
            fd = os.open(str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\njournal_id={journal.journal_id}\ncreated_at={now}\n")
        return candidate
    return None


def _free_slot_count(slot_dir: Path, provider: MinerUProvider) -> int:
    provider_dir = slot_dir / provider.name
    taken = 0
    for index in range(provider.max_concurrency):
        if (provider_dir / f"slot_{index}.lock").exists():
            taken += 1
    return provider.max_concurrency - taken


@contextmanager
def _mineru_slot(
    journal: JournalRecord,
    cfg: dict[str, Any],
    providers: list[MinerUProvider] | None = None,
    exclude: set[str] | None = None,
):
    """抢占式拿一个 MinerU 实例的槽位；谁先空谁被拿走。

    槽位是文件锁，天然跨进程（甚至跨主机，只要 output_root 是共享盘），
    所以这里的并发上限对整批任务都成立，不像 VLM 那边的信号量只在进程内有效。
    """
    mineru_cfg = cfg.get("mineru", {})
    poll_seconds = max(0.1, float(mineru_cfg.get("slot_poll_seconds", 2)))
    stale_seconds = max(0.0, float(mineru_cfg.get("slot_stale_seconds", 7200)))
    all_providers = providers if providers is not None else mineru_providers(cfg)
    excluded = exclude or set()
    usable = [provider for provider in all_providers if provider.name not in excluded] or all_providers
    shared_cooldown = SharedCooldown(
        runtime_dir(cfg, "mineru_cooldown", fallback=Path(journal.output_dir).parent),
        ttl_seconds=float(cfg.get("runtime", {}).get("cooldown_refresh_seconds", 1.0)),
    )
    slot_dir = _mineru_slot_dir(cfg, journal)
    slot_dir.mkdir(parents=True, exist_ok=True)

    slot_path: Path | None = None
    taken: MinerUProvider | None = None
    try:
        while slot_path is None:
            now = time.time()
            ranked = [provider for provider in usable if not shared_cooldown.cooling_down(provider.name)] or usable
            # 空槽最多的排前面；同样空闲时按 weight 倾斜。起点用 pid 打散，
            # 避免多个进程每轮都从同一个 provider 开始抢。
            offset = os.getpid()
            ranked = sorted(
                enumerate(ranked),
                key=lambda item: (
                    -_free_slot_count(slot_dir, item[1]) / item[1].max_concurrency,
                    -item[1].weight,
                    (item[0] + offset) % max(1, len(ranked)),
                ),
            )
            for _, provider in ranked:
                candidate = _try_take_slot(slot_dir, provider, journal, stale_seconds, now)
                if candidate is not None:
                    slot_path, taken = candidate, provider
                    break
            if slot_path is None:
                time.sleep(poll_seconds)
        yield taken
    finally:
        if slot_path is not None:
            try:
                slot_path.unlink()
            except FileNotFoundError:
                pass


def _sleep_before_retry(attempt: int, cfg: dict[str, Any]) -> None:
    mineru_cfg = cfg.get("mineru", {})
    base_seconds = max(0.0, float(mineru_cfg.get("retry_backoff_seconds", 0)))
    multiplier = max(1.0, float(mineru_cfg.get("retry_backoff_multiplier", 1)))
    delay = base_seconds * (multiplier ** max(0, attempt - 1))
    if delay > 0:
        time.sleep(delay)


def parse_journal_with_mineru(journal: JournalRecord, cfg: dict[str, Any]) -> MinerUParseResult:
    output_dir = Path(journal.output_dir)
    raw_path, parsed_path, image_map_path, status_path = _mineru_paths(journal, cfg)
    if bool(cfg["runtime"]["reuse_mineru"]) and raw_path.exists() and parsed_path.exists() and image_map_path.exists():
        try:
            payload = read_json(raw_path)
            content = read_json(parsed_path)
            image_map = read_json(image_map_path)
            if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
                raise MinerUParseIncompleteError("cached MinerU content is not a list of objects")
            if not isinstance(image_map, dict):
                raise MinerUParseIncompleteError("cached MinerU image_map is not an object")
            status = _validate_mineru_result(journal, content, image_map, cfg, "cache")
            write_json(status_path, status)
            return MinerUParseResult(
                payload=payload,
                content=content,
                parsed_path=str(parsed_path.relative_to(output_dir)),
                image_map=image_map,
                status=status,
                content_hash=str(status["content_hash"]),
                refreshed=False,
            )
        except Exception as exc:
            failed_status = _build_mineru_status(journal, [], {}, cfg, "cache", f"ignored invalid cache: {exc}")
            write_json(status_path, failed_status)

    attempts = max(1, int(cfg.get("mineru", {}).get("retry_count", 0)) + 1)
    providers = mineru_providers(cfg)
    cooldown_seconds = max(0.0, float(cfg.get("mineru", {}).get("cooldown_seconds", 300.0)))
    shared_cooldown = SharedCooldown(
        runtime_dir(cfg, "mineru_cooldown", fallback=Path(journal.output_dir).parent),
        ttl_seconds=float(cfg.get("runtime", {}).get("cooldown_refresh_seconds", 1.0)),
    )
    last_error: Exception | None = None
    tried: set[str] = set()
    for attempt in range(1, attempts + 1):
        provider: MinerUProvider | None = None
        try:
            with _mineru_slot(journal, cfg, providers, exclude=tried) as provider:
                tried.add(provider.name)
                payload = MinerUClient(provider.cfg).parse_pdf(Path(journal.source_pdf))
            shared_cooldown.clear(provider.name)
            content = find_content_list(payload)
            source = f"mineru_attempt_{attempt}_{provider.name}"
            status = _validate_mineru_result(journal, content, {}, cfg, source)
            image_map = save_extracted_images(payload, content, output_dir, cfg)
            status = _build_mineru_status(journal, content, image_map, cfg, source)
            write_json(raw_path, strip_embedded_images(payload))
            write_json(parsed_path, content)
            write_json(image_map_path, image_map)
            write_json(status_path, status)
            return MinerUParseResult(
                payload=payload,
                content=content,
                parsed_path=str(parsed_path.relative_to(output_dir)),
                image_map=image_map,
                status=status,
                content_hash=str(status["content_hash"]),
                refreshed=True,
            )
        except Exception as exc:
            last_error = exc
            provider_name = provider.name if provider is not None else "unknown"
            # 只有实例本身不可用才拉黑；解析结果不完整是这份 PDF 的问题，换实例也一样。
            if provider is not None and not isinstance(exc, MinerUParseIncompleteError):
                shared_cooldown.mark(provider_name, cooldown_seconds)
            if len(tried) >= len(providers):
                tried.clear()
            failed_status = _build_mineru_status(
                journal, [], {}, cfg, f"mineru_attempt_{attempt}_{provider_name}", str(exc)
            )
            try:
                write_json(status_path, failed_status)
            except Exception:
                pass
            if attempt >= attempts:
                break
            _sleep_before_retry(attempt, cfg)

    raise RuntimeError(f"MinerU parse failed for journal_id={journal.journal_id}: {last_error}") from last_error
