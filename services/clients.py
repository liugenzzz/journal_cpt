from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from ..core.io_utils import maybe_json


class RequestsClient:
    def __init__(self) -> None:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Please install requests to call MinerU or VLM services.") from exc
        self.requests = requests


class MinerUClient(RequestsClient):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

    def _endpoint(self, route: str) -> str:
        parsed = urlparse(str(self.cfg["url"]))
        path = parsed.path.rstrip("/")
        for suffix in ("/file_parse", "/url_parse"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
        return urlunparse(parsed._replace(path=f"{path}/{route}".replace("//", "/"), params="", query="", fragment=""))

    def parse_pdf(self, pdf_path: Path) -> Any:
        backend = str(self.cfg.get("backend", "pipeline"))
        data: list[tuple[str, str]] = [
            ("backend", backend),
            ("parse_method", str(self.cfg.get("parse_method", "auto"))),
            ("formula_enable", str(bool(self.cfg.get("formula_enable", True))).lower()),
            ("table_enable", str(bool(self.cfg.get("table_enable", True))).lower()),
            ("return_md", str(bool(self.cfg.get("return_md", True))).lower()),
            ("return_middle_json", str(bool(self.cfg.get("return_middle_json", True))).lower()),
            ("return_content_list", str(bool(self.cfg.get("return_content_list", True))).lower()),
            ("return_images", str(bool(self.cfg.get("return_images", True))).lower()),
            ("response_format_zip", str(bool(self.cfg.get("response_format_zip", False))).lower()),
            ("return_original_file", str(bool(self.cfg.get("return_original_file", False))).lower()),
        ]
        # http-client 后端必须把推理服务地址一并发过去，否则服务端会去读
        # MINERU_VL_SERVER 环境变量，读不到就抛 ValueError 并以 409 返回。
        server_url = str(self.cfg.get("server_url", "") or "").strip()
        if server_url and "http-client" in backend:
            data.append(("server_url", server_url))
        for lang in self.cfg.get("lang_list") or ["ch"]:
            data.append(("lang_list", str(lang)))
        with pdf_path.open("rb") as handle:
            files = [("files", (pdf_path.name, handle, "application/pdf"))]
            response = self.requests.post(
                self._endpoint("file_parse"),
                files=files,
                data=data,
                timeout=int(self.cfg.get("timeout", 3600)),
            )
        if response.status_code >= 400:
            detail = ""
            try:
                detail = str(response.json())[:500]
            except Exception:
                detail = response.text[:500]
            raise RuntimeError(f"MinerU {response.status_code} for {pdf_path.name}: {detail}")
        return response.json()


@dataclass(frozen=True)
class VlmResponse:
    text: str
    provider_name: str
    model: str


class VlmClient(RequestsClient):
    def __init__(self, cfg: dict[str, Any], prompts: dict[str, str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.prompts = prompts
        self.name = str(cfg.get("name") or cfg.get("model") or cfg.get("url") or "vlm")
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key_env = str(self.cfg.get("api_key_env") or "").strip()
        api_key = str(self.cfg.get("api_key") or os.getenv(api_key_env, "") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _image_to_data_url(self, path: Path) -> str:
        image_bytes = path.read_bytes()
        max_bytes = int(self.cfg.get("max_image_bytes") or 0)
        if max_bytes and len(image_bytes) > max_bytes:
            image_bytes = self._compress_image(path, image_bytes)
            mime = "image/jpeg"
        else:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _compress_image(self, path: Path, image_bytes: bytes) -> bytes:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return image_bytes
        max_side = int(self.cfg.get("max_image_side") or 1600)
        quality = int(self.cfg.get("image_jpeg_quality") or 85)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()

    def _apply_rate_limit(self) -> None:
        min_interval = float(self.cfg.get("min_interval_seconds") or 0.0)
        if min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            delay = self._last_request_at + min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()

    def _extract_message(self, payload: Any) -> str:
        content = payload["choices"][0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
        return str(content).strip()

    def _extract_stream(self, response: Any) -> str:
        parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            text = str(line).strip()
            if text.startswith("data:"):
                text = text[5:].strip()
            if not text or text == "[DONE]":
                continue
            payload = maybe_json(text)
            if isinstance(payload, dict):
                choice = payload.get("choices", [{}])[0]
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                parts.append(str(delta.get("content", "")) if isinstance(delta, dict) else "")
        return "".join(parts).strip()

    def chat(self, prompt: str, images: list[Path] | None = None) -> str:
        content: str | list[dict[str, Any]]
        if images:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}})
        else:
            content = prompt
        payload: dict[str, Any] = {
            "model": self.cfg["model"],
            "messages": [
                {"role": "system", "content": self.prompts["system"]},
                {"role": "user", "content": content},
            ],
            "temperature": self.cfg.get("temperature", 0.4),
            "stream": bool(self.cfg.get("stream", False)),
        }
        if self.cfg.get("max_tokens") is not None:
            payload["max_tokens"] = self.cfg["max_tokens"]
        template_kwargs = self.cfg.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict) and template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        extra_payload = self.cfg.get("extra_payload")
        if isinstance(extra_payload, dict):
            payload.update(extra_payload)
        try:
            self._apply_rate_limit()
            response = self.requests.post(
                self.cfg["url"],
                json=payload,
                headers=self._headers(),
                timeout=int(self.cfg["timeout"]),
                stream=bool(self.cfg.get("stream", False)),
            )
        except self.requests.exceptions.RequestException as exc:
            # 连不上 / 超时 / DNS 之类，这类才是真的 "unavailable"
            raise RuntimeError(
                f"VLM service unreachable: {self.cfg['url']} ({type(exc).__name__}: {exc})"
            ) from exc
        if response.status_code >= 400:
            # 服务端有响应但报错：把状态码和返回体带出来，否则根本没法定位
            detail = ""
            try:
                detail = str(response.json())
            except Exception:
                detail = response.text or ""
            hint = ""
            if response.status_code == 401:
                hint = " —— api_key 不对或没发送，检查 provider 的 api_key 字段"
            elif response.status_code == 404:
                hint = " —— url 或 model 名字不对，确认与 --served-model-name 一致"
            elif response.status_code == 400:
                hint = " —— 请求体被拒，常见是超出 max_model_len 或 max_tokens 过大"
            raise RuntimeError(
                f"VLM HTTP {response.status_code} from {self.cfg['url']}{hint}: {detail[:600]}"
            )
        if self.cfg.get("stream"):
            return self._extract_stream(response)
        return self._extract_message(response.json())


class _PooledVlmClient:
    def __init__(self, client: VlmClient, cfg: dict[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.name = client.name
        self.model = str(cfg.get("model") or "")
        self.weight = max(1, int(cfg.get("weight") or 1))
        self.max_concurrency = max(1, int(cfg.get("max_concurrency") or 1))
        self.task_types = {str(item) for item in cfg.get("task_types", []) if str(item)}
        self.capabilities = {str(item) for item in cfg.get("capabilities", []) if str(item)}
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._active = 0
        self._cooldown_until = 0.0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def is_cooling_down(self, now: float) -> bool:
        with self._lock:
            return self._cooldown_until > now

    def mark_failure(self, cooldown_seconds: float, now: float) -> None:
        if cooldown_seconds <= 0:
            return
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, now + cooldown_seconds)

    def mark_success(self) -> None:
        with self._lock:
            self._cooldown_until = 0.0

    def supports(self, task_type: str, has_images: bool) -> bool:
        if self.cfg.get("enabled") is False:
            return False
        if self.task_types and task_type not in self.task_types:
            return False
        if has_images and "image" not in self.capabilities:
            return False
        return True

    def chat(self, prompt: str, images: list[Path] | None = None) -> VlmResponse:
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
        try:
            return VlmResponse(
                text=self.client.chat(prompt, images),
                provider_name=self.name,
                model=self.model,
            )
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()


class VlmPool:
    def __init__(
        self,
        clients: list[_PooledVlmClient],
        fallback_enabled: bool = True,
        max_attempts: int = 2,
        cooldown_seconds: float = 300.0,
        now_fn: Any | None = None,
    ) -> None:
        if not clients:
            raise ValueError("VLM pool requires at least one provider.")
        self.clients = clients
        self.fallback_enabled = fallback_enabled
        self.max_attempts = max(1, max_attempts)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._now_fn = now_fn or time.monotonic
        self._rr_index = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], prompts: dict[str, str]) -> "VlmPool":
        pool_cfg = cfg.get("vlm_pool")
        fallback_cfg: dict[str, Any] = {}
        provider_defaults: dict[str, Any] = {}
        explicit_pool = False
        if isinstance(pool_cfg, dict):
            providers = pool_cfg.get("providers")
            defaults_raw = pool_cfg.get("provider_defaults")
            if isinstance(defaults_raw, dict):
                provider_defaults = defaults_raw
            fallback_raw = pool_cfg.get("fallback")
            if isinstance(fallback_raw, dict):
                fallback_cfg = fallback_raw
        else:
            providers = None

        if not isinstance(providers, list) or not providers:
            providers = [cfg.get("vlm", {})]
        else:
            explicit_pool = True

        legacy_shared_cfg = dict(cfg.get("vlm", {}))
        clients: list[_PooledVlmClient] = []
        for index, provider in enumerate(providers, start=1):
            if not isinstance(provider, dict) or provider.get("enabled") is False:
                continue
            provider_cfg = dict(provider_defaults)
            if not explicit_pool:
                provider_cfg.update(legacy_shared_cfg)
            provider_cfg.update(provider)
            provider_cfg.setdefault("name", f"vlm_{index}")
            provider_cfg.setdefault("capabilities", ["text", "image"])
            provider_cfg.setdefault("task_types", [])
            clients.append(_PooledVlmClient(VlmClient(provider_cfg, prompts), provider_cfg))

        return cls(
            clients,
            fallback_enabled=bool(fallback_cfg.get("enabled", True)),
            max_attempts=int(fallback_cfg.get("max_attempts", 2)),
            cooldown_seconds=float(fallback_cfg.get("cooldown_seconds", 300.0)),
        )

    def _candidates(self, task_type: str, has_images: bool) -> list[_PooledVlmClient]:
        candidates = [client for client in self.clients if client.supports(task_type, has_images)]
        if candidates:
            return candidates
        requirement = "image-capable " if has_images else ""
        raise RuntimeError(f"No {requirement}VLM provider is configured for task_type={task_type}.")

    def _ranked_candidates(self, task_type: str, has_images: bool) -> tuple[list[_PooledVlmClient], bool]:
        candidates = self._candidates(task_type, has_images)
        now = float(self._now_fn())
        available = [client for client in candidates if not client.is_cooling_down(now)]
        all_cooling_down = not available
        if available:
            candidates = available
        with self._lock:
            start = self._rr_index
            self._rr_index = (self._rr_index + 1) % max(1, len(candidates))
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                item[1].active / item[1].max_concurrency,
                ((item[0] - start) % len(candidates)) / item[1].weight,
            )
        )
        return [client for _, client in indexed], all_cooling_down

    def chat(self, task_type: str, prompt: str, images: list[Path] | None = None) -> VlmResponse:
        has_images = bool(images)
        candidates, all_cooling_down = self._ranked_candidates(task_type, has_images)
        if self.fallback_enabled:
            attempts = len(candidates) if all_cooling_down else min(len(candidates), self.max_attempts)
        else:
            attempts = 1
        last_error: Exception | None = None
        attempt_errors: list[str] = []
        for client in candidates[:attempts]:
            try:
                response = client.chat(prompt, images)
                client.mark_success()
                return response
            except Exception as exc:
                last_error = exc
                attempt_errors.append(f"{client.name}({client.model}): {exc}")
                client.mark_failure(self.cooldown_seconds, float(self._now_fn()))
                if not self.fallback_enabled:
                    break
        if last_error is not None:
            detail = " | ".join(attempt_errors) if attempt_errors else str(last_error)
            raise RuntimeError(f"All attempted VLM providers failed for task_type={task_type}: {detail}") from last_error
        raise RuntimeError(f"No VLM provider attempted for task_type={task_type}.")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    if start < 0:
        return text
    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", text)


def _escape_invalid_json_string_backslashes(text: str) -> str:
    chars: list[str] = []
    in_string = False
    index = 0
    valid_simple_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    while index < len(text):
        char = text[index]
        if char == '"':
            chars.append(char)
            in_string = not in_string
            index += 1
            continue
        if char != "\\" or not in_string:
            chars.append(char)
            index += 1
            continue

        next_char = text[index + 1] if index + 1 < len(text) else ""
        if next_char in valid_simple_escapes:
            chars.append(char)
            chars.append(next_char)
            index += 2
            continue
        if next_char == "u":
            unicode_digits = text[index + 2 : index + 6]
            if len(unicode_digits) == 4 and all(item in "0123456789abcdefABCDEF" for item in unicode_digits):
                chars.append(text[index : index + 6])
                index += 6
                continue

        chars.append("\\\\")
        index += 1
    return "".join(chars)


def _escape_unescaped_inner_quotes(text: str) -> str:
    chars: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            chars.append(char)
            escaped = True
            continue
        if char != '"':
            chars.append(char)
            continue
        if not in_string:
            chars.append(char)
            in_string = True
            continue

        next_index = index + 1
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        next_char = text[next_index] if next_index < len(text) else ""
        if next_char in {",", "}", "]", ":"}:
            chars.append(char)
            in_string = False
        else:
            chars.append('\\"')
    return "".join(chars)


def _loads_json_array(text: str) -> Any:
    extracted = _extract_json_array(text)
    without_trailing_commas = _remove_trailing_commas(extracted)
    backslash_repaired = _escape_invalid_json_string_backslashes(without_trailing_commas)
    candidates = [
        text,
        extracted,
        without_trailing_commas,
        backslash_repaired,
        _escape_unescaped_inner_quotes(without_trailing_commas),
        _escape_unescaped_inner_quotes(backslash_repaired),
    ]
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise ValueError(f"VLM response is not valid JSON array: {last_error}") from last_error
    raise ValueError("VLM response is empty.")


def parse_json_array(text: str) -> list[dict[str, Any]]:
    stripped = _strip_json_fence(text)
    payload = _loads_json_array(stripped)
    if not isinstance(payload, list):
        raise ValueError("VLM response must be a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


def parse_jsonl_objects(text: str) -> list[dict[str, Any]]:
    stripped = _strip_json_fence(text)
    if not stripped:
        return []
    if stripped.lstrip().startswith("["):
        return parse_json_array(stripped)

    rows: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        candidate = line.strip().rstrip(",")
        if not candidate or candidate in {"[", "]"} or candidate.startswith("```"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end <= start:
                continue
            payload = json.loads(candidate[start : end + 1])
        if isinstance(payload, dict):
            rows.append(payload)

    if rows:
        return rows

    payload = json.loads(stripped)
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("VLM response must be JSONL objects, a JSON object, or a JSON array.")
