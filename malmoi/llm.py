"""Thin wrappers around the Google Gen AI SDK (Gemini on Vertex AI / Agent Platform).

Authentication uses Application Default Credentials:
  - locally: `gcloud auth application-default login` (from the course setup guide)
  - on Cloud Run: the service's service account, automatically
"""

from __future__ import annotations

import contextvars
import json
import logging
import random
import re
import threading
import time

from google import genai
from google.genai import types

from . import config

log = logging.getLogger("malmoi.llm")

_client: genai.Client | None = None
_lock = threading.Lock()


def client() -> genai.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = genai.Client(
                    vertexai=True,
                    project=config.project_id(),
                    location=config.GOOGLE_CLOUD_LOCATION,
                    http_options=types.HttpOptions(timeout=90_000),
                )
    return _client


class LLMError(Exception):
    pass


# --- Rate limits -----------------------------------------------------------------
# Gemini on Vertex AI returns 429 (RESOURCE_EXHAUSTED) when requests arrive faster
# than the project's quota allows, or when shared capacity is busy. Google's
# recommended fix is to retry with exponential backoff, which every call here
# does. On top of that:
#   - at most GEMINI_MAX_CONCURRENT calls run at once per server instance;
#   - background work (pre-building Explore topics, sample answers, the next
#     drill) is marked BACKGROUND and limited to 2 concurrent calls, so it can
#     never crowd out what the user just clicked;
#   - after any 429, background work pauses for a bit before trying again.

BACKGROUND: contextvars.ContextVar[bool] = contextvars.ContextVar("malmoi_background", default=False)

_all_slots = threading.BoundedSemaphore(config.GEMINI_MAX_CONCURRENT)
_bg_slots = threading.BoundedSemaphore(2)
_pause_lock = threading.Lock()
_background_paused_until = 0.0
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def is_rate_limit(exc: Exception) -> bool:
    return _status(exc) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def _retryable(exc: Exception) -> bool:
    if _status(exc) in RETRYABLE_CODES:
        return True
    msg = str(exc)
    return any(s in msg for s in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "overloaded")) or \
        exc.__class__.__name__ in ("ReadTimeout", "ConnectTimeout", "RemoteProtocolError", "ConnectError")


def with_retries(fn):
    """Run one Gemini request with concurrency limits and exponential backoff."""
    global _background_paused_until
    background = BACKGROUND.get()
    attempts = 7 if background else 5          # foreground waits at most ~15 s in total
    delay = 1.0
    for attempt in range(1, attempts + 1):
        if background:
            wait = _background_paused_until - time.time()
            if wait > 0:
                time.sleep(wait)
            _bg_slots.acquire()
        _all_slots.acquire()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not _retryable(exc):
                raise
            if is_rate_limit(exc):
                with _pause_lock:
                    _background_paused_until = max(_background_paused_until, time.time() + 20)
            log.warning("Gemini %s (attempt %d/%d, %s); retrying in ~%.0fs",
                        _status(exc) or exc.__class__.__name__, attempt, attempts,
                        "background" if background else "foreground", delay)
        finally:
            _all_slots.release()
            if background:
                _bg_slots.release()
        time.sleep(delay * random.uniform(0.7, 1.3))
        delay = min(delay * 2, 30 if background else 8)
    raise RuntimeError("unreachable")


# Thinking levels the model rejected (so we stop sending them).
_unsupported_levels: set[str] = set()


def _is_thinking_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("thinking" in msg or "thinking_level" in msg) and (
        getattr(exc, "code", None) == 400 or "invalid" in msg or "not supported" in msg)


def generate(contents, cfg: types.GenerateContentConfig, thinking: str | None = None):
    """generate_content with a thinking level, falling back gracefully if the
    model doesn't support that level (or thinking settings at all)."""
    levels = [lvl for lvl in dict.fromkeys([thinking or config.GEMINI_TOOL_THINKING, "low"])
              if lvl and lvl != "off" and lvl not in _unsupported_levels]
    for level in levels:
        call_cfg = cfg.model_copy(update={"thinking_config": types.ThinkingConfig(thinking_level=level)})
        try:
            return with_retries(lambda: client().models.generate_content(
                model=config.GEMINI_MODEL, contents=contents, config=call_cfg))
        except Exception as exc:  # noqa: BLE001
            if not _is_thinking_error(exc):
                raise
            _unsupported_levels.add(level)
            log.warning("Model rejected thinking_level=%s; trying without it", level)
    return with_retries(lambda: client().models.generate_content(
        model=config.GEMINI_MODEL, contents=contents, config=cfg))


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def generate_stream(contents, cfg: types.GenerateContentConfig, thinking: str | None = None):
    """Streaming version of generate(): yields response chunks as they arrive.
    Applies the same thinking-level fallback (checked on the first chunk)."""
    levels = [lvl for lvl in dict.fromkeys([thinking or config.GEMINI_TOOL_THINKING, "low"])
              if lvl and lvl != "off" and lvl not in _unsupported_levels]
    for level in levels + [None]:
        call_cfg = cfg if level is None else cfg.model_copy(
            update={"thinking_config": types.ThinkingConfig(thinking_level=level)})
        def open_stream(call_cfg=call_cfg):
            # Retries cover the request up to its first chunk; after text has
            # started streaming, a failure is handled by the caller instead.
            stream = client().models.generate_content_stream(
                model=config.GEMINI_MODEL, contents=contents, config=call_cfg)
            return stream, next(stream, None)

        try:
            stream, first = with_retries(open_stream)
        except Exception as exc:  # noqa: BLE001
            if level is not None and _is_thinking_error(exc):
                _unsupported_levels.add(level)
                log.warning("Model rejected thinking_level=%s; trying without it", level)
                continue
            raise
        if first is not None:
            yield first
        yield from stream
        return


def generate_json(prompt: str, schema: dict, system: str | None = None, thinking: str | None = None) -> dict:
    """Ask Gemini for JSON that matches `schema` (a JSON Schema dict)."""
    t0 = time.time()
    try:
        resp = generate(prompt, types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
        ), thinking)
    except Exception as exc:  # noqa: BLE001
        raise LLMError(describe_api_error(exc)) from exc
    log.info("timing: generate_json %.1fs", time.time() - t0)
    try:
        return json.loads(_strip_fences(resp.text or ""))
    except (ValueError, TypeError) as exc:
        raise LLMError("the model returned malformed JSON") from exc


def grounded_answer(prompt: str, system: str | None = None) -> tuple[str, list[dict]]:
    """Answer with Google Search grounding. Returns (text, sources).

    Kept in its own call (not mixed with function calling) so it works with
    any Gemini model that supports grounding.
    """
    t0 = time.time()
    try:
        resp = generate(prompt, types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ), "low")
        log.info("timing: grounded search %.1fs", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        raise LLMError(describe_api_error(exc)) from exc
    sources: list[dict] = []
    seen = set()
    try:
        meta = resp.candidates[0].grounding_metadata
        for chunk in (meta.grounding_chunks or []) if meta else []:
            web = getattr(chunk, "web", None)
            if web and web.uri and web.uri not in seen:
                seen.add(web.uri)
                sources.append({"title": web.title or web.domain or "Source", "url": web.uri})
    except (AttributeError, IndexError):
        pass
    return (resp.text or "").strip(), sources[:6]


def describe_api_error(exc: Exception) -> str:
    """Turn SDK errors into messages that say what to do next."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    if code == 404 or "NOT_FOUND" in msg:
        return (f"Gemini model '{config.GEMINI_MODEL}' isn't available in location "
                f"'{config.GOOGLE_CLOUD_LOCATION}'. Set GEMINI_MODEL to the model used in the course starter.")
    if code == 403 or "PERMISSION_DENIED" in msg:
        return ("Permission denied calling Gemini. Make sure the Agent Platform (Vertex AI) API is enabled and the "
                "service account has the 'Vertex AI User' role.")
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return ("Gemini is busy right now (rate limit), and retrying for several seconds didn't help. "
                "Try again in a moment.")
    if code == 401 or "UNAUTHENTICATED" in msg or "DefaultCredentialsError" in exc.__class__.__name__:
        return "Not authenticated with Google Cloud. Locally, run: gcloud auth application-default login"
    return f"Gemini request failed ({exc.__class__.__name__}: {msg[:200]})"
