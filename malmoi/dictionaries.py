"""Clients for the National Institute of Korean Language (국립국어원) dictionaries.

Lookup order (first hit wins):
  1. 한국어기초사전 krdict   — learner's dictionary: English translations, level, examples
  2. 표준국어대사전 stdict   — standard dictionary: advanced and specialized words
  3. 우리말샘 opendict       — open dictionary: newer words, some slang

The parsers are deliberately defensive: they search the response tree for the
fields they need instead of assuming one exact layout.
"""

from __future__ import annotations

import logging
import re
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any

import httpx

from . import config
from .korean import romanize_headword

log = logging.getLogger("malmoi.dict")

KRDICT_SEARCH = "https://krdict.korean.go.kr/api/search"
KRDICT_VIEW = "https://krdict.korean.go.kr/api/view"
STDICT_SEARCH = "https://stdict.korean.go.kr/api/search.do"
STDICT_VIEW = "https://stdict.korean.go.kr/api/view.do"
OPENDICT_SEARCH = "https://opendict.korean.go.kr/api/search"

SOURCE_LABELS = {
    "krdict": "Basic Korean Dictionary (한국어기초사전)",
    "stdict": "Standard Korean Dictionary (표준국어대사전)",
    "opendict": "Urimalsaem open dictionary (우리말샘)",
}
LEVELS_EN = {"초급": "beginner", "중급": "intermediate", "고급": "advanced"}


class DictError(Exception):
    """A dictionary request failed in a way worth reporting."""


class DictNotConfigured(DictError):
    """The API key for this dictionary is not set."""


_http = httpx.Client(
    timeout=httpx.Timeout(8.0, connect=5.0),
    verify=config.NIKL_SSL_VERIFY,
    headers={"User-Agent": "Malmoi/1.0 (educational project; Columbia University)"},
    follow_redirects=True,
)

# Small LRU cache so repeated lookups don't spend API quota.
_cache: OrderedDict[tuple, Any] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 2000


def _cached(key: tuple, fn):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    value = fn()
    with _cache_lock:
        _cache[key] = value
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return value


def _get(url: str, params: dict) -> httpx.Response:
    try:
        resp = _http.get(url, params=params)
    except httpx.TimeoutException as exc:
        raise DictError("request timed out") from exc
    except httpx.ConnectError as exc:
        hint = " (SSL problem — see NIKL_SSL_VERIFY in the README)" if "SSL" in str(exc) or "certificate" in str(exc) else ""
        raise DictError(f"could not connect{hint}") from exc
    except httpx.HTTPError as exc:
        raise DictError(f"network error: {exc.__class__.__name__}") from exc
    if resp.status_code != 200:
        raise DictError(f"HTTP {resp.status_code}")
    return resp


_TAGS = re.compile(r"<[^>]+>")


def _clean(text: Any) -> str:
    if text is None:
        return ""
    text = _TAGS.sub("", str(text))
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def _clean_headword(word: str) -> str:
    """stdict/opendict mark morpheme boundaries with '-' and spaces with '^'."""
    return _clean(word).replace("^", " ").replace("-", "").strip()


def find_all(obj: Any, key: str) -> list:
    """Recursively collect every value stored under `key` in nested dicts/lists."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.extend(v if isinstance(v, list) else [v])
            found.extend(find_all(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_all(item, key))
    return found


def _first(values: list) -> str:
    for v in values:
        if isinstance(v, (str, int, float)) and str(v).strip():
            return _clean(v)
    return ""


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _json(resp: httpx.Response) -> dict:
    if not resp.content.strip():
        return {}
    try:
        data = resp.json()
    except ValueError as exc:
        raise DictError("unexpected response format") from exc
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        code = err.get("error_code", "") if isinstance(err, dict) else ""
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        raise DictError(_key_hint(code, msg))
    return data if isinstance(data, dict) else {}


def _key_hint(code: str, msg: str) -> str:
    code = str(code)
    if code in ("020", "021"):
        return f"API key rejected ({msg}). Check that the key is copied exactly and active."
    if code in ("010", "022"):
        return f"daily request limit reached ({msg}). Try again tomorrow."
    return f"error {code}: {msg}".strip()


# --- krdict (한국어기초사전) --------------------------------------------------------


def _x(el: ET.Element | None, tag: str) -> str:
    if el is None:
        return ""
    child = el.find(tag)
    return _clean(child.text) if child is not None and child.text else ""


def _krdict_root(resp: httpx.Response) -> ET.Element:
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise DictError("unexpected response format") from exc
    if root.tag == "error" or root.find("error_code") is not None:
        raise DictError(_key_hint(_x(root, "error_code"), _x(root, "message")))
    return root


def krdict_search(q: str, num: int = 10) -> list[dict]:
    if not config.KRDICT_API_KEY:
        raise DictNotConfigured("KRDICT_API_KEY is not set")

    def run():
        resp = _get(KRDICT_SEARCH, {
            "key": config.KRDICT_API_KEY, "q": q, "part": "word", "sort": "dict",
            "num": max(10, num), "translated": "y", "trans_lang": "1",
            "advanced": "y", "method": "exact",
        })
        root = _krdict_root(resp)
        items = []
        for it in root.iter("item"):
            senses = []
            for s in it.iter("sense"):
                tr = s.find("translation")
                senses.append({
                    "definition_ko": _x(s, "definition"),
                    "english_word": _x(tr, "trans_word"),
                    "english_definition": _x(tr, "trans_dfn"),
                })
            items.append({
                "word": _x(it, "word"),
                "sup_no": _x(it, "sup_no"),
                "target_code": _x(it, "target_code"),
                "origin": _x(it, "origin"),
                "pronunciation": _x(it, "pronunciation"),
                "level": _x(it, "word_grade"),
                "pos": _x(it, "pos"),
                "link": _x(it, "link"),
                "senses": senses,
            })
        return items

    return _cached(("krdict_search", q), run)


def krdict_view(target_code: str) -> dict:
    if not config.KRDICT_API_KEY or not target_code:
        return {}

    def run():
        resp = _get(KRDICT_VIEW, {
            "key": config.KRDICT_API_KEY, "method": "target_code", "q": target_code,
            "translated": "y", "trans_lang": "1",
        })
        root = _krdict_root(resp)
        examples = []
        for ex in root.iter("example_info"):
            text = _x(ex, "example")
            kind = _x(ex, "type")
            if text and kind in ("문장", "대화", ""):
                examples.append(text)
        if not examples:  # some responses list examples under a different parent
            examples = [_clean(e.text) for e in root.iter("example") if e.text]
        pron = next((_clean(p.text) for p in root.iter("pronunciation") if p.text), "")
        origin = next((_clean(o.text) for o in root.iter("original_language") if o.text), "")
        return {"examples": examples, "pronunciation": pron, "origin": origin}

    return _cached(("krdict_view", target_code), run)


# --- stdict (표준국어대사전) ---------------------------------------------------------


def stdict_search(q: str) -> list[dict]:
    if not config.STDICT_API_KEY:
        raise DictNotConfigured("STDICT_API_KEY is not set")

    def run():
        resp = _get(STDICT_SEARCH, {
            "key": config.STDICT_API_KEY, "q": q, "req_type": "json",
            "advanced": "y", "method": "exact", "num": 10,
        })
        data = _json(resp)
        items = []
        for it in _as_list(data.get("channel", {}).get("item")):
            senses = []
            for s in _as_list(it.get("sense")):
                senses.append({
                    "definition_ko": _clean(s.get("definition")),
                    "pos": _clean(s.get("pos")),
                    "category": _clean(s.get("cat") or s.get("type")),
                })
            items.append({
                "word": _clean_headword(it.get("word", "")),
                "sup_no": _clean(it.get("sup_no")),
                "target_code": _clean(it.get("target_code")),
                "pos": senses[0]["pos"] if senses else "",
                "link": _clean(_first([s.get("link") for s in _as_list(it.get("sense"))])),
                "senses": senses,
            })
        return items

    return _cached(("stdict_search", q), run)


def stdict_view(target_code: str) -> dict:
    if not config.STDICT_API_KEY or not target_code:
        return {}

    def run():
        resp = _get(STDICT_VIEW, {
            "key": config.STDICT_API_KEY, "method": "target_code", "q": target_code, "req_type": "json",
        })
        data = _json(resp)
        return {
            "pronunciation": _first(find_all(data, "pronunciation")),
            "origin": _first(find_all(data, "original_language")),
            "examples": [_clean(e) for e in find_all(data, "example") if isinstance(e, str) and _clean(e)],
        }

    return _cached(("stdict_view", target_code), run)


# --- opendict (우리말샘) -------------------------------------------------------------


def opendict_search(q: str) -> list[dict]:
    if not config.OPENDICT_API_KEY:
        raise DictNotConfigured("OPENDICT_API_KEY is not set")

    def run():
        resp = _get(OPENDICT_SEARCH, {
            "key": config.OPENDICT_API_KEY, "q": q, "req_type": "json", "part": "word",
            "advanced": "y", "method": "exact", "sort": "dict", "num": 10,
        })
        data = _json(resp)
        items = []
        for it in _as_list(data.get("channel", {}).get("item")):
            senses = []
            for s in _as_list(it.get("sense")):
                senses.append({
                    "definition_ko": _clean(s.get("definition")),
                    "pos": _clean(s.get("pos")),
                    "category": _clean(s.get("cat") or s.get("type")),
                    "origin": _clean(s.get("origin")),
                    "pronunciation": _clean(s.get("pronunciation")),
                })
            items.append({
                "word": _clean_headword(it.get("word", "")),
                "sup_no": _clean(it.get("sup_no")),
                "pos": senses[0]["pos"] if senses else "",
                "link": _clean(_first([s.get("link") for s in _as_list(it.get("sense"))])),
                "senses": senses,
            })
        return items

    return _cached(("opendict_search", q), run)


# --- Unified lookup --------------------------------------------------------------------


def _build_entry(source: str, items: list[dict], extra: dict) -> dict:
    primary = items[0]
    senses = []
    for idx, it in enumerate(items[:3]):
        for s in it["senses"]:
            senses.append({
                "n": len(senses) + 1,
                "homograph": it.get("sup_no") or (str(idx + 1) if len(items) > 1 else ""),
                "pos": it.get("pos") or s.get("pos", ""),
                "definition_ko": s.get("definition_ko", ""),
                "english_word": s.get("english_word", ""),
                "english_definition": s.get("english_definition", ""),
                "category": s.get("category", ""),
            })
            if len(senses) >= 8:
                break
        if len(senses) >= 8:
            break
    pronunciation = extra.get("pronunciation") or primary.get("pronunciation", "") or next(
        (s.get("pronunciation") for it in items for s in it["senses"] if s.get("pronunciation")), "")
    origin = extra.get("origin") or primary.get("origin", "") or next(
        (s.get("origin") for it in items for s in it["senses"] if s.get("origin")), "")
    level = primary.get("level", "")
    word = primary["word"] or ""
    return {
        "word": word,
        "found": True,
        "verified": True,
        "source": source,
        "source_label": SOURCE_LABELS[source],
        "pronunciation": pronunciation,
        "romanization": romanize_headword(word, pronunciation),
        "level": level,
        "level_en": LEVELS_EN.get(level, "not in learner's dictionary" if source != "krdict" else "unrated"),
        "origin": origin,
        "pos": primary.get("pos", ""),
        "senses": senses,
        "examples": extra.get("examples", [])[:4],
        "link": primary.get("link", ""),
        "homograph_count": len(items),
    }


def _lookup_krdict(word: str) -> dict | None:
    items = [i for i in krdict_search(word) if i["word"].replace(" ", "") == word.replace(" ", "")] or []
    if not items:
        return None
    try:
        extra = krdict_view(items[0]["target_code"])
    except DictError as exc:
        log.info("krdict view failed: %s", exc)
        extra = {}
    return _build_entry("krdict", items, extra)


def _lookup_stdict(word: str) -> dict | None:
    items = [i for i in stdict_search(word) if i["word"].replace(" ", "") == word.replace(" ", "")]
    if not items:
        return None
    try:
        extra = stdict_view(items[0]["target_code"])
    except DictError as exc:
        log.info("stdict view failed: %s", exc)
        extra = {}
    return _build_entry("stdict", items, extra)


def _lookup_opendict(word: str) -> dict | None:
    items = [i for i in opendict_search(word) if i["word"].replace(" ", "") == word.replace(" ", "")]
    if not items:
        return None
    return _build_entry("opendict", items, {})


CHAIN = (("krdict", _lookup_krdict), ("stdict", _lookup_stdict), ("opendict", _lookup_opendict))


def lookup(word: str, sources: tuple[str, ...] = ("krdict", "stdict", "opendict")) -> tuple[dict | None, list[str], bool]:
    """Return (entry or None, error messages, whether any dictionary is configured)."""
    errors: list[str] = []
    configured = False
    for name, fn in CHAIN:
        if name not in sources:
            continue
        try:
            entry = fn(word)
            configured = True
        except DictNotConfigured:
            continue
        except DictError as exc:
            configured = True
            errors.append(f"{SOURCE_LABELS[name]}: {exc}")
            continue
        if entry:
            return entry, errors, configured
    return None, errors, configured


def quick_check(word: str) -> dict | None:
    """Cheap existence check (search endpoints only, no examples). Used to verify
    LLM-proposed candidates. Returns a compact summary or None."""
    for name, search in (("krdict", krdict_search), ("stdict", stdict_search), ("opendict", opendict_search)):
        try:
            items = [i for i in search(word) if i["word"].replace(" ", "") == word.replace(" ", "")]
        except DictError:
            continue
        if items:
            first = items[0]
            sense = first["senses"][0] if first["senses"] else {}
            return {
                "source": name,
                "source_label": SOURCE_LABELS[name],
                "english_word": sense.get("english_word", ""),
                "definition_ko": sense.get("definition_ko", ""),
                "level": first.get("level", ""),
                "level_en": LEVELS_EN.get(first.get("level", ""), "not in learner's dictionary" if name != "krdict" else "unrated"),
                "origin": first.get("origin", ""),
                "pronunciation": first.get("pronunciation", ""),
            }
    return None


def any_configured() -> bool:
    return bool(config.KRDICT_API_KEY or config.STDICT_API_KEY or config.OPENDICT_API_KEY)
