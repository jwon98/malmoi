"""Malmoi's tools.

Every tool returns a JSON-safe dict:
  success → {"ok": true, ...data}
  failure → {"ok": false, "error": "<what happened>", "hint": "<what the model should do next>"}

Design rule: each tool does something the model can't do reliably on its own —
read a real dictionary, read live search data, run morphological analysis, or
keep state. Where a tool uses Gemini internally (scenario writing, candidate
generation), the output is verified against a dictionary or Kiwi before it's
returned ("generate, then verify").
"""

from __future__ import annotations

import inspect
import logging
import random
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from . import cache, config, dictionaries, korean, llm, naver
from .storage import MASTERED_AT, store

log = logging.getLogger("malmoi.tools")
# Separate thread pools so nested work can never deadlock: generation tasks
# (_gen_pool) submit dictionary checks (_pool), never the other way around.
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="dict")
_gen_pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="gen")
# Background warm-up gets its own workers so it never delays what the user just clicked.
_warm_gen_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="warmgen")


def _no_emit(_event: dict) -> None:
    pass


@dataclass
class ToolContext:
    user: str
    session: dict = field(default_factory=dict)
    session_id: str = ""
    emit: Callable[[dict], None] = _no_emit  # progress events for the streaming UI


def ok(**data) -> dict:
    return {"ok": True, **data}


def fail(error: str, hint: str = "") -> dict:
    out = {"ok": False, "error": error}
    if hint:
        out["hint"] = hint
    return out


_PUNCT = re.compile(r"^[\s\"'“”‘’「」『』()\[\].,!?~…]+|[\s\"'“”‘’「」『』()\[\].,!?~…]+$")


def _norm(word: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub("", word or "")).strip()


def _parallel(fn, items: list) -> list:
    return list(_pool.map(fn, items))


# =============================================================================
# 1. lookup_word — dictionary lookup (Korean → English), saved automatically
# =============================================================================

def _lemma_candidates(word: str) -> list[str]:
    lemmas = [lem for lem in korean.content_lemmas(word) if lem != word]
    verbs = [lem for lem in lemmas if lem.endswith("다")]
    seen, out = set(), []
    for lem in verbs + lemmas:
        if lem not in seen and len(lem) >= 2:
            seen.add(lem)
            out.append(lem)
    return out[:2]


AI_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "word": {"type": "string", "description": "dictionary form"},
        "pos": {"type": "string"},
        "english_word": {"type": "string"},
        "english_definition": {"type": "string"},
        "definition_ko": {"type": "string"},
        "examples": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["word", "english_word", "english_definition"],
}


def _ai_entry(ctx: ToolContext, word: str, context_sentence: str) -> dict:
    """Used only when no dictionary API key is configured at all."""
    data = llm.generate_json(
        f"Give a concise learner's-dictionary entry for the Korean word '{word}'"
        + (f", as used in: '{context_sentence}'" if context_sentence else "")
        + ". Use the dictionary form. Examples must be natural Korean sentences.",
        AI_ENTRY_SCHEMA,
    )
    w = _norm(data.get("word") or word)
    entry = {
        "word": w,
        "found": True,
        "verified": False,
        "source": "ai",
        "source_label": "Gemini (not dictionary-verified)",
        "pronunciation": "",
        "romanization": korean.romanize(w),
        "level": "",
        "level_en": "unrated",
        "origin": "",
        "pos": data.get("pos", ""),
        "senses": [{"n": 1, "homograph": "", "pos": data.get("pos", ""), "definition_ko": data.get("definition_ko", ""),
                    "english_word": data.get("english_word", ""), "english_definition": data.get("english_definition", ""),
                    "category": ""}],
        "examples": data.get("examples", [])[:3],
        "link": "",
        "homograph_count": 1,
    }
    saved = store.record_lookup(ctx.user, entry, context_sentence)
    return ok(**entry, saved_to_word_bank=True, mastery=saved.get("mastery", 0),
              warning="No dictionary API key is configured, so this entry comes from Gemini and is unverified. Say so.")


def lookup_word(ctx: ToolContext, word: str, context_sentence: str = "") -> dict:
    word = _norm(word)
    context_sentence = (context_sentence or "").strip()[:500]
    if not word:
        return fail("No word was given.", "Call again with one Korean word, e.g. word='눈치'.")
    if not korean.has_hangul(word):
        return fail(f"'{word}' isn't written in Hangul.", "For English → Korean, call find_korean_words instead.")
    if len(word) > 30:
        return fail("That's longer than a word or short expression.",
                    "Pass a single word (dictionary form if you know it). For a whole passage, call mine_vocabulary.")

    entry, errors, configured = dictionaries.lookup(word)
    tried = [word]
    if not entry and configured:
        for lemma in _lemma_candidates(word):  # conjugated form → dictionary form
            tried.append(lemma)
            entry, more, _ = dictionaries.lookup(lemma)
            errors += more
            if entry:
                break
    if not configured:
        return _ai_entry(ctx, word, context_sentence)
    if not entry:
        return ok(
            found=False, word=word, tried_forms=tried, dictionary_errors=errors,
            next_step=("Not found in the configured dictionaries. Call search_slang to look it up on the web — it works "
                       "for rare, formal, or specialized words as well as slang — and say the result is web-sourced. "
                       "If it's a phrase, look up its main word instead."),
        )

    saved = store.record_lookup(ctx.user, entry, context_sentence)
    result = ok(**entry, saved_to_word_bank=True, times_looked_up=saved.get("lookups", 1),
                mastery=saved.get("mastery", 0))
    if entry["word"] != word:
        result["looked_up_as"] = f"'{word}' is a conjugated form of '{entry['word']}'."
    if context_sentence:
        result["context_sentence"] = context_sentence
        result["task"] = "Pick the numbered sense that fits context_sentence and explain the word as used there."
    if errors:
        result["dictionary_warnings"] = errors
    return result


# =============================================================================
# 2. find_korean_words — English → Korean, candidates verified in dictionaries
# =============================================================================

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "Korean dictionary form (verbs/adjectives end in 다)"},
                    "register": {"type": "string", "enum": ["casual", "neutral", "formal", "written", "slang"]},
                    "nuance": {"type": "string", "description": "When a native speaker picks this one over the others (1 sentence)"},
                    "example_ko": {"type": "string"},
                    "example_en": {"type": "string"},
                },
                "required": ["word", "register", "nuance", "example_ko", "example_en"],
            },
        }
    },
    "required": ["candidates"],
}


def find_korean_words(ctx: ToolContext, english: str, context: str = "", count: int = 5) -> dict:
    english = (english or "").strip()
    if not english:
        return fail("No English word or phrase was given.", "Call again with english='awkward' (for example).")
    if korean.has_hangul(english) and not re.search(r"[A-Za-z]", english):
        return fail("That input is Korean.", "For Korean → English, call lookup_word.")
    count = max(2, min(int(count or 5), 8))
    key = cache.make_key("ko_words", english.lower(), (context or "").strip().lower(), count)
    return cache.cached(key, 14 * cache.DAY, lambda: _find_korean_words(english, context, count))


def _find_korean_words(english: str, context: str, count: int) -> dict:
    data = llm.generate_json(
        f"An advanced learner of Korean wants Korean equivalents for the English '{english}'."
        + (f" Situation: {context}." if context else "")
        + f" Give {count} candidates that differ in nuance or register. Use dictionary forms. Include colloquial "
          "and formal options when both exist; include slang only if it is genuinely common. Order from most to "
          "least commonly used.",
        CANDIDATES_SCHEMA,
    )
    cands = [c for c in data.get("candidates", []) if korean.has_hangul(c.get("word", ""))][:count]
    checks = _parallel(lambda c: dictionaries.quick_check(_norm(c["word"])), cands) if dictionaries.any_configured() else [None] * len(cands)
    out = []
    for c, chk in zip(cands, checks):
        w = _norm(c["word"])
        out.append({
            "word": w,
            "romanization": korean.romanize_headword(w, (chk or {}).get("pronunciation", "")),
            "register": c.get("register", ""),
            "nuance": c.get("nuance", ""),
            "example_ko": c.get("example_ko", ""),
            "example_en": c.get("example_en", ""),
            "verified": bool(chk) if dictionaries.any_configured() else None,
            "dictionary": (chk or {}).get("source_label", ""),
            "dictionary_english": (chk or {}).get("english_word", ""),
            "level": (chk or {}).get("level_en", ""),
        })
    note = ("verified=true means the word exists in a NIKL dictionary. verified=false usually means slang or a "
            "phrase; label it as unverified.")
    if not dictionaries.any_configured():
        note = "No dictionary keys are configured, so candidates are unverified. Say so."
    return ok(english=english, candidates=out, note=note)


# =============================================================================
# 3. search_slang — slang/new words: 우리말샘 + Google-Search-grounded explanation
# =============================================================================

SLANG_FIELDS = ["Meaning", "Literal breakdown", "Origin", "Register", "Example", "Still current?"]


def _parse_labeled(text: str) -> dict:
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        line = line.strip().lstrip("*-• ").replace("**", "")
        m = re.match(r"^([A-Za-z ?]+):\s*(.*)$", line)
        if m and m.group(1).strip() in SLANG_FIELDS:
            current = m.group(1).strip()
            out[current] = m.group(2).strip()
        elif current and line:
            out[current] += " " + line
    return out


def search_slang(ctx: ToolContext, term: str, context: str = "") -> dict:
    term = _norm(term)
    if not term or not korean.has_hangul(term):
        return fail("Pass the Korean slang term in Hangul.", "e.g. term='킹받다'.")
    entry, _errors, _ = dictionaries.lookup(term)
    prompt = (
        f'Explain the Korean slang or new expression "{term}" for an advanced learner. '
        + (f'It appeared in: "{context}". ' if context else "")
        + f"Today's date is {date.today().isoformat()}.\n"
        "Reply in English using exactly these labeled lines:\n"
        "Meaning: <one sentence>\n"
        "Literal breakdown: <how the parts combine, if relevant>\n"
        "Origin: <where and roughly when it came from>\n"
        "Register: <who uses it and where; how casual or rude it is>\n"
        "Example: <one natural Korean example> — <English translation>\n"
        "Still current?: <rising, mainstream, or dated as of today>\n"
        "Only state what the search results support. If they're thin or disagree, say so in that line."
    )
    slang_key = cache.make_key("slang", term, (context or "").strip())
    try:
        hit = cache.get(slang_key)
        if hit:
            text, sources = hit["text"], hit["sources"]
        else:
            text, sources = llm.grounded_answer(prompt)
            cache.put(slang_key, {"text": text, "sources": sources}, 3 * cache.DAY)
    except llm.LLMError as exc:
        if entry:
            saved = store.record_lookup(ctx.user, entry, context)
            return ok(term=term, dictionary_entry=entry, web=None, sources=[], saved_to_word_bank=bool(saved),
                      warning=f"Web search failed ({exc}); answer from the dictionary entry only.")
        return fail(f"Web search failed: {exc}", "Answer from your own knowledge, clearly labeled as unverified.")
    fields = _parse_labeled(text)
    meaning = fields.get("Meaning", "")
    bank_entry = entry or {
        "word": term, "romanization": korean.romanize(term), "gloss": meaning[:120],
        "senses": [{"english_word": meaning[:120], "english_definition": meaning}], "level_en": "slang",
        "source": "web", "examples": [fields.get("Example", "")] if fields.get("Example") else [],
    }
    store.record_lookup(ctx.user, bank_entry, context, source_override="" if entry else "web")
    return ok(
        term=term,
        romanization=(entry or {}).get("romanization") or korean.romanize(term),
        in_dictionary=bool(entry),
        dictionary_entry={k: entry[k] for k in ("source_label", "senses", "level_en")} if entry else None,
        web=fields or {"Summary": text},
        sources=sources,
        reliability="Web-sourced: summarize it and point the user to the sources." if not entry
        else "In a NIKL dictionary, plus web context.",
        saved_to_word_bank=True,
    )


# =============================================================================
# 4. word_trend — Naver DataLab search interest over time
# =============================================================================

def _trend_summary(points: list[dict]) -> dict:
    if not points:
        return {}
    this_month = date.today().strftime("%Y-%m")
    full = points[:-1] if points[-1]["period"].startswith(this_month) and len(points) > 1 else points
    peak = max(full, key=lambda p: p["ratio"])
    latest = full[-1]
    last3 = [p["ratio"] for p in full[-3:]]
    prev3 = [p["ratio"] for p in full[-6:-3]] or last3
    a, b = sum(last3) / len(last3), (sum(prev3) / len(prev3)) or 0.01
    direction = "rising" if a > b * 1.25 else "falling" if a < b * 0.8 else "steady"
    return {"peak_period": peak["period"], "peak_ratio": peak["ratio"], "latest_period": latest["period"],
            "latest_ratio": latest["ratio"], "latest_vs_peak_pct": round(100 * latest["ratio"] / peak["ratio"]) if peak["ratio"] else 0,
            "direction_last_3_periods": direction}


def word_trend(ctx: ToolContext, terms: list[str], months: int = 24) -> dict:
    terms = [_norm(t) for t in (terms or []) if _norm(t)][:5]
    if not terms:
        return fail("No terms given.", "Pass 1–5 Korean words, e.g. terms=['킹받다'].")
    months = max(3, min(int(months or 24), 120))
    try:
        data = naver.search_trend(terms, months)
    except naver.NaverNotConfigured:
        return fail("Search-trend data isn't configured on this deployment.",
                    "Tell the user the chart is unavailable; use search_slang's 'Still current?' line instead.")
    except naver.NaverError as exc:
        return fail(f"Naver DataLab: {exc}", "Tell the user the trend chart couldn't load and continue without it.")
    for s in data["series"]:
        s["summary"] = _trend_summary(s["points"])
        s["romanization"] = korean.romanize(s["term"])
    return ok(**data, note=("Values are Naver search interest relative to the highest point among these terms "
                            "(100 = peak), not absolute counts. The current month is partial."))


# =============================================================================
# 5. check_naturalness — compare phrasings by real usage counts
# =============================================================================

def check_naturalness(ctx: ToolContext, phrases: list[str]) -> dict:
    phrases = [_norm(p) for p in (phrases or []) if korean.has_hangul(p or "")][:5]
    if len(phrases) < 2:
        return fail("Need at least two Korean phrasings to compare.",
                    "e.g. phrases=['결정을 내리다', '결정을 하다']. Use the conjugated form people would actually write.")

    def count(p):
        try:
            blog = naver.search_count(p, "blog")
            news = naver.search_count(p, "news")
            return {"phrase": p, "blog_results": blog["total"], "news_results": news["total"],
                    "example": blog["snippet"] or news["snippet"]}
        except naver.NaverNotConfigured:
            raise
        except naver.NaverError as exc:
            return {"phrase": p, "error": str(exc)}

    try:
        rows = _parallel(count, phrases)
    except naver.NaverNotConfigured:
        return fail("Usage counts need the Naver Search API, which isn't configured here.",
                    "Answer from your own knowledge and say the comparison is unverified.")
    total = sum(r.get("blog_results", 0) + r.get("news_results", 0) for r in rows) or 1
    for r in rows:
        r["romanization"] = korean.romanize(r["phrase"])
        r["share_pct"] = round(100 * (r.get("blog_results", 0) + r.get("news_results", 0)) / total, 1)
    rows.sort(key=lambda r: r.get("share_pct", 0), reverse=True)
    return ok(results=rows, note=("Counts are Naver's estimated totals for the exact phrase in blogs and news. Use them "
                                  "to compare options, not as precise frequencies."))


# =============================================================================
# 6. mine_vocabulary — find advanced words in a pasted passage
# =============================================================================

BASIC = set("""하다 되다 있다 없다 같다 보다 이다 아니다 그렇다 이렇다 저렇다 어떻다 많다 적다 좋다 나쁘다 크다 작다
오다 가다 주다 받다 알다 모르다 싶다 않다 말다 먹다 자다 사다 살다 쓰다 읽다 듣다 만나다 만들다 나오다 나가다 들어가다
사람 것 때 수 등 일 말 년 월 일 오늘 내일 어제 지금 진짜 너무 정말 그냥 다시 또 더 잘 안 못 좀 다 많이 제일 아주
우리 저희 생각 시간 문제 경우 정도 이번 다음 처음 마지막 이유 의미 사실 부분 하나 자기 그때 여기 거기 저기 같이
보이다 생각하다 말하다 시작하다 사용하다 이야기 얘기 친구 사회 나라 세계 회사 학교 집 돈 오늘날""".split())


def mine_vocabulary(ctx: ToolContext, text: str, max_words: int = 12) -> dict:
    text = (text or "").strip()
    if not korean.has_hangul(text):
        return fail("The passage has no Korean in it.", "Ask the user to paste Korean text (an article, message, or post).")
    if len(text) > 4000:
        text = text[:4000]
    max_words = max(3, min(int(max_words or 12), 25))
    counts: dict[str, int] = {}
    for lemma in korean.content_lemmas(text):
        if lemma in BASIC or not korean.has_hangul(lemma) or len(lemma.replace("다", "")) < 2:
            continue
        counts[lemma] = counts.get(lemma, 0) + 1
    bank = {w["word"]: w for w in store.all_words(ctx.user)}
    candidates = [w for w in counts if bank.get(w, {}).get("mastery", 0) < MASTERED_AT][:30]
    if not candidates:
        return ok(words=[], note="Every content word here is basic or already mastered.")

    if not dictionaries.any_configured():
        words = [{"word": w, "romanization": korean.romanize(w), "count": counts[w], "in_bank": w in bank,
                  "level": "unknown", "gloss": ""} for w in candidates[:max_words]]
        return ok(words=words, note="No dictionary keys configured, so words weren't filtered by level.")

    checks = _parallel(dictionaries.quick_check, candidates)
    rank = {"advanced": 0, "not in learner's dictionary": 1, "intermediate": 2, "unrated": 3}
    words = []
    for w, chk in zip(candidates, checks):
        if not chk or chk.get("level_en") == "beginner":
            continue
        words.append({"word": w, "romanization": korean.romanize_headword(w, chk.get("pronunciation", "")),
                      "gloss": chk.get("english_word") or chk.get("definition_ko", "")[:80],
                      "level": chk.get("level_en", ""), "dictionary": chk.get("source_label", ""),
                      "count": counts[w], "in_bank": w in bank})
    words.sort(key=lambda x: rank.get(x["level"], 4))
    return ok(words=words[:max_words], scanned_words=len(counts),
              note="Beginner words, basic words, and words you've mastered were filtered out. Words are NOT saved "
                   "automatically; the user can save the ones they want.")


# =============================================================================
# 7. explore_domain — themed vocabulary, verified in dictionaries
# =============================================================================

LEVEL_GUIDE = {
    "upper-intermediate": "words fluent heritage speakers often understand but rarely use themselves",
    "advanced": "vocabulary from news, workplaces, and educated conversation",
    "native-level": ("idiomatic, literary, or specialized words that even fluent learners rarely know; high-level "
                     "words an advanced learner might also know are fine too"),
}

EXPLORE_SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "dictionary form"},
                    "english": {"type": "string"},
                    "why_useful": {"type": "string", "description": "one sentence: when you'd hear or need it"},
                    "example_ko": {"type": "string"},
                    "example_en": {"type": "string"},
                },
                "required": ["word", "english", "why_useful", "example_ko", "example_en"],
            },
        }
    },
    "required": ["words"],
}


def _verify(items: list[dict]) -> list[dict | None]:
    if not dictionaries.any_configured():
        return [None] * len(items)
    return _parallel(lambda i: dictionaries.quick_check(_norm(i["word"])), items)


def _card(item: dict, chk: dict | None) -> dict:
    w = item["word"]
    return {**item, "romanization": korean.romanize_headword(w, (chk or {}).get("pronunciation", "")),
            "level": (chk or {}).get("level_en", ""), "dictionary": (chk or {}).get("source_label", ""),
            "verified": bool(chk) if dictionaries.any_configured() else None}


# Each topic pool is built from three smaller requests made in parallel, each from
# a different angle. Smaller responses come back faster, the angles keep the
# batches from repeating each other, and each batch is shown as soon as it's verified.
ANGLES = [
    "words people use when talking about it in everyday conversation",
    "words from news articles, formal writing, and official settings about it",
    "expressions, idiomatic phrases, and nuanced words natives use about it",
]
POOL_TTL = 14 * cache.DAY


def _generate_batch(category: str, level: str, angle: str, n: int, avoid: list[str]) -> list[dict]:
    data = llm.generate_json(
        f"List {n} Korean words for a learner exploring the topic '{category}', focusing on {angle}. "
        f"Level: {LEVEL_GUIDE[level]}. Use dictionary forms; single words or very common fixed expressions only. "
        "Every word must be a real, standard Korean word. Keep why_useful under 12 words and each example under "
        "12 words." + (f" Do not include: {', '.join(avoid[:120])}." if avoid else ""),
        EXPLORE_SCHEMA,
    )
    out = []
    for i in data.get("words", []):
        w = _norm(i.get("word", ""))
        if korean.has_hangul(w):
            out.append({**i, "word": w})
    return out


def _grow_pool(category: str, level: str, existing: list[dict], on_batch=None, per_batch: int = 8,
               executor: ThreadPoolExecutor | None = None) -> list[dict]:
    """Generate three batches in parallel, verify each, and return the new cards
    (verified first). `on_batch(cards)` is called as each batch finishes."""
    seen = {c["word"] for c in existing}
    avoid = sorted(seen)
    lock = threading.Lock()
    new_cards: list[dict] = []

    def one(angle: str) -> None:
        items = _generate_batch(category, level, angle, per_batch, avoid)
        with lock:
            items = [i for i in items if i["word"] not in seen]
            seen.update(i["word"] for i in items)
        cards = [_card(i, c) for i, c in zip(items, _verify(items))]
        with lock:
            new_cards.extend(cards)
        if on_batch and cards:
            on_batch(cards)

    futures = [(executor or _gen_pool).submit(one, angle) for angle in ANGLES]
    errors = []
    for f in futures:
        try:
            f.result()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    if errors and not new_cards:
        raise errors[0]
    return sorted(new_cards, key=lambda c: c["verified"] is False)


def topic_pool(category: str, level: str, on_batch=None, background: bool = False) -> list[dict]:
    """The shared, cached pool of word cards for a topic and level (not personal:
    filtering out words a user already knows happens afterwards)."""
    key = cache.make_key("pool", category.lower().strip(), level)
    pool = cache.get(key)
    if pool is not None:
        return pool
    with cache.build_lock(key):
        pool = cache.get(key)
        if pool is not None:
            return pool
        pool = _grow_pool(category, level, [], on_batch, executor=_warm_gen_pool if background else None)
        if pool:
            cache.put(key, pool, POOL_TTL)
        return pool


def extend_topic_pool(category: str, level: str, pool: list[dict], on_batch=None) -> list[dict]:
    key = cache.make_key("pool", category.lower().strip(), level)
    with cache.build_lock(key):
        bigger = pool + _grow_pool(category, level, pool, on_batch)
        cache.put(key, bigger, POOL_TTL)
        return bigger


def explore_domain(ctx: ToolContext, category: str, level: str = "advanced", count: int = 9) -> dict:
    category = (category or "").strip()[:80]
    if not category:
        return fail("No category given.", "e.g. category='finance' or 'dating and relationships'.")
    level = level if level in LEVEL_GUIDE else "advanced"
    count = max(4, min(int(count or 9), 12))
    known = {w["word"] for w in store.all_words(ctx.user)}
    # Words already shown for this topic+level in this session, so "Show more" never repeats.
    shown_all = ctx.session.setdefault("explore_shown", {})
    key = f"{category.lower()}|{level}"
    shown = set(shown_all.get(key, []))

    picked: list[dict] = []
    picked_words: set[str] = set()

    def take(cards: list[dict], include_unverified: bool) -> None:
        fresh = []
        for c in cards:
            if len(picked) >= count:
                break
            if c["word"] in known or c["word"] in shown or c["word"] in picked_words:
                continue
            if c["verified"] is False and not include_unverified:
                continue
            picked.append(c)
            picked_words.add(c["word"])
            fresh.append(c)
        if fresh:  # stream cards to the UI as they're ready
            ctx.emit({"type": "partial", "tool": "explore_domain", "words": fresh})

    pool = topic_pool(category, level, on_batch=lambda cards: take(cards, False))
    take(pool, False)  # dictionary-verified words first
    take(pool, True)   # then clearly labeled unverified ones, already in the pool (no extra wait)
    if len(picked) < count:  # pool used up for this user: only now generate more
        pool = extend_topic_pool(category, level, pool, on_batch=lambda cards: take(cards, False))
        take(pool, False)
        take(pool, True)

    filled = sum(1 for c in picked if c["verified"] is False)
    shown_all[key] = sorted(shown | picked_words)[-300:]
    if not dictionaries.any_configured():
        note = "Dictionary keys aren't configured, so these words are unverified."
    elif filled:
        note = (f"{len(picked) - filled} words are in a NIKL dictionary. {filled} weren't found in the configured "
                "dictionaries (common for high-level vocabulary, which the learner's dictionary doesn't cover); "
                "they're marked unverified, so say so.")
    else:
        note = "Every word was checked against NIKL dictionaries."
    return ok(category=category, level=level, words=picked, unverified_count=filled,
              shown_this_session=len(shown_all[key]), note=note)


# =============================================================================
# 8. hanja_family — split a Sino-Korean word into roots, find related words
# =============================================================================

HANJA_SCHEMA = {
    "type": "object",
    "properties": {
        "roots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hanja": {"type": "string"},
                    "reading": {"type": "string", "description": "Korean reading (one Hangul syllable)"},
                    "meaning_en": {"type": "string", "description": "core meaning of the character, a few words"},
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"word": {"type": "string"}, "hanja": {"type": "string"},
                                           "english": {"type": "string"}},
                            "required": ["word", "hanja", "english"],
                        },
                    },
                },
                "required": ["hanja", "reading", "meaning_en", "words"],
            },
        }
    },
    "required": ["roots"],
}


def hanja_family(ctx: ToolContext, word: str) -> dict:
    word = _norm(word)
    if not word or not korean.has_hangul(word):
        return fail("Pass a Korean word in Hangul.", "e.g. word='경제'.")
    key = cache.make_key("hanja", word)
    hit = cache.get(key)
    if hit:
        return hit
    result = _hanja_family(word)
    if result.get("ok"):
        cache.put(key, result, 30 * cache.DAY)
    return result


def _hanja_family(word: str) -> dict:
    if not (config.KRDICT_API_KEY or config.STDICT_API_KEY):
        return fail("Hanja roots need the krdict or stdict API key, which isn't configured.",
                    "Explain the roots from your own knowledge and label it unverified.")
    chk = dictionaries.quick_check(word)
    if not chk:
        return fail(f"'{word}' wasn't found in the dictionary.",
                    "Try the dictionary form of the word, or skip the hanja breakdown.")
    entry = {"word": word, "origin": chk.get("origin", ""),
             "romanization": korean.romanize_headword(word, chk.get("pronunciation", ""))}
    chars = korean.hanja_chars(entry.get("origin", ""))
    if not chars:
        return ok(word=entry["word"], origin=entry.get("origin", ""), has_hanja=False,
                  message="This is a native Korean word (고유어) or a loanword, so it has no hanja roots.")
    origin = entry["origin"]
    readings = {}
    if len(origin) == len(entry["word"]):
        readings = {ch: entry["word"][i] for i, ch in enumerate(origin) if korean.HANJA_CHAR.match(ch)}
    data = llm.generate_json(
        f"The Korean word {entry['word']} is written {origin} in hanja. For each of these characters: "
        f"{', '.join(chars[:4])}, give its Korean reading, its core meaning, and 5 common Korean words that use that "
        "same character with the same meaning (with each word's full hanja). Prefer words an advanced learner would "
        f"find useful. Don't include {entry['word']} itself.",
        HANJA_SCHEMA,
    )
    roots = []
    for root in data.get("roots", [])[:4]:
        ch = root.get("hanja", "")[:1]
        fam = [w for w in root.get("words", []) if korean.has_hangul(w.get("word", ""))][:6]
        checks = _parallel(lambda w: dictionaries.quick_check(_norm(w["word"])), fam)
        verified = []
        for w, chk in zip(fam, checks):
            if not chk:
                continue
            confirmed = ch in (chk.get("origin") or "") if chk.get("origin") else None
            if confirmed is False:
                continue  # dictionary says this word doesn't contain that character
            verified.append({"word": _norm(w["word"]), "hanja": chk.get("origin") or w.get("hanja", ""),
                             "english": chk.get("english_word") or w.get("english", ""),
                             "romanization": korean.romanize_headword(_norm(w["word"]), chk.get("pronunciation", "")),
                             "root_confirmed_by_dictionary": bool(confirmed)})
        roots.append({"hanja": ch, "reading": readings.get(ch) or root.get("reading", ""),
                      "meaning": root.get("meaning_en", ""), "family": verified[:5]})
    return ok(word=entry["word"], origin=origin, romanization=entry["romanization"], has_hanja=True, roots=roots,
              note="Family words were checked in the dictionary; words whose listed hanja didn't contain the root were dropped.")


# =============================================================================
# 9–10. Texting drill
# =============================================================================

PERSONAS = {
    "close friend": {"name": "민지", "relationship": "your close friend", "level": "casual"},
    "coworker": {"name": "김 대리", "relationship": "a coworker you're friendly with", "level": "polite"},
    "boss": {"name": "박 팀장님", "relationship": "your team lead at work", "level": "formal"},
    "older relative": {"name": "이모", "relationship": "your aunt", "level": "polite"},
    "crush": {"name": "서준", "relationship": "someone you've just started talking to (썸)", "level": "polite"},
    "stranger": {"name": "중고거래 판매자", "relationship": "a seller on a secondhand marketplace app", "level": "polite"},
}

# Used only when the word bank is empty: words fluent learners often half-know.
STARTER_WORDS = [
    ("서운하다", "to feel hurt/let down by someone close"), ("아쉽다", "to be a shame; to wish it had gone better"),
    ("답답하다", "to feel stifled or frustrated"), ("억울하다", "to feel wronged"), ("뿌듯하다", "to feel proud and fulfilled"),
    ("민망하다", "to feel embarrassed (for oneself or others)"), ("어색하다", "to be awkward"),
    ("부담스럽다", "to feel pressured or burdened"), ("망설이다", "to hesitate"), ("설레다", "to feel a flutter of excitement"),
    ("핑계", "an excuse"), ("눈치", "the ability to read a situation"), ("애매하다", "to be ambiguous or borderline"),
    ("흐지부지되다", "to fizzle out"), ("생색내다", "to make a show of a favor"), ("미루다", "to put off"),
    ("소홀하다", "to neglect"), ("기특하다", "to be admirable (said of someone younger)"), ("얄밉다", "to be annoyingly smug"),
    ("어이없다", "to be absurd; to leave one speechless"), ("서먹하다", "to feel distant or awkward with someone"),
    ("무리하다", "to overdo it"), ("체면", "face; one's dignity"), ("꼼꼼하다", "to be meticulous"),
    ("털털하다", "to be easygoing and unfussy"), ("능청스럽다", "to be coolly sly"), ("황당하다", "to be absurd/baffling"),
    ("한결같다", "to be consistent and unchanging"), ("허세", "showing off; bluster"), ("뒤끝", "lingering grudge"),
]

PRODUCE_SCHEMA = {
    "type": "object",
    "properties": {
        "partner_message_ko": {"type": "string", "description": "1–2 short, natural text messages from the partner that set up the situation"},
        "situation_en": {"type": "string", "description": "one sentence of context in English"},
        "task_en": {"type": "string", "description": "What the learner should text back, in natural English (≤ 20 words). Must make the target word the most natural choice without containing a literal translation of it in quotes."},
        "model_answer_ko": {"type": "string", "description": "a natural reply using the target word at the right speech level"},
        "hint_en": {"type": "string", "description": "a hint that points toward the word without revealing it"},
    },
    "required": ["partner_message_ko", "situation_en", "task_en", "model_answer_ko", "hint_en"],
}

RECOGNIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "partner_message_ko": {"type": "string", "description": "1–2 natural text messages that use the target word"},
        "situation_en": {"type": "string"},
        "question_en": {"type": "string", "description": "e.g. 'What is she telling you, and how does she feel?'"},
        "expected_meaning_en": {"type": "string", "description": "what a good answer should capture"},
        "translation_en": {"type": "string"},
        "hint_en": {"type": "string"},
    },
    "required": ["partner_message_ko", "situation_en", "question_en", "expected_meaning_en", "translation_en", "hint_en"],
}


def _bank_word(ctx: ToolContext, recent: set[str]) -> tuple[str, str] | None:
    due = [w for w in store.list_words(ctx.user, "due", limit=10) if korean.has_hangul(w.get("word", ""))]
    pool = [w for w in due if w["word"] not in recent] or due
    if not pool:
        pool = [w for w in store.list_words(ctx.user, "recent", limit=40)
                if w.get("mastery", 0) < 5 and w["word"] not in recent]
    if not pool:
        return None
    w = random.choice(pool[:6])
    return w["word"], w.get("gloss", "")


def _random_word(recent: set[str]) -> tuple[str, str]:
    pool = [p for p in STARTER_WORDS if p[0] not in recent] or STARTER_WORDS
    return random.choice(pool)


def _topic_word(topic: str, level: str, recent: set[str]) -> tuple[str, str] | None:
    pool = topic_pool(topic, level)
    candidates = [c for c in pool if c["word"] not in recent]
    preferred = [c for c in candidates if c["verified"] is not False] or candidates
    if not preferred:
        return None
    c = random.choice(preferred[:12])
    return c["word"], c.get("english", "")


def _pick_drill_word(ctx: ToolContext, source: str, topic: str, level: str) -> tuple[str, str, str]:
    """Returns (word, gloss, source_note)."""
    recent = set(ctx.session.get("recent_drill_words", []))
    if source == "topic" and topic:
        got = _topic_word(topic, level, recent)
        if got:
            article = "An" if level[0] in "aeiou" else "A"
            return got[0], got[1], f"{article} {level} word about {topic}"
        source = "random"
    if source == "word_bank":
        got = _bank_word(ctx, recent)
        if got:
            return got[0], got[1], "From your word bank"
        word, gloss = _random_word(recent)
        return word, gloss, "A random useful word (your word bank is empty)"
    word, gloss = _random_word(recent)
    return word, gloss, "A random useful word"


def _build_drill(user: str, recent: list[str], mode: str, persona_key: str, source: str, topic: str, level: str,
                 word: str = "") -> dict:
    """Pick a word and write the scenario. Pure: doesn't touch the session, so it
    can also run in the background to pre-generate the next drill."""
    p = PERSONAS[persona_key]
    if word:
        target, gloss, source_note = word, "", "The word you chose"
    else:
        target, gloss, source_note = _pick_drill_word(ToolContext(user=user, session={"recent_drill_words": recent}),
                                                      source, topic, level)
    if not gloss and dictionaries.any_configured():
        chk = dictionaries.quick_check(target)
        gloss = (chk or {}).get("english_word") or (chk or {}).get("definition_ko", "")
    level_label = korean.SPEECH_LEVELS[p["level"]]
    base = (f"Write a realistic KakaoTalk-style texting scenario between a learner and {p['name']} "
            f"({p['relationship']}). The partner writes in {level_label} as fits the relationship. "
            f"Target word: {target}" + (f" (meaning: {gloss})" if gloss else "") + ". Keep messages short and natural, "
            "the way people really text (casual punctuation, no romanization, no English in Korean messages).")
    for _attempt in range(2):
        if mode == "produce":
            data = llm.generate_json(
                base + f" The learner must reply in {level_label} and the most natural reply should use the target "
                "word (any conjugation). The task tells the learner, in English, what to say back.",
                PRODUCE_SCHEMA)
            check_text = data.get("model_answer_ko", "")
        else:
            data = llm.generate_json(
                base + " The partner's message must contain the target word (any conjugation); the learner will "
                "explain in English what the partner means.", RECOGNIZE_SCHEMA)
            check_text = data.get("partner_message_ko", "")
        if korean.target_used(target, check_text)["used"]:
            break
    answer_key = {
        "target_word": target,
        "target_romanization": korean.romanize(target),
        "gloss": gloss,
        "hint_en": data.get("hint_en", ""),
        "model_answer_ko": data.get("model_answer_ko", ""),
        "expected_meaning_en": data.get("expected_meaning_en", ""),
        "translation_en": data.get("translation_en", ""),
    }
    return {
        "drill_id": uuid.uuid4().hex[:8], "mode": mode, "persona": persona_key, "partner_name": p["name"],
        "relationship": p["relationship"], "word_source": source_note, "expected_speech_level": level_label,
        "expected_level": p["level"], "partner_message_ko": data.get("partner_message_ko", ""),
        "situation_en": data.get("situation_en", ""), "task_en": data.get("task_en") or data.get("question_en", ""),
        "answer_key": answer_key,
    }


# Pre-generated next drills, kept in memory per session (never persisted).
_prefetched: dict[str, tuple[str, dict]] = {}
_prefetch_running: set[str] = set()
_prefetch_lock = threading.Lock()


def _prefetch_next(ctx: ToolContext, settings_key: str, args: tuple) -> None:
    """Write the next drill in the background so "Next word" is nearly instant."""
    sid = ctx.session_id
    if not sid:
        return
    with _prefetch_lock:
        if sid in _prefetch_running:
            return
        _prefetch_running.add(sid)
    recent = list(ctx.session.get("recent_drill_words", []))

    def run():
        try:
            built = _build_drill(ctx.user, recent, *args)
            with _prefetch_lock:
                _prefetched[sid] = (settings_key, built)
                if len(_prefetched) > 500:
                    _prefetched.pop(next(iter(_prefetched)))
        except Exception as exc:  # noqa: BLE001
            log.info("Drill prefetch failed (will build on demand): %s", exc)
        finally:
            with _prefetch_lock:
                _prefetch_running.discard(sid)

    threading.Thread(target=run, daemon=True).start()


def start_text_drill(ctx: ToolContext, mode: str = "produce", word: str = "", persona: str = "close friend",
                     source: str = "word_bank", topic: str = "", level: str = "advanced") -> dict:
    mode = mode if mode in ("produce", "recognize") else "produce"
    persona_key = persona if persona in PERSONAS else "close friend"
    level = level if level in LEVEL_GUIDE else "advanced"
    source = source if source in ("word_bank", "topic", "random") else "word_bank"
    topic = (topic or "").strip()[:80]
    word = _norm(word)
    if word.startswith("(") or word.lower() in ("none", "(none)"):
        word = ""
    if word and not korean.has_hangul(word):
        return fail("The drill word must be Korean.", "Pass a Korean word, or leave word empty to pick one automatically.")
    args = (mode, persona_key, source, topic, level)
    settings_key = "|".join(args)
    recent = list(ctx.session.get("recent_drill_words", []))

    built = None
    if not word and ctx.session_id:
        with _prefetch_lock:
            key, ready = _prefetched.pop(ctx.session_id, (None, None))
        if key == settings_key and ready and ready["answer_key"]["target_word"] not in recent[-3:]:
            built = ready
    if built is None:
        built = _build_drill(ctx.user, recent, *args, word=word)

    target = built["answer_key"]["target_word"]
    ctx.session["recent_drill_words"] = (recent + [target])[-30:]
    ctx.session["drill"] = {"id": built["drill_id"], "mode": mode, "persona": persona_key,
                            "expected_level": built["expected_level"], "task_en": built["task_en"],
                            "partner_message_ko": built["partner_message_ko"], "answer_key": built["answer_key"],
                            "attempts": 0, "status": "active", "recorded": False}
    if not word:
        _prefetch_next(ctx, settings_key, args)
    return ok(
        **{k: v for k, v in built.items() if k != "expected_level"},
        instructions=("answer_key is hidden from the learner in the UI. Never reveal target_word or model_answer_ko "
                      "before the learner answers or gives up. hint_en may be shared if they ask for a hint."),
    )


PRODUCE_JUDGE = {
    "type": "object",
    "properties": {
        "meaning_matches": {"type": "boolean", "description": "Does the reply say what the task asked?"},
        "uses_acceptable_alternative": {"type": "boolean", "description": "If the target word is missing, is another word used that a native would accept here?"},
        "alternative_word": {"type": "string"},
        "feedback_en": {"type": "string", "description": "1–2 sentences, specific and kind"},
        "better_version_ko": {"type": "string", "description": "the learner's reply, minimally corrected to sound natural"},
        "partner_reply_ko": {"type": "string", "description": "the partner's in-character text reply"},
    },
    "required": ["meaning_matches", "uses_acceptable_alternative", "feedback_en", "better_version_ko", "partner_reply_ko"],
}

RECOGNIZE_JUDGE = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "miss"]},
        "feedback_en": {"type": "string"},
        "partner_reply_ko": {"type": "string"},
    },
    "required": ["verdict", "feedback_en", "partner_reply_ko"],
}


def _ensure_in_bank(ctx: ToolContext, word: str, gloss: str) -> None:
    if store.get_word(ctx.user, word):
        return
    entry, _e, _c = dictionaries.lookup(word) if dictionaries.any_configured() else (None, [], False)
    store.record_lookup(ctx.user, entry or {"word": word, "romanization": korean.romanize(word), "gloss": gloss,
                                            "senses": [{"english_word": gloss}], "source": "drill"})


def check_drill_answer(ctx: ToolContext, reply: str = "", gave_up: bool = False) -> dict:
    drill = ctx.session.get("drill")
    if not drill or drill.get("status") != "active":
        return fail("No drill is in progress.", "Call start_text_drill to begin a new one.")
    key = drill["answer_key"]
    target = key["target_word"]
    _ensure_in_bank(ctx, target, key.get("gloss", ""))

    def record(outcome: str) -> int:
        if drill.get("recorded"):
            return int((store.get_word(ctx.user, target) or {}).get("mastery", 0))
        drill["recorded"] = True
        return int(store.record_review(ctx.user, target, outcome).get("mastery", 0))

    if gave_up:
        drill["status"] = "revealed"
        mastery = record("miss")
        return ok(verdict="revealed", answer_key=key, mastery_after=mastery, drill_status="revealed",
                  feedback_en="Here's the answer. This word will come back sooner in your reviews.")

    reply = (reply or "").strip()[:500]
    if not reply:
        return fail("The reply was empty.", "Pass the learner's message verbatim as reply.")
    drill["attempts"] += 1

    if drill["mode"] == "recognize":
        try:
            j = llm.generate_json(
                f"A learner read this Korean text: '{drill['partner_message_ko']}'. Target word: {target}. "
                f"A good answer captures: {key['expected_meaning_en']}. The learner answered: '{reply}'. "
                "Judge leniently on wording, strictly on meaning. Then write the partner's short in-character reply.",
                RECOGNIZE_JUDGE, thinking="low")
        except llm.LLMError as exc:
            return fail(f"Couldn't grade the answer: {exc}", "Ask the learner to try again in a moment.")
        verdict = j.get("verdict", "miss")
        outcome = {"correct": "correct", "partial": "partial"}.get(verdict, "miss")
        mastery = record(outcome)
        if verdict == "correct":
            drill["status"] = "done"
        return ok(verdict=verdict, feedback_en=j.get("feedback_en", ""), partner_reply_ko=j.get("partner_reply_ko", ""),
                  answer_key=key if verdict == "correct" or drill["attempts"] >= 2 else None,
                  mastery_after=mastery, attempts=drill["attempts"], drill_status=drill["status"])

    usage = korean.target_used(target, reply)
    detected = korean.detect_speech_level(reply)
    lvl_ok = korean.level_ok(drill["expected_level"], detected)
    facts = (f"Task given to the learner: '{drill['task_en']}'. Partner's message: '{drill['partner_message_ko']}'. "
             f"Target word: {target}. Learner's reply: '{reply}'. Morphological analysis says the target word was "
             f"{'USED' if usage['used'] else 'NOT used'}. Expected speech level: "
             f"{korean.SPEECH_LEVELS[drill['expected_level']]}; detected: {korean.SPEECH_LEVELS.get(detected, detected)}.")
    try:
        j = llm.generate_json(facts + " Judge the reply as a native speaker would. Keep feedback short; mention "
                              "speech level only if it's wrong for the relationship.", PRODUCE_JUDGE, thinking="low")
    except llm.LLMError:
        j = {"meaning_matches": usage["used"], "uses_acceptable_alternative": False, "feedback_en": "",
             "better_version_ko": key["model_answer_ko"], "partner_reply_ko": ""}

    if usage["used"] and j.get("meaning_matches") and lvl_ok:
        verdict, outcome = "correct", "correct"
    elif usage["used"]:
        verdict, outcome = "close", "partial"
    elif j.get("uses_acceptable_alternative") and j.get("meaning_matches"):
        verdict, outcome = "alternative", "partial"
    else:
        verdict, outcome = "miss", "miss"
    mastery = record(outcome)
    if verdict in ("correct", "alternative") or drill["attempts"] >= 3:
        drill["status"] = "done"
    reveal = verdict in ("correct", "alternative") or drill["status"] == "done"
    return ok(
        verdict=verdict,
        used_target_word=usage["used"],
        detected_speech_level=korean.SPEECH_LEVELS.get(detected, detected),
        expected_speech_level=korean.SPEECH_LEVELS[drill["expected_level"]],
        speech_level_ok=lvl_ok,
        alternative_word=j.get("alternative_word", "") if verdict == "alternative" else "",
        feedback_en=j.get("feedback_en", ""),
        better_version_ko=j.get("better_version_ko", ""),
        partner_reply_ko=j.get("partner_reply_ko", ""),
        answer_key=key if reveal else None,
        mastery_after=mastery,
        attempts=drill["attempts"],
        drill_status=drill["status"],
        next_step=("If verdict is close or miss and the drill is still active, encourage another try or offer a hint "
                   "(hint_en) — don't reveal the word.") if not reveal else "Offer the next drill.",
    )


# =============================================================================
# 11–12. Word bank
# =============================================================================

def _compact(w: dict) -> dict:
    return {k: w.get(k) for k in ("word", "romanization", "gloss", "level", "mastery", "starred", "source",
                                  "lookups", "next_review", "note") if w.get(k) not in (None, "")} | (
        {"seen_in": w["contexts"][0]} if w.get("contexts") else {})


def get_word_bank(ctx: ToolContext, view: str = "recent", limit: int = 20, search: str = "") -> dict:
    views = ("recent", "due", "starred", "struggling", "mastered", "alphabetical")
    view = view if view in views else "recent"
    limit = max(1, min(int(limit or 20), 100))
    words = store.list_words(ctx.user, view, limit, search)
    return ok(view=view, stats=store.stats(ctx.user), words=[_compact(w) for w in words],
              note="" if words else "Nothing here yet. Words are saved automatically when the user looks them up.")


def update_word(ctx: ToolContext, word: str, action: str, note: str = "") -> dict:
    word = _norm(word)
    existing = store.get_word(ctx.user, word)
    if not existing:
        return fail(f"'{word}' isn't in the word bank.",
                    "Call lookup_word first (lookups are saved automatically), then retry.")
    if action == "delete":
        store.delete_word(ctx.user, word)
        return ok(word=word, action="deleted")
    changes = {
        "star": {"starred": True},
        "unstar": {"starred": False},
        "add_note": {"note": note.strip()[:500]},
        "mark_known": {"mastery": 5, "next_review": "9999-12-31T00:00:00+00:00"},
        "reset": {"mastery": 0, "next_review": existing.get("created_at", "")},
    }.get(action)
    if changes is None:
        return fail(f"Unknown action '{action}'.", "Use one of: star, unstar, add_note, mark_known, reset, delete.")
    if action == "add_note" and not changes["note"]:
        return fail("The note is empty.", "Pass the note text in `note`.")
    updated = store.put_word(ctx.user, word, changes)
    return ok(word=word, action=action, entry=_compact(updated))


# =============================================================================
# Declarations (what the model sees) and the executor
# =============================================================================

DECLARATIONS = {
    "lookup_word": {
        "description": (
            "Look up a Korean word in the National Institute of Korean Language dictionaries (learner's dictionary "
            "with English, then the standard dictionary, then 우리말샘). Returns numbered senses with English, "
            "pronunciation, level, hanja origin, and examples, and saves the word to the user's word bank. Accepts "
            "conjugated forms (망설여져서 → 망설이다). Pass context_sentence whenever the user quotes where they saw "
            "the word, so you can pick the right sense. Use this for every Korean → English question."),
        "parameters": {
            "type": "object",
            "properties": {
                "word": {"type": "string", "description": "One Korean word or short expression, ideally in dictionary form."},
                "context_sentence": {"type": "string", "description": "The sentence the user saw the word in, verbatim. Empty if none."},
            },
            "required": ["word"],
        },
    },
    "find_korean_words": {
        "description": (
            "English → Korean. Generates several Korean candidates for an English word or phrase, each with register "
            "(casual/neutral/formal/written/slang), a nuance note, and an example, then verifies each candidate in the "
            "NIKL dictionaries. Use when the user asks 'how do I say X' or wants options for an English concept."),
        "parameters": {
            "type": "object",
            "properties": {
                "english": {"type": "string", "description": "The English word or phrase, e.g. 'awkward' or 'to let something slide'."},
                "context": {"type": "string", "description": "Optional situation, e.g. 'texting a friend after a bad date'."},
                "count": {"type": "integer", "description": "How many candidates (2–8). Default 5."},
            },
            "required": ["english"],
        },
    },
    "search_slang": {
        "description": (
            "Explain Korean slang, internet language, abbreviations, very new words, or rare words the dictionaries "
            "don't have, using 우리말샘 (if configured) plus a live "
            "Google Search–grounded summary (meaning, origin, register, example, whether it's still current) with "
            "source links. Saves the term to the word bank. Use when lookup_word finds nothing or the term is clearly slang."),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "The slang term in Hangul, e.g. '킹받다', '갓생', '스불재'."},
                "context": {"type": "string", "description": "Where the user saw it, if they said. Empty if none."},
            },
            "required": ["term"],
        },
    },
    "word_trend": {
        "description": (
            "Chart how much Koreans have searched for up to 5 words on Naver over time, with peak, latest value, and "
            "direction. Shows whether slang is rising, peaking, or fading, and lets the user compare synonyms. The "
            "UI draws the chart from this result."),
        "parameters": {
            "type": "object",
            "properties": {
                "terms": {"type": "array", "items": {"type": "string"}, "description": "1–5 Korean terms."},
                "months": {"type": "integer", "description": "How far back to look (3–120). Default 24."},
            },
            "required": ["terms"],
        },
    },
    "check_naturalness": {
        "description": (
            "Compare 2–5 alternative Korean phrasings (collocations, verb choices, particles) by how often each exact "
            "phrase appears in Naver blogs and news, with a real example sentence. Use when the user asks which "
            "wording is more natural, e.g. '결정을 내리다' vs '결정을 하다'."),
        "parameters": {
            "type": "object",
            "properties": {
                "phrases": {"type": "array", "items": {"type": "string"},
                            "description": "Korean phrasings to compare, written the way people would actually write them."},
            },
            "required": ["phrases"],
        },
    },
    "mine_vocabulary": {
        "description": (
            "Scan a pasted Korean passage (article, message, post, lyrics the user wrote) with a morphological analyzer, "
            "convert words to dictionary form, drop basic and already-mastered words, and return the intermediate/"
            "advanced words worth learning, with glosses. Does not save automatically."),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The Korean passage, verbatim (up to ~4000 characters)."},
                "max_words": {"type": "integer", "description": "Maximum words to return (3–25). Default 12."},
            },
            "required": ["text"],
        },
    },
    "explore_domain": {
        "description": (
            "Suggest Korean vocabulary for a topic (finance, politics, science, dating, workplace, feelings, internet "
            "culture, food, or any custom topic) at a chosen level, skipping words the user already has and words "
            "already shown for that topic in this session (so calling it again gives a fresh set). Words are checked "
            "against the dictionary; any that couldn't be verified are labeled. Each comes with why it's useful and "
            "an example."),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Topic, e.g. 'finance and investing' or 'dating and relationships'."},
                "level": {"type": "string", "enum": ["upper-intermediate", "advanced", "native-level"]},
                "count": {"type": "integer", "description": "How many words (4–12). Default 9."},
            },
            "required": ["category"],
        },
    },
    "hanja_family": {
        "description": (
            "Break a Sino-Korean word into its hanja (Chinese character) roots using the dictionary's origin field, "
            "and find other common words that share each root (經 → 경제, 경영, 경험). Use when the user wants to "
            "understand a word's structure or learn related words. Native Korean words have no hanja."),
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string", "description": "A Korean word in Hangul, e.g. '경제'."}},
            "required": ["word"],
        },
    },
    "start_text_drill": {
        "description": (
            "Start a texting drill. mode='produce': the partner texts the learner and the learner must reply in Korean "
            "using the target word. mode='recognize': the partner's text uses the target word and the learner explains "
            "it in English. Chooses the word from `source`: word_bank (due for review first; falls back to a random "
            "word if the bank is empty), topic (a dictionary-checked word for `topic` at `level`), or random (a "
            "useful word fluent learners often half-know). A non-empty `word` overrides source. Returns a hidden "
            "answer_key."),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["produce", "recognize"]},
                "word": {"type": "string", "description": "Specific Korean word to practice. Empty = choose using source."},
                "source": {"type": "string", "enum": ["word_bank", "topic", "random"]},
                "topic": {"type": "string", "description": "Topic when source is 'topic', e.g. 'finance, investing, and the economy'."},
                "level": {"type": "string", "enum": ["upper-intermediate", "advanced", "native-level"]},
                "persona": {"type": "string", "enum": list(PERSONAS.keys()),
                            "description": "Who the learner is texting; sets the expected speech level."},
            },
            "required": ["mode"],
        },
    },
    "check_drill_answer": {
        "description": (
            "Grade the learner's reply to the active drill. Uses morphological analysis to detect the target word in "
            "any conjugation, checks the speech level against the relationship, judges meaning and alternatives, "
            "writes the partner's reply, and updates spaced-repetition mastery. Pass gave_up=true if the learner asks "
            "to see the answer."),
        "parameters": {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "description": "The learner's message, verbatim."},
                "gave_up": {"type": "boolean", "description": "True if the learner wants the answer revealed."},
            },
            "required": [],
        },
    },
    "get_word_bank": {
        "description": (
            "Read the user's saved words. Views: recent, due (due for review), starred, struggling, mastered, "
            "alphabetical. Includes stats (total, due, mastered). Use when the user asks about their words or "
            "progress, or wants a quiz on what they've looked up."),
        "parameters": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["recent", "due", "starred", "struggling", "mastered", "alphabetical"]},
                "limit": {"type": "integer", "description": "1–100. Default 20."},
                "search": {"type": "string", "description": "Optional filter text (Korean, English gloss, or romanization)."},
            },
            "required": [],
        },
    },
    "update_word": {
        "description": "Change a saved word: star, unstar, add_note, mark_known, reset (progress), or delete.",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
                "action": {"type": "string", "enum": ["star", "unstar", "add_note", "mark_known", "reset", "delete"]},
                "note": {"type": "string", "description": "Note text when action is add_note."},
            },
            "required": ["word", "action"],
        },
    },
}

FUNCTIONS = {
    "lookup_word": lookup_word,
    "find_korean_words": find_korean_words,
    "search_slang": search_slang,
    "word_trend": word_trend,
    "check_naturalness": check_naturalness,
    "mine_vocabulary": mine_vocabulary,
    "explore_domain": explore_domain,
    "hanja_family": hanja_family,
    "start_text_drill": start_text_drill,
    "check_drill_answer": check_drill_answer,
    "get_word_bank": get_word_bank,
    "update_word": update_word,
}


def execute(name: str, args: dict | None, ctx: ToolContext) -> dict:
    fn = FUNCTIONS.get(name)
    if fn is None:
        return fail(f"Unknown tool '{name}'.", f"Available tools: {', '.join(FUNCTIONS)}.")
    accepted = set(inspect.signature(fn).parameters) - {"ctx"}
    clean = {k: v for k, v in (args or {}).items() if k in accepted}
    try:
        return fn(ctx, **clean)
    except TypeError as exc:
        return fail(f"Bad arguments for {name}: {exc}", "Check the required parameters in the tool description.")
    except llm.LLMError as exc:
        return fail(str(exc), "Tell the user this step couldn't run right now and continue with what you have.")
    except dictionaries.DictError as exc:
        return fail(f"Dictionary error: {exc}", "Tell the user the dictionary is unreachable and label any answer unverified.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Tool %s crashed", name)
        return fail(f"{name} failed unexpectedly ({exc.__class__.__name__}).",
                    "Tell the user this step failed and continue with what you have.")
