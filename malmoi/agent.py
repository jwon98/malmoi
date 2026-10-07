"""The agent loop.

One /chat turn:
  1. Load the session (its own history; sessions never share state).
  2. Send history + the new message to Gemini with the tools for this mode.
  3. While Gemini asks for function calls: run each tool, record
     {name, args, result}, and send the results back.
  4. Return the final text and every tool call.

Function calling is done manually (not the SDK's automatic mode) so every call
can be recorded for the UI and so tools receive the signed-in user's context.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from google.genai import types

from . import cache, config, llm
from .storage import store
from .tools import DECLARATIONS, ToolContext, execute

log = logging.getLogger("malmoi.agent")
_tool_pool = ThreadPoolExecutor(max_workers=6)

MODE_TOOLS = {
    "ask": ["lookup_word", "find_korean_words", "search_slang", "word_trend", "check_naturalness",
            "mine_vocabulary", "hanja_family", "explore_domain", "get_word_bank", "update_word"],
    "drill": ["start_text_drill", "check_drill_answer", "lookup_word", "get_word_bank"],
    "explore": ["explore_domain", "lookup_word", "hanja_family", "word_trend"],
}

COMMON = """You are Malmoi (말모이), a Korean vocabulary coach for advanced learners: people who are conversational or
fluent but keep running into words whose nuance a one-word translation doesn't capture. The name comes from the first
Korean dictionary manuscript, compiled word by word in the 1910s.

Ground rules
- Never define a Korean word from memory when a tool can check it. Call lookup_word (or search_slang for slang).
- Be honest about sources. Results say where they came from: NIKL dictionaries are verified; search_slang results are
  web-sourced; anything marked verified=false or source "ai" is unverified. Say so in a few words when it matters.
- If a tool returns ok=false, read its hint and follow it. Don't retry the same failing call more than once.
- The interface automatically shows romanization above every Korean word and a play button for audio, so never write
  romanization or pronunciation guides yourself.
- The interface renders rich cards from tool results (dictionary entries, charts, candidate lists). Don't repeat every
  field; add the explanation a good tutor would give: nuance, when natives use it, what it's often confused with.
- Write in English unless the user asks otherwise. Korean examples get an English translation.
- Be concise: lead with the answer, then 1–2 short paragraphs or a short list. Use **bold** for the Korean headword.
- Today's date is {today}."""

ASK = """
Mode: Ask (the main chat)
How to choose tools
- "What does X mean?" / a Korean word → lookup_word. If the user quotes a sentence, pass it as context_sentence and
  explain the specific numbered sense used there (this is the main reason users come here: dictionaries list many
  senses, and the user needs the one in front of them).
- lookup_word returns found=false → call search_slang (it handles rare and specialized words as well as slang).
- Slang, internet language, abbreviations → search_slang. If word_trend is available, also call it for the same term
  to show whether it's rising or fading; otherwise rely on search_slang's "Still current?" line.
- "How do I say X?" / English input → find_korean_words. Explain how the options differ.
- "Which is more natural, A or B?" → check_naturalness if available (write phrases the way people would type them).
  If it isn't available, call lookup_word on the key words and explain from the dictionary entries and examples,
  saying your judgment of naturalness isn't backed by usage data.
- A pasted Korean passage → mine_vocabulary, then point out the 2–3 most useful words.
- "Break down", "roots", "hanja", "related words" → hanja_family.
- Questions about saved words or progress → get_word_bank. Star/note/delete requests → update_word.
- Lookups save to the word bank automatically; you don't need to mention it unless asked.
You may chain tools in one turn when it helps (for example lookup_word for two easily confused words, then compare)."""

DRILL = """
Mode: Text drill. The learner is practicing by texting. The phone screen shows the partner's messages from tool
results; your text appears in a small coach panel beside the phone, so keep it to 1–3 short sentences in English.
- A message asking to start or for a new drill → call start_text_drill with the mode, persona, source, topic, level,
  and word the message specifies (leave word empty if it says none). Then briefly coach: who they're texting, where
  the word came from (word_source), and, in produce mode, which speech level to use.
- A message asking for a hint → give answer_key.hint_en in your own words. Never reveal target_word or
  model_answer_ko before the learner answers correctly or gives up.
- A message saying they give up / want the answer → call check_drill_answer with gave_up=true.
- Any other message is the learner's answer: call check_drill_answer with reply set to it verbatim. Then coach based on
  the verdict: praise correct answers specifically; for close/miss, say what to fix and invite another try; for
  alternative, acknowledge their word and teach the target word."""

EXPLORE = """
Mode: Explore. The user picked a topic to browse. Call explore_domain with the category, level, and count from the
message (when they ask for more, call it again with the same category and level; it returns a fresh set). The
interface shows the words as cards, so reply with one or two sentences: what this set covers and one word worth
noticing first. If the result says some words are unverified, mention that in a few words. Don't list the words again."""

PROMPTS = {"ask": ASK, "drill": DRILL, "explore": EXPLORE}

NAVER_TOOLS = {"word_trend", "check_naturalness"}


def active_tools(mode: str) -> list[str]:
    """Tools offered to the model in this mode. Naver-backed tools are hidden
    when no Naver keys are configured, so the model never calls a tool that
    can only fail."""
    tools = MODE_TOOLS[mode]
    if not config.features()["naver"]:
        tools = [t for t in tools if t not in NAVER_TOOLS]
    return tools


def system_prompt(mode: str) -> str:
    prompt = COMMON.format(today=date.today().isoformat()) + "\n" + PROMPTS[mode]
    return prompt + "\nTools available right now: " + ", ".join(active_tools(mode)) + ". Only call these."


def _tool_config(mode: str) -> list[types.Tool]:
    decls = [
        types.FunctionDeclaration(
            name=name,
            description=DECLARATIONS[name]["description"],
            parameters_json_schema=DECLARATIONS[name]["parameters"],
        )
        for name in active_tools(mode)
    ]
    return [types.Tool(function_declarations=decls)]


# --- Sessions ----------------------------------------------------------------------

_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
MAX_HISTORY = 60  # Content objects kept per session


def _lock_for(session_id: str) -> threading.Lock:
    with _locks_guard:
        return _session_locks.setdefault(session_id, threading.Lock())


def _load(session_id: str | None, user: str, mode: str) -> tuple[str, dict, list[types.Content]]:
    if session_id:
        data = store.load_session(session_id)
        if data and data.get("user") == user and data.get("mode") == mode:
            history = []
            for raw in data.get("history", []):
                try:
                    history.append(types.Content.model_validate_json(raw))
                except Exception:  # noqa: BLE001
                    history = []
                    break
            return session_id, data, history
    new_id = session_id if session_id and not store.load_session(session_id) else uuid.uuid4().hex
    return new_id, {"user": user, "mode": mode, "history": [], "state": {}}, []


def _trim(history: list[types.Content]) -> list[types.Content]:
    """Keep the tail of the history, starting at a plain user message so that
    function calls and their responses are never split apart."""
    if len(history) <= MAX_HISTORY:
        return history
    tail = history[-MAX_HISTORY:]
    for i, content in enumerate(tail):
        if content.role == "user" and any(p.text for p in (content.parts or [])):
            return tail[i:]
    return tail


def _save(session_id: str, data: dict, history: list[types.Content]) -> None:
    data["history"] = [c.model_dump_json(exclude_none=True) for c in _trim(history)]
    store.save_session(session_id, data)


# --- One turn ------------------------------------------------------------------------

# Tools whose result already IS the reply in a given mode. When the model calls
# only these and they succeed, the turn ends without a second "write a reply"
# Gemini call — the UI renders the result directly. Saves one model round trip.
TERMINAL_TOOLS = {
    "explore": {"explore_domain"},
    "drill": {"start_text_drill", "check_drill_answer"},
}


def _terminal_text(call: dict) -> str:
    r = call["result"]
    if call["name"] == "explore_domain":
        n, unverified = len(r.get("words", [])), r.get("unverified_count", 0)
        if not n:
            return "No new words left for this topic at this level. Try another level or topic."
        text = f"Here are {n} {r.get('level', '')} words about {r.get('category', 'this topic')}."
        if unverified:
            text += f" {unverified} weren't in the learner's dictionary, so they're marked."
        return text
    if call["name"] == "start_text_drill":
        if r.get("mode") == "recognize":
            return f"You're texting {r['partner_name']}, {r['relationship']}. Read the message and explain what it means in English."
        return f"You're texting {r['partner_name']}, {r['relationship']}. Reply in {r['expected_speech_level']}."
    if call["name"] == "check_drill_answer":
        return {
            "correct": "Nice work. Press Next word for another one.",
            "alternative": "That works. The target word is worth learning too, so it's shown below.",
            "close": "Almost. Read the notes below, then try again or tap Hint.",
            "partial": "Partly right. Read the notes below and try again.",
            "miss": "Not quite. Try again, or tap Hint for a nudge.",
            "revealed": "Here's the answer. This word will come back sooner in your reviews.",
        }.get(r.get("verdict", ""), "Checked.")
    return ""


TOOL_STATUS = {
    "lookup_word": lambda a: f"Checking the dictionary for {a.get('word', 'the word')}",
    "find_korean_words": lambda a: f"Finding Korean words for \"{a.get('english', '')}\" and checking each one",
    "search_slang": lambda a: f"Searching the web for {a.get('term', 'that term')}",
    "word_trend": lambda a: "Loading search trends",
    "check_naturalness": lambda a: "Comparing how often each phrasing is used",
    "mine_vocabulary": lambda a: "Scanning the passage for words worth learning",
    "explore_domain": lambda a: f"Finding {a.get('level', 'advanced')} words and checking each in the dictionary",
    "hanja_family": lambda a: f"Breaking {a.get('word', 'the word')} into hanja roots",
    "start_text_drill": lambda a: "Writing the conversation",
    "check_drill_answer": lambda a: "Checking your reply",
    "get_word_bank": lambda a: "Reading your word bank",
    "update_word": lambda a: "Updating your word bank",
}


def _noop(_event: dict) -> None:
    pass


def _call_model(history, cfg, emit) -> types.Content | None:
    """One model call. Streams text to the UI as it's written ({"type": "text"}
    events), then returns the whole message as a single Content for the history.
    Falls back to a normal call if streaming fails before producing text."""
    text_parts: list[str] = []
    calls: list[types.Part] = []
    signature = None
    try:
        for chunk in llm.generate_stream(history, cfg, config.GEMINI_THINKING):
            cand = chunk.candidates[0] if chunk.candidates else None
            for part in (cand.content.parts if cand and cand.content and cand.content.parts else []):
                if part.function_call:
                    calls.append(part)  # function calls arrive whole, with their signature
                elif part.thought:
                    continue
                elif part.text:
                    text_parts.append(part.text)
                    emit({"type": "text", "delta": part.text})
                if part.thought_signature and not part.function_call:
                    signature = part.thought_signature
    except Exception as exc:  # noqa: BLE001
        if text_parts:
            emit({"type": "text_reset"})
        if getattr(exc, "code", None) in (401, 403, 429):
            raise
        log.info("Streaming failed (%s); retrying without streaming", exc.__class__.__name__)
        resp = llm.generate(history, cfg, config.GEMINI_THINKING)
        cand = resp.candidates[0] if resp.candidates else None
        return cand.content if cand else None
    parts = []
    if text_parts:
        parts.append(types.Part(text="".join(text_parts), thought_signature=signature))
    parts.extend(calls)
    return types.Content(role="model", parts=parts) if parts else None


# --- Answer cache -------------------------------------------------------------------
# The first question in a new Ask conversation is cached with the tool calls the
# model chose and the answer it wrote. When anyone asks the same thing again
# (the sample questions, or the README queries graders paste), the tools run
# again (so the word is saved to that user's word bank, and results are fresh)
# but the two model round trips are skipped. Only turns whose tools don't depend
# on who's asking are cached.
CACHEABLE_TOOLS = {"lookup_word", "find_korean_words", "search_slang", "hanja_family",
                   "check_naturalness", "word_trend", "mine_vocabulary"}
ANSWER_TTL = 60 * cache.DAY


def _answer_key(message: str) -> str:
    norm = re.sub(r"\s+", " ", message.strip().lower())
    return cache.make_key("answer", config.GEMINI_MODEL, norm)


def _run_tools(calls: list[tuple[str, dict]], ctx: ToolContext, available: list[str], mode: str, emit) -> list[dict]:
    def run_one(name: str, args: dict) -> dict:
        t1 = time.time()
        if name in available:
            result = execute(name, args, ctx)
        else:
            result = {"ok": False, "error": f"'{name}' isn't available in {mode} mode.",
                      "hint": f"Use one of: {', '.join(available)}."}
        log.info("timing: tool %s %.1fs", name, time.time() - t1)
        emit({"type": "tool", "name": name, "ok": bool(result.get("ok"))})
        return {"name": name, "args": args, "result": result}

    for name, args in calls:
        status = TOOL_STATUS.get(name)
        if status:
            emit({"type": "status", "text": status(args)})
    if len(calls) == 1:
        return [run_one(*calls[0])]
    # Independent tool calls (e.g. looking up two words) run in parallel. Each task
    # gets a copy of this thread's context so priority flags carry over.
    parent = contextvars.copy_context()
    return list(_tool_pool.map(lambda c: parent.copy().run(run_one, *c), calls))


def _replay(message: str, ctx: ToolContext, available: list[str], emit) -> dict | None:
    record = cache.get(_answer_key(message))
    if not record:
        return None
    done = _run_tools([(c["name"], c["args"]) for c in record["calls"]], ctx, available, "ask", emit)
    if not all(d["result"].get("ok") for d in done):
        return None  # something changed (e.g. a dictionary error): answer normally instead
    emit({"type": "text", "delta": record["response"]})
    return {"response": record["response"], "tool_calls": done}


def run_turn(message: str, session_id: str | None, user: str, mode: str = "ask", on_event=None,
             action: dict | None = None) -> dict:
    """One /chat turn. `on_event` (optional) receives progress events for streaming:
    {"type": "status", "text": ...} and {"type": "tool", "name": ..., "ok": ...}.

    `action` ({"tool": ..., "args": {...}}) is for interface buttons that map to
    exactly one tool: starting a drill, grading a drill reply, browsing a topic.
    Those run the tool directly instead of asking the model which tool to use,
    which saves a Gemini round trip and involves no judgment. Typed questions
    always go through the model."""
    emit = on_event or _noop
    mode = mode if mode in MODE_TOOLS else "ask"
    session_id, data, history = _load(session_id, user, mode)
    turn_t0 = time.time()
    with _lock_for(session_id):
        ctx = ToolContext(user=user, session=data.setdefault("state", {}), session_id=session_id, emit=emit)
        turn_start = len(history)
        history.append(types.Content(role="user", parts=[types.Part(text=message)]))

        if action:
            return _run_action(action, mode, session_id, data, history, turn_start, ctx, emit, turn_t0)
        if mode == "ask" and turn_start == 0:
            replayed = _replay(message, ctx, active_tools(mode), emit)
            if replayed:
                history.append(types.Content(role="model", parts=[types.Part(text=replayed["response"])]))
                _save(session_id, data, history)
                log.info("timing: turn (ask, cached answer) %.1fs", time.time() - turn_t0)
                return {"response": replayed["response"], "session_id": session_id,
                        "tool_calls": _json_safe(replayed["tool_calls"])}
        tool_calls: list[dict] = []
        available = active_tools(mode)
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt(mode),
            tools=_tool_config(mode),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text = ""
        try:
            for step in range(config.MAX_AGENT_STEPS):
                emit({"type": "status", "text": "Reading your message" if step == 0 else "Writing the answer"})
                t0 = time.time()
                content = _call_model(history, cfg, emit)
                log.info("timing: gemini step %d %.1fs", step + 1, time.time() - t0)
                if content is None or not content.parts:
                    text = "I couldn't produce a response to that. Could you rephrase it?"
                    break
                history.append(content)
                calls = [p.function_call for p in content.parts if p.function_call]
                if not calls:
                    text = "".join(p.text for p in content.parts if p.text and not p.thought).strip()
                    break

                done = _run_tools([(call.name, dict(call.args or {})) for call in calls], ctx, available, mode, emit)
                tool_calls.extend(done)
                history.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response={"result": d["result"]}))
                    for call, d in zip(calls, done)]))

                terminal = TERMINAL_TOOLS.get(mode, set())
                if terminal and all(d["name"] in terminal and d["result"].get("ok") for d in done):
                    text = " ".join(t for t in (_terminal_text(d) for d in done) if t)
                    history.append(types.Content(role="model", parts=[types.Part(text=text)]))
                    break
            else:
                text = "I ran out of steps before finishing. Try asking a narrower question."
        except Exception as exc:  # noqa: BLE001
            log.exception("Agent turn failed")
            del history[turn_start:]  # never persist a half-finished turn
            return {"response": llm.describe_api_error(exc), "session_id": session_id, "tool_calls": tool_calls}

        _save(session_id, data, history)
        log.info("timing: turn (%s) %.1fs, %d tool call(s)", mode, time.time() - turn_t0, len(tool_calls))
        if not text:
            text = "Here's what I found." if tool_calls else "Could you say a bit more about what you're looking for?"
        elif (mode == "ask" and turn_start == 0 and tool_calls and len(tool_calls) <= 4
              and all(c["name"] in CACHEABLE_TOOLS and c["result"].get("ok") for c in tool_calls)):
            cache.put(_answer_key(message), {
                "response": text, "calls": [{"name": c["name"], "args": c["args"]} for c in tool_calls],
            }, ANSWER_TTL)
        return {"response": text, "session_id": session_id, "tool_calls": _json_safe(tool_calls)}


DIRECT_ACTIONS = {
    "drill": {"start_text_drill", "check_drill_answer"},
    "explore": {"explore_domain"},
}


def _run_action(action, mode, session_id, data, history, turn_start, ctx, emit, turn_t0) -> dict:
    name = str(action.get("tool", ""))
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if name not in DIRECT_ACTIONS.get(mode, set()):
        del history[turn_start:]
        return {"response": f"'{name}' can't be run directly in {mode} mode.", "session_id": session_id,
                "tool_calls": []}
    done = _run_tools([(name, args)], ctx, active_tools(mode), mode, emit)
    result = done[0]["result"]
    if result.get("ok"):
        text = _terminal_text(done[0])
    else:
        text = result.get("error", "That didn't work.")
        if "rate limit" in text.lower() or "busy" in text.lower():
            text = "Gemini is busy right now. Please try again in a few seconds."
    # The conversation history records what happened in plain text, so the
    # session stays coherent if the next message goes through the model.
    history.append(types.Content(role="model", parts=[types.Part(text=text)]))
    _save(session_id, data, history)
    log.info("timing: turn (%s, direct %s) %.1fs", mode, name, time.time() - turn_t0)
    return {"response": text, "session_id": session_id, "tool_calls": _json_safe(done)}


def reveal_drill_answer(session_id: str, user: str) -> dict:
    """The drill's "Show answer" button: no model call needed, so this runs the
    grading tool directly (gave_up=True) to record the miss and return the key."""
    data = store.load_session(session_id) if session_id else None
    if not data or data.get("user") != user:
        return {"ok": False, "error": "That drill session wasn't found.", "hint": "Start a new drill."}
    with _lock_for(session_id):
        ctx = ToolContext(user=user, session=data.setdefault("state", {}), session_id=session_id)
        result = execute("check_drill_answer", {"gave_up": True}, ctx)
        store.save_session(session_id, data)
        return result


def _json_safe(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
