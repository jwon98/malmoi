"""Malmoi (말모이) — a Korean vocabulary agent for advanced learners.

Run locally:   uv run app.py            → http://localhost:8080
Cloud Run:     uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import genanki
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from malmoi import config, dictionaries, korean
from malmoi.agent import reveal_drill_answer, run_turn
from malmoi.storage import store, warm_up
from malmoi.tools import LEVEL_GUIDE, ToolContext, hanja_family, lookup_word, topic_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("malmoi")

BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warm_up()
    # Load the Korean morphological analyzer in the background (~1–2 s, ~0.5 GB).
    threading.Thread(target=korean.kiwi, daemon=True).start()
    yield


app = FastAPI(title="Malmoi", description="Korean vocabulary agent for advanced learners", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def current_user(request: Request) -> str:
    """Identity-Aware Proxy adds the signed-in Google account to every request.
    Locally there's no IAP, so everyone is LOCAL_USER."""
    header = request.headers.get("x-goog-authenticated-user-email", "")
    email = header.split(":", 1)[-1].strip().lower()
    return email or config.LOCAL_USER


SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


# --- Pages -----------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}


# --- Chat (same response shape as the course starter) ------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    session_id: str | None = None
    mode: str = "ask"  # ask | drill | explore


@app.post("/chat")
def chat(body: ChatRequest, request: Request) -> dict:
    session_id = body.session_id if body.session_id and SESSION_RE.match(body.session_id) else None
    result = run_turn(body.message.strip(), session_id, current_user(request), body.mode)
    # Exactly the starter's shape: response, session_id, tool_calls[{name, args, result}]
    return {"response": result["response"], "session_id": result["session_id"], "tool_calls": result["tool_calls"]}


@app.post("/chat/stream")
def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Same turn as /chat, streamed as Server-Sent Events so the UI can show live
    progress ("Checking the dictionary for 눈치…"). The final `done` event carries
    exactly the /chat payload: response, session_id, tool_calls."""
    session_id = body.session_id if body.session_id and SESSION_RE.match(body.session_id) else None
    user = current_user(request)
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = run_turn(body.message.strip(), session_id, user, body.mode, on_event=events.put)
            events.put({"type": "done", "response": result["response"], "session_id": result["session_id"],
                        "tool_calls": result["tool_calls"]})
        except Exception as exc:  # noqa: BLE001
            log.exception("Streamed turn failed")
            events.put({"type": "error", "detail": f"Server error ({exc.__class__.__name__})."})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class RevealRequest(BaseModel):
    session_id: str


@app.post("/api/drill/reveal")
def drill_reveal(body: RevealRequest, request: Request) -> dict:
    """"Show answer" in the drill: records the miss and returns the answer key
    without a model call, so it's instant."""
    if not SESSION_RE.match(body.session_id):
        raise HTTPException(400, "Invalid session.")
    result = reveal_drill_answer(body.session_id, current_user(request))
    if not result.get("ok"):
        raise HTTPException(404, result.get("error", "No drill in progress."))
    return result


class WarmRequest(BaseModel):
    level: str = "advanced"
    topics: list[str] = Field(default_factory=list, max_length=16)


_warming: set[str] = set()
_warm_lock = threading.Lock()
_warm_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="warm")


@app.post("/api/warm")
def warm(body: WarmRequest) -> dict:
    """Pre-build the Explore topic pools (and a few sample answers) so clicking a
    topic is instant. The browser calls this when the site opens. Results are
    cached in Firestore for 14 days, so after the first time this returns quickly.
    The request stays open while it works, which keeps Cloud Run's CPU allocated."""
    level = body.level if body.level in LEVEL_GUIDE else "advanced"
    topics = [t.strip()[:80] for t in body.topics if t.strip()]
    with _warm_lock:
        if level in _warming:
            return {"status": "already warming", "level": level}
        _warming.add(level)
    t0 = time.time()
    try:
        jobs = [_warm_pool.submit(topic_pool, t, level, None, True) for t in topics]
        if level == "advanced":  # the sample queries on the Ask tab
            jobs.append(_warm_pool.submit(hanja_family, ToolContext(user="warmup"), "경제"))
            jobs.append(_warm_pool.submit(dictionaries.lookup, "눈치"))
        ready = 0
        for job in jobs:
            try:
                job.result(timeout=240)
                ready += 1
            except Exception as exc:  # noqa: BLE001
                log.info("warm-up item failed: %s", exc)
        log.info("timing: warm-up (%s) %.1fs, %d/%d ready", level, time.time() - t0, ready, len(jobs))
        return {"status": "ok", "level": level, "ready": ready, "total": len(jobs), "seconds": round(time.time() - t0, 1)}
    finally:
        with _warm_lock:
            _warming.discard(level)


# --- Status ----------------------------------------------------------------------------------

@app.get("/api/me")
def me(request: Request) -> dict:
    user = current_user(request)
    return {
        "user": user,
        "storage": store.mode,
        "storage_note": store.init_error if store.mode == "memory" else "",
        "features": config.features(),
        "model": config.GEMINI_MODEL,
        "stats": store.stats(user),
    }


# --- Word bank -------------------------------------------------------------------------------

@app.get("/api/words")
def list_words(request: Request, view: str = "recent", q: str = "", limit: int = 200) -> dict:
    user = current_user(request)
    return {"words": store.list_words(user, view, max(1, min(limit, 500)), q), "stats": store.stats(user)}


class AddWord(BaseModel):
    word: str = Field(..., min_length=1, max_length=40)
    context: str = ""
    gloss: str = Field("", max_length=200)      # shown on the card the user saved from
    example: str = Field("", max_length=300)


@app.post("/api/words")
def add_word(body: AddWord, request: Request) -> dict:
    """Save a word from a card (Explore, vocabulary scan). Runs the same lookup tool;
    if no dictionary has the word, it's saved with the card's English and marked unverified."""
    user = current_user(request)
    result = lookup_word(ToolContext(user=user), body.word, body.context)
    if result.get("ok") and result.get("found", True):
        return {"word": store.get_word(user, result["word"])}
    if not korean.has_hangul(body.word):
        raise HTTPException(400, result.get("error") or "Only Korean words can be saved.")
    word = body.word.strip()
    store.record_lookup(user, {"word": word, "romanization": korean.romanize(word), "gloss": body.gloss,
                               "senses": [{"english_word": body.gloss}], "level_en": "",
                               "examples": [body.example] if body.example else []},
                        body.context, source_override="ai")
    return {"word": store.get_word(user, word), "verified": False}


class PatchWord(BaseModel):
    starred: bool | None = None
    note: str | None = Field(None, max_length=500)
    mastery: int | None = Field(None, ge=0, le=5)


@app.patch("/api/words/{word}")
def patch_word(word: str, body: PatchWord, request: Request) -> dict:
    user = current_user(request)
    if not store.get_word(user, word):
        raise HTTPException(404, f"'{word}' isn't in your word bank.")
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    return {"word": store.put_word(user, word, changes)}


@app.delete("/api/words/{word}")
def delete_word(word: str, request: Request) -> dict:
    if not store.delete_word(current_user(request), word):
        raise HTTPException(404, f"'{word}' isn't in your word bank.")
    return {"deleted": word}


# --- Exports -----------------------------------------------------------------------------------

ANKI_MODEL = genanki.Model(
    1607392319,
    "Malmoi word",
    fields=[{"name": "Korean"}, {"name": "Romanization"}, {"name": "Meaning"}, {"name": "Definition"},
            {"name": "Example"}, {"name": "SeenIn"}, {"name": "Note"}],
    templates=[{
        "name": "Recognize",
        "qfmt": '<div class="ko">{{Korean}}</div><div class="rom">{{Romanization}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="meaning">{{Meaning}}</div><div class="def">{{Definition}}</div>'
                '{{#Example}}<div class="ex">{{Example}}</div>{{/Example}}'
                '{{#SeenIn}}<div class="seen">Seen in: {{SeenIn}}</div>{{/SeenIn}}'
                '{{#Note}}<div class="note">{{Note}}</div>{{/Note}}',
    }],
    css=".card{font-family:'Apple SD Gothic Neo','Noto Sans KR',sans-serif;text-align:center;color:#203029}"
        ".ko{font-size:42px;margin-top:12px}.rom{color:#6b7a73;font-size:16px}.meaning{font-size:22px;margin:10px 0}"
        ".def,.ex,.seen,.note{font-size:15px;color:#46554f;margin:6px auto;max-width:520px}",
)


@app.get("/api/export.apkg")
def export_anki(request: Request) -> Response:
    user = current_user(request)
    words = store.list_words(user, "alphabetical", 5000)
    if not words:
        raise HTTPException(404, "Your word bank is empty. Look up a few words first.")
    deck = genanki.Deck(int(hashlib.sha1(user.encode()).hexdigest()[:8], 16), "Malmoi word bank")
    for w in words:
        deck.add_note(genanki.Note(model=ANKI_MODEL, guid=genanki.guid_for(user, w["word"]), fields=[
            w["word"], w.get("romanization", ""), w.get("gloss", ""), w.get("definition", ""),
            w.get("example", ""), (w.get("contexts") or [""])[0], w.get("note", ""),
        ]))
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        path = tmp.name
    try:
        genanki.Package(deck).write_to_file(path)
        data = Path(path).read_bytes()
    finally:
        os.unlink(path)
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": 'attachment; filename="malmoi-word-bank.apkg"'})


@app.get("/api/export.csv")
def export_csv(request: Request) -> Response:
    words = store.list_words(current_user(request), "alphabetical", 5000)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["korean", "romanization", "meaning", "definition", "level", "mastery", "example", "seen_in", "note"])
    for w in words:
        writer.writerow([w["word"], w.get("romanization", ""), w.get("gloss", ""), w.get("definition", ""),
                         w.get("level", ""), w.get("mastery", 0), w.get("example", ""),
                         (w.get("contexts") or [""])[0], w.get("note", "")])
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="malmoi-word-bank.csv"'})


# --- Romanization (used by the UI for every Korean word on screen) --------------------------------

class RomanizeRequest(BaseModel):
    items: list[str] = Field(default_factory=list, max_length=400)


@app.post("/api/romanize")
def romanize(body: RomanizeRequest) -> dict:
    out = {}
    for item in body.items:
        item = item.strip()[:40]
        if item and korean.has_hangul(item):
            out[item] = korean.romanize(item)
    return {"romanized": out}


# --- Text-to-speech (Google Cloud TTS; the UI falls back to the browser's voice) ---------------

_tts_cache: OrderedDict[str, bytes] = OrderedDict()
_tts_lock = threading.Lock()
_creds = None
_tts_disabled_until = 0.0


def _access_token() -> str:
    global _creds
    import google.auth
    from google.auth.transport.requests import Request as AuthRequest

    if _creds is None:
        _creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not _creds.valid:
        _creds.refresh(AuthRequest())
    return _creds.token


@app.get("/api/tts")
def tts(text: str) -> Response:
    global _tts_disabled_until
    text = text.strip()[:300]
    if not korean.has_hangul(text):
        raise HTTPException(400, "Text-to-speech is for Korean text.")
    with _tts_lock:
        if text in _tts_cache:
            _tts_cache.move_to_end(text)
            return Response(_tts_cache[text], media_type="audio/mpeg", headers={"Cache-Control": "max-age=86400"})
    if time.time() < _tts_disabled_until:
        raise HTTPException(503, "Cloud Text-to-Speech is unavailable; use the browser voice.")
    try:
        headers = {"Authorization": f"Bearer {_access_token()}"}
        if config.project_id():
            headers["x-goog-user-project"] = config.project_id()
        resp = httpx.post(
            "https://texttospeech.googleapis.com/v1/text:synthesize",
            headers=headers, timeout=15,
            json={"input": {"text": text},
                  "voice": {"languageCode": "ko-KR", "name": config.TTS_VOICE},
                  "audioConfig": {"audioEncoding": "MP3", "speakingRate": 0.95}},
        )
        resp.raise_for_status()
        audio = base64.b64decode(resp.json()["audioContent"])
    except Exception as exc:  # noqa: BLE001
        log.warning("TTS failed (%s); disabling for 10 minutes", exc)
        _tts_disabled_until = time.time() + 600
        raise HTTPException(503, "Cloud Text-to-Speech is unavailable; use the browser voice.") from exc
    with _tts_lock:
        _tts_cache[text] = audio
        if len(_tts_cache) > 500:
            _tts_cache.popitem(last=False)
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "max-age=86400"})


@app.exception_handler(Exception)
async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error")
    return JSONResponse({"detail": f"Server error ({exc.__class__.__name__})."}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=int(os.environ.get("PORT", 8080)), reload=False)
