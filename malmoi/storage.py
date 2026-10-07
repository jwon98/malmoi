"""Persistence for word banks and chat sessions.

Firestore layout (database: FIRESTORE_DATABASE, default "(default)"):
    users/{user_email}/words/{word}   one document per saved word
    sessions/{session_id}             chat history for /chat sessions

If Firestore isn't reachable (e.g. running locally without it), everything is
kept in memory instead and the UI says so. Nothing else in the app changes.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from . import config

log = logging.getLogger("malmoi.storage")

# Spaced repetition: mastery 0–5, days until the next review at each level.
REVIEW_INTERVAL_DAYS = [0, 1, 2, 4, 8, 16]
MASTERED_AT = 4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _doc_id(word: str) -> str:
    return word.strip().replace("/", "∕")[:300] or "_"


class Store:
    def __init__(self) -> None:
        self._db = None
        self._mode: str | None = None
        self.init_error = ""
        self._init_lock = threading.Lock()
        self._mem_words: dict[str, dict[str, dict]] = {}
        self._mem_sessions: dict[str, dict] = {}
        self._mem_lock = threading.Lock()

    # --- setup ---------------------------------------------------------------
    @property
    def mode(self) -> str:
        if self._mode is None:
            with self._init_lock:
                if self._mode is None:
                    self._connect()
        return self._mode  # type: ignore[return-value]

    def _connect(self) -> None:
        if config.USE_FIRESTORE == "false":
            self._mode = "memory"
            self.init_error = "Firestore disabled with USE_FIRESTORE=false"
            return
        try:
            from google.cloud import firestore

            db = firestore.Client(project=config.project_id(), database=config.FIRESTORE_DATABASE)
            db.collection("_malmoi").document("healthcheck").set({"at": now_iso()}, timeout=6)
            self._db = db
            self._mode = "firestore"
            log.info("Storage: Firestore (%s)", config.FIRESTORE_DATABASE)
        except Exception as exc:  # noqa: BLE001
            self._mode = "memory"
            self.init_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
            log.warning("Storage: Firestore unavailable, using memory. %s", self.init_error)

    def _words(self, user: str):
        return self._db.collection("users").document(user).collection("words")  # type: ignore[union-attr]

    # --- words ---------------------------------------------------------------
    def get_word(self, user: str, word: str) -> dict | None:
        if self.mode == "firestore":
            snap = self._words(user).document(_doc_id(word)).get()
            return snap.to_dict() if snap.exists else None
        with self._mem_lock:
            item = self._mem_words.get(user, {}).get(_doc_id(word))
            return dict(item) if item else None

    def put_word(self, user: str, word: str, data: dict) -> dict:
        data = {**data, "word": word, "updated_at": now_iso()}
        if self.mode == "firestore":
            self._words(user).document(_doc_id(word)).set(data, merge=True)
            return self.get_word(user, word) or data
        with self._mem_lock:
            bucket = self._mem_words.setdefault(user, {})
            merged = {**bucket.get(_doc_id(word), {}), **data}
            bucket[_doc_id(word)] = merged
            return dict(merged)

    def delete_word(self, user: str, word: str) -> bool:
        if self.mode == "firestore":
            ref = self._words(user).document(_doc_id(word))
            existed = ref.get().exists
            ref.delete()
            return existed
        with self._mem_lock:
            return self._mem_words.get(user, {}).pop(_doc_id(word), None) is not None

    def all_words(self, user: str, cap: int = 1000) -> list[dict]:
        if self.mode == "firestore":
            return [d.to_dict() for d in self._words(user).limit(cap).stream()]
        with self._mem_lock:
            return [dict(v) for v in list(self._mem_words.get(user, {}).values())[:cap]]

    def list_words(self, user: str, view: str = "recent", limit: int = 50, query: str = "") -> list[dict]:
        words = self.all_words(user)
        now = now_iso()
        q = query.strip().lower()
        if q:
            words = [w for w in words if q in w.get("word", "").lower() or q in (w.get("gloss") or "").lower()
                     or q in (w.get("romanization") or "").lower()]
        if view == "due":
            words = [w for w in words if w.get("mastery", 0) < 5 and (w.get("next_review") or "") <= now]
            words.sort(key=lambda w: (w.get("next_review") or "", w.get("mastery", 0)))
        elif view == "starred":
            words = [w for w in words if w.get("starred")]
            words.sort(key=lambda w: w.get("updated_at", ""), reverse=True)
        elif view == "struggling":
            words = [w for w in words if w.get("reviews", 0) > 0 and w.get("mastery", 0) <= 1]
            words.sort(key=lambda w: w.get("updated_at", ""), reverse=True)
        elif view == "mastered":
            words = [w for w in words if w.get("mastery", 0) >= MASTERED_AT]
            words.sort(key=lambda w: w.get("updated_at", ""), reverse=True)
        elif view == "alphabetical":
            words.sort(key=lambda w: w.get("word", ""))
        else:  # recent
            words.sort(key=lambda w: w.get("updated_at", ""), reverse=True)
        return words[:limit]

    def stats(self, user: str) -> dict:
        words = self.all_words(user)
        now = now_iso()
        return {
            "total": len(words),
            "due": sum(1 for w in words if w.get("mastery", 0) < 5 and (w.get("next_review") or "") <= now),
            "mastered": sum(1 for w in words if w.get("mastery", 0) >= MASTERED_AT),
            "starred": sum(1 for w in words if w.get("starred")),
        }

    def record_lookup(self, user: str, entry: dict, context: str = "", source_override: str = "") -> dict:
        """Add or refresh a word after it was looked up."""
        word = entry["word"]
        existing = self.get_word(user, word) or {}
        contexts = list(existing.get("contexts", []))
        if context and context not in contexts:
            contexts = ([context] + contexts)[:5]
        first_sense = (entry.get("senses") or [{}])[0]
        data = {
            "romanization": entry.get("romanization", ""),
            "gloss": entry.get("gloss") or first_sense.get("english_word") or existing.get("gloss", ""),
            "definition": first_sense.get("english_definition") or first_sense.get("definition_ko") or existing.get("definition", ""),
            "definition_ko": first_sense.get("definition_ko", "") or existing.get("definition_ko", ""),
            "level": entry.get("level_en") or existing.get("level", ""),
            "origin": entry.get("origin", "") or existing.get("origin", ""),
            "pos": entry.get("pos", "") or existing.get("pos", ""),
            "source": source_override or entry.get("source", "") or existing.get("source", ""),
            "example": (entry.get("examples") or [existing.get("example", "")])[0],
            "contexts": contexts,
            "lookups": existing.get("lookups", 0) + 1,
        }
        if not existing:
            data.update({"created_at": now_iso(), "mastery": 0, "reviews": 0, "next_review": now_iso(),
                         "starred": False, "note": ""})
        return self.put_word(user, word, data)

    def record_review(self, user: str, word: str, outcome: str) -> dict:
        """outcome: 'correct' | 'partial' | 'miss'. Returns the updated word."""
        existing = self.get_word(user, word) or {"mastery": 0, "reviews": 0}
        mastery = int(existing.get("mastery", 0))
        if outcome == "correct":
            mastery = min(5, mastery + 1)
        elif outcome == "miss":
            mastery = max(0, mastery - 2)
        days = REVIEW_INTERVAL_DAYS[mastery] if outcome != "partial" else 1
        next_review = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")
        return self.put_word(user, word, {
            "mastery": mastery,
            "reviews": int(existing.get("reviews", 0)) + 1,
            "last_result": outcome,
            "next_review": next_review,
        })

    # --- sessions ------------------------------------------------------------
    def load_session(self, session_id: str) -> dict | None:
        with self._mem_lock:
            if session_id in self._mem_sessions:
                return self._mem_sessions[session_id]
        if self.mode == "firestore":
            try:
                snap = self._db.collection("sessions").document(session_id).get(timeout=6)  # type: ignore[union-attr]
                if snap.exists:
                    data = snap.to_dict()
                    with self._mem_lock:
                        self._mem_sessions[session_id] = data
                    return data
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load session %s: %s", session_id, exc)
        return None

    def save_session(self, session_id: str, data: dict) -> None:
        data["updated_at"] = now_iso()
        with self._mem_lock:
            self._mem_sessions[session_id] = data
            self._evict_old_sessions()
        if self.mode == "firestore":
            try:
                self._db.collection("sessions").document(session_id).set(data, timeout=6)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not persist session %s: %s", session_id, exc)

    def _evict_old_sessions(self, keep: int = 500) -> None:
        if len(self._mem_sessions) <= keep:
            return
        oldest = sorted(self._mem_sessions.items(), key=lambda kv: kv[1].get("updated_at", ""))
        for sid, _ in oldest[: len(self._mem_sessions) - keep]:
            self._mem_sessions.pop(sid, None)


store = Store()


def warm_up() -> None:
    """Connect to storage in the background at startup so the first request is fast."""
    def run():
        t = time.time()
        _ = store.mode
        log.info("Storage ready (%s) in %.1fs", store.mode, time.time() - t)

    threading.Thread(target=run, daemon=True).start()
