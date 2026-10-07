"""Cache for expensive results that don't depend on who's asking.

Examples: the vocabulary pool for a topic, the hanja breakdown of 경제, the
web-grounded explanation of a slang term. Two levels:

  memory    → instant, per server instance
  Firestore → survives restarts and is shared by every Cloud Run instance
              (collection "cache"; skipped automatically if Firestore is off)

Personal data (word banks) never goes here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from .storage import store

log = logging.getLogger("malmoi.cache")

HOUR = 3600
DAY = 24 * HOUR

_mem: OrderedDict[str, tuple[float, Any]] = OrderedDict()
_mem_lock = threading.Lock()
_build_locks: dict[str, threading.Lock] = {}
_build_guard = threading.Lock()
MEM_MAX = 3000


def make_key(namespace: str, *parts: Any) -> str:
    raw = json.dumps([namespace, *parts], ensure_ascii=False, sort_keys=True)
    return namespace + ":" + hashlib.sha1(raw.encode()).hexdigest()[:24]


def get(key: str) -> Any | None:
    now = time.time()
    with _mem_lock:
        hit = _mem.get(key)
        if hit and hit[0] > now:
            _mem.move_to_end(key)
            return hit[1]
    if store.mode == "firestore":
        try:
            snap = store._db.collection("cache").document(key).get(timeout=4)  # noqa: SLF001
            if snap.exists:
                doc = snap.to_dict()
                if doc.get("expires", 0) > now:
                    value = json.loads(doc["value"])
                    _remember(key, value, doc["expires"])
                    return value
        except Exception as exc:  # noqa: BLE001
            log.info("cache read failed for %s: %s", key, exc)
    return None


def put(key: str, value: Any, ttl: float) -> None:
    expires = time.time() + ttl
    _remember(key, value, expires)
    if store.mode == "firestore":
        try:
            store._db.collection("cache").document(key).set(  # noqa: SLF001
                {"value": json.dumps(value, ensure_ascii=False), "expires": expires}, timeout=6)
        except Exception as exc:  # noqa: BLE001
            log.info("cache write failed for %s: %s", key, exc)


def _remember(key: str, value: Any, expires: float) -> None:
    with _mem_lock:
        _mem[key] = (expires, value)
        _mem.move_to_end(key)
        while len(_mem) > MEM_MAX:
            _mem.popitem(last=False)


def build_lock(key: str) -> threading.Lock:
    """One builder per key: a second request for the same thing waits for the
    first to finish and then reads the cache, instead of doing the work twice."""
    with _build_guard:
        return _build_locks.setdefault(key, threading.Lock())


def cached(key: str, ttl: float, fn: Callable[[], Any]) -> Any:
    value = get(key)
    if value is not None:
        return value
    with build_lock(key):
        value = get(key)
        if value is not None:
            return value
        value = fn()
        if value is not None:
            put(key, value, ttl)
        return value


def clear_memory() -> None:
    with _mem_lock:
        _mem.clear()
