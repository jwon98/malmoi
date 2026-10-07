"""Central configuration. Every setting comes from an environment variable.

Locally, values are read from a `.env` file (see `.env.example`).
On Cloud Run, set them under "Variables & Secrets" (see README → Deploy).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("malmoi")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else default


# --- Gemini (Vertex AI / Agent Platform) --------------------------------------
# Match the model string used in the course starter if it differs.
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.8-flash")
GOOGLE_CLOUD_LOCATION = _env("GOOGLE_CLOUD_LOCATION", "global")
# How much the model "thinks" before answering. Lower = faster. Gemini 3 accepts
# minimal | low | medium | high; "off" sends no setting (use for older models).
GEMINI_THINKING = _env("GEMINI_THINKING", "low").lower()            # the agent loop
GEMINI_TOOL_THINKING = _env("GEMINI_TOOL_THINKING", "minimal").lower()  # JSON calls inside tools
# Max Gemini requests in flight at once per server instance (helps avoid 429s).
GEMINI_MAX_CONCURRENT = int(_env("GEMINI_MAX_CONCURRENT", "6"))

# --- National Institute of Korean Language (국립국어원) dictionaries -----------
KRDICT_API_KEY = _env("KRDICT_API_KEY")      # 한국어기초사전 (learner's dictionary, has English)
STDICT_API_KEY = _env("STDICT_API_KEY")      # 표준국어대사전 (standard dictionary)
OPENDICT_API_KEY = _env("OPENDICT_API_KEY")  # 우리말샘 (open dictionary, newer words)
# Some Korean government sites occasionally serve incomplete certificate chains.
# Only set this to "false" if lookups fail with SSL errors.
NIKL_SSL_VERIFY = _env("NIKL_SSL_VERIFY", "true").lower() != "false"

# --- Naver (search trend + search counts) via NAVER API HUB ----------------------
NAVER_CLIENT_ID = _env("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = _env("NAVER_CLIENT_SECRET")
NAVER_API = _env("NAVER_API", "hub").lower()  # "hub" (default) or "legacy" (old Developers Center keys)

# --- Storage -------------------------------------------------------------------
# "auto": use Firestore if reachable, otherwise fall back to in-memory storage.
USE_FIRESTORE = _env("USE_FIRESTORE", "auto").lower()
FIRESTORE_DATABASE = _env("FIRESTORE_DATABASE", "(default)")

# --- Text-to-speech ------------------------------------------------------------
TTS_VOICE = _env("TTS_VOICE", "ko-KR-Neural2-A")

# --- Misc ----------------------------------------------------------------------
LOCAL_USER = _env("LOCAL_USER", "local-dev")
MAX_AGENT_STEPS = int(_env("MAX_AGENT_STEPS", "8"))


@lru_cache(maxsize=1)
def project_id() -> str | None:
    """GCP project ID from the environment, or from Application Default Credentials."""
    for name in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        if _env(name):
            return _env(name)
    try:
        import google.auth

        _, project = google.auth.default()
        return project
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not determine GCP project: %s", exc)
        return None


def features() -> dict[str, bool]:
    """Which optional integrations are configured (shown in the UI)."""
    return {
        "krdict": bool(KRDICT_API_KEY),
        "stdict": bool(STDICT_API_KEY),
        "opendict": bool(OPENDICT_API_KEY),
        "naver": bool(NAVER_CLIENT_ID and NAVER_CLIENT_SECRET),
    }
