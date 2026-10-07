"""Deterministic Korean-language helpers (no LLM involved).

- romanize(): Revised Romanization that follows *pronunciation*, not spelling.
  국물 → gungmul (not gukmul), 신라 → silla, 같이 → gachi.
- analyze()/lemmas(): morphological analysis with Kiwi, so conjugated forms
  (망설여져서, 걱정했어) can be matched to dictionary forms (망설이다, 걱정하다).
- detect_speech_level(): casual (반말) vs polite (해요체) vs formal (합쇼체).
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

from korean_romanizer.romanizer import Romanizer

HANGUL_RUN = re.compile(r"[가-힣]+")
HANJA_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# Jamo tables used by Unicode Hangul syllable composition.
CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNG_I = 20  # index of ㅣ

K_CLASS = {"ㄱ", "ㄲ", "ㅋ", "ㄳ", "ㄺ"}
T_CLASS = {"ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅎ"}
P_CLASS = {"ㅂ", "ㅍ", "ㄼ", "ㄿ", "ㅄ"}


def has_hangul(text: str) -> bool:
    return bool(HANGUL_RUN.search(text or ""))


def _decompose(syllable: str) -> list:
    code = ord(syllable) - 0xAC00
    return [code // 588, (code % 588) // 28, code % 28]


def _compose(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + cho * 588 + jung * 28 + jong)


def approximate_pronunciation(word: str) -> str:
    """Apply the most common Korean sound-change rules inside one Hangul run.

    Covers palatalization (같이 → 가치), nasalization (국물 → 궁물),
    liquidization (신라 → 실라, 설날 → 설랄), ㄹ → ㄴ after ㅁ/ㅇ (심리 → 심니),
    and ㅎ-dropping before a vowel (좋아 → 조아). It is an approximation:
    dictionary pronunciations are used instead whenever they are available.
    """
    sylls = [_decompose(ch) for ch in word if "가" <= ch <= "힣"]
    if len(sylls) != len(word):
        return word
    for i in range(len(sylls) - 1):
        cur, nxt = sylls[i], sylls[i + 1]
        j = JONG[cur[2]]
        c = CHO[nxt[0]]
        if not j:
            continue
        # Palatalization: ㄷ/ㅌ + 이 → 지/치, ㄷ + 히 → 치
        if nxt[1] == JUNG_I and c == "ㅇ" and j in ("ㄷ", "ㅌ"):
            cur[2] = 0
            nxt[0] = CHO.index("ㅈ" if j == "ㄷ" else "ㅊ")
            continue
        if nxt[1] == JUNG_I and c == "ㅎ" and j == "ㄷ":
            cur[2] = 0
            nxt[0] = CHO.index("ㅊ")
            continue
        # ㅎ disappears before a vowel: 좋아 → 조아, 많이 → 마니
        if c == "ㅇ" and j in ("ㅎ", "ㄶ", "ㅀ"):
            cur[2] = {"ㅎ": 0, "ㄶ": JONG.index("ㄴ"), "ㅀ": JONG.index("ㄹ")}[j]
            continue
        # Nasalization before ㄴ/ㅁ
        if c in ("ㄴ", "ㅁ"):
            if j in K_CLASS:
                cur[2] = JONG.index("ㅇ")
            elif j in T_CLASS:
                cur[2] = JONG.index("ㄴ")
            elif j in P_CLASS:
                cur[2] = JONG.index("ㅁ")
            elif j == "ㄹ" and c == "ㄴ":  # 설날 → 설랄
                nxt[0] = CHO.index("ㄹ")
            continue
        # Rules triggered by a following ㄹ
        if c == "ㄹ":
            if j == "ㄴ":  # 신라 → 실라
                cur[2] = JONG.index("ㄹ")
            elif j in ("ㅁ", "ㅇ"):  # 심리 → 심니
                nxt[0] = CHO.index("ㄴ")
            elif j in K_CLASS:  # 국립 → 궁닙
                cur[2] = JONG.index("ㅇ")
                nxt[0] = CHO.index("ㄴ")
            elif j in P_CLASS:  # 협력 → 혐녁
                cur[2] = JONG.index("ㅁ")
                nxt[0] = CHO.index("ㄴ")
    return "".join(_compose(*s) for s in sylls)


@lru_cache(maxsize=20000)
def _romanize_run(run: str, already_pronounced: bool) -> str:
    source = run if already_pronounced else approximate_pronunciation(run)
    try:
        out = Romanizer(source).romanize()
    except Exception:  # noqa: BLE001
        return ""
    # The library writes ㄹ+ㄹ as "lr"; Revised Romanization uses "ll".
    return out.replace("lr", "ll")


def romanize(text: str, already_pronounced: bool = False) -> str:
    """Romanize every Hangul run in `text`, leaving other characters as they are."""
    if not text:
        return ""
    return HANGUL_RUN.sub(lambda m: _romanize_run(m.group(0), already_pronounced), text)


def clean_pronunciation(pron: str) -> str:
    """Dictionary pronunciations look like '궁ː물' or '마는/만는'. Keep the first form."""
    if not pron:
        return ""
    first = re.split(r"[/,;]", pron)[0]
    return "".join(ch for ch in first if "가" <= ch <= "힣" or ch == " ").strip()


def _matches(word: str, pron: str) -> bool:
    """Sound changes alter consonants, not vowels, so a dictionary pronunciation
    for `word` has the same number of syllables and (almost) the same vowels."""
    a = [ch for ch in word if "가" <= ch <= "힣"]
    b = [ch for ch in pron if "가" <= ch <= "힣"]
    if not a or len(a) != len(b):
        return False
    same = sum(1 for x, y in zip(a, b) if _decompose(x)[1] == _decompose(y)[1])
    return same >= max(1, round(len(a) * 0.7))


def romanize_headword(word: str, pronunciation: str = "") -> str:
    """Prefer the dictionary's pronunciation (accurate), fall back to the rules."""
    pron = clean_pronunciation(pronunciation)
    if pron and _matches(word, pron):
        return romanize(pron, already_pronounced=True)
    return romanize(word)


# --- Morphological analysis (Kiwi) ---------------------------------------------

_kiwi = None
_kiwi_lock = threading.Lock()


def kiwi():
    """Kiwi loads a ~0.5 GB model, so it is created once, on first use."""
    global _kiwi
    if _kiwi is None:
        with _kiwi_lock:
            if _kiwi is None:
                from kiwipiepy import Kiwi

                _kiwi = Kiwi()
    return _kiwi


def analyze(text: str) -> list[dict]:
    tokens = kiwi().tokenize(text or "")
    return [{"form": t.form, "tag": t.tag, "lemma": getattr(t, "lemma", t.form) or t.form} for t in tokens]


CONTENT_TAGS = ("NNG", "NNP", "VV", "VA", "XR", "MAG")


def content_lemmas(text: str) -> list[str]:
    """Dictionary forms of the content words in `text`, in order of appearance.

    걱정했어 → ['걱정', '걱정하다']; 망설여져서 → ['망설이다'].
    """
    toks = analyze(text)
    out: list[str] = []
    for i, t in enumerate(toks):
        tag = t["tag"].split("-")[0]
        if tag not in CONTENT_TAGS:
            continue
        lemma = t["lemma"]
        if tag in ("VV", "VA") and not lemma.endswith("다"):
            lemma = lemma + "다"
        out.append(lemma)
        # Noun + 하다/되다/스럽다 derivations: 걱정 + 하 → 걱정하다
        if tag in ("NNG", "XR") and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt["tag"] in ("XSV", "XSA") and nxt["form"] in ("하", "되", "스럽", "롭", "답", "시키"):
                out.append(t["form"] + nxt["form"] + "다")
    return out


def target_used(target: str, text: str) -> dict:
    """Did the learner use `target` (any conjugation) in `text`?"""
    target = (target or "").strip()
    text = text or ""
    if not target or not text:
        return {"used": False, "how": "empty"}
    lemmas = set(content_lemmas(text))
    parts = target.split()
    hits = []
    for part in parts:
        stem = part[:-1] if part.endswith("다") and len(part) > 1 else part
        ok = part in lemmas or stem in lemmas
        if not ok and part.endswith("하다") and part[:-2] in lemmas:
            ok = True  # 걱정하다 written as 걱정 했어 / 걱정을 했어
        if not ok and len(stem) >= 2 and stem in text:
            ok = True  # surface fallback for words Kiwi splits (e.g. new slang)
        hits.append(ok)
    used = all(hits)
    return {"used": used, "how": "morphology" if used else "not_found", "lemmas_found": sorted(lemmas)[:20]}


# --- Speech level ----------------------------------------------------------------

SPEECH_LEVELS = {
    "casual": "Casual (반말)",
    "polite": "Polite (해요체)",
    "formal": "Formal (합쇼체)",
}


def detect_speech_level(text: str) -> str:
    """Classify the sentence-final speech level of a text message."""
    cleaned = re.sub(r"[^가-힣\s]", " ", text or "")
    sentences = [s.strip() for s in re.split(r"\s{2,}|\n", cleaned) if s.strip()]
    if not sentences:
        return "unknown"
    endings = []
    for s in sentences:
        words = s.split()
        if not words:
            continue
        # "알겠습니다 팀장님" — skip a trailing term of address.
        if len(words) > 1 and words[-1].endswith(("님", "씨", "선배", "형", "누나", "언니", "오빠")):
            endings.append(words[-2])
        else:
            endings.append(words[-1])
    if any(e.endswith(("습니다", "습니까", "십시오", "니다", "니까")) for e in endings):
        return "formal"
    if any(e.endswith(("요", "죠", "세요")) for e in endings):
        return "polite"
    return "casual"


def level_ok(expected: str, detected: str) -> bool:
    if expected in ("", "any", "unknown") or detected == "unknown":
        return True
    if expected == "casual":
        return detected == "casual"
    if expected == "polite":
        return detected in ("polite", "formal")
    if expected == "formal":
        return detected in ("formal", "polite")
    return True


def hanja_chars(origin: str) -> list[str]:
    return HANJA_CHAR.findall(origin or "")
