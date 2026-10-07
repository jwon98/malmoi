"""Naver search-trend and blog/news search-count APIs.

Since 2026-07-31, new keys for these APIs come from NAVER API HUB on Naver Cloud
Platform (console → Application Services → NAVER API HUB → Application → 인증 정보).
Select "검색" and "검색어 트렌드" for the application, then set NAVER_CLIENT_ID and
NAVER_CLIENT_SECRET to its Client ID / Client Secret.

Keys issued by the old Naver Developers Center before 2026-07-31 still work until
2027-06-30; set NAVER_API=legacy to use them.
"""

from __future__ import annotations

import re
from datetime import date

import httpx

from . import config

HUB = "https://naverapihub.apigw.ntruss.com"
ENDPOINTS = {
    "hub": {"trend": HUB + "/search-trend/v1/search", "search": HUB + "/search/v1/{kind}"},
    "legacy": {"trend": "https://openapi.naver.com/v1/datalab/search",
               "search": "https://openapi.naver.com/v1/search/{kind}.json"},
}


def _mode() -> str:
    return "legacy" if config.NAVER_API == "legacy" else "hub"

_http = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))


class NaverError(Exception):
    pass


class NaverNotConfigured(NaverError):
    pass


def _headers() -> dict:
    if not (config.NAVER_CLIENT_ID and config.NAVER_CLIENT_SECRET):
        raise NaverNotConfigured("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET are not set")
    if _mode() == "legacy":
        return {"X-Naver-Client-Id": config.NAVER_CLIENT_ID, "X-Naver-Client-Secret": config.NAVER_CLIENT_SECRET}
    return {"X-NCP-APIGW-API-KEY-ID": config.NAVER_CLIENT_ID, "X-NCP-APIGW-API-KEY": config.NAVER_CLIENT_SECRET}


def _raise_for(resp: httpx.Response, api: str) -> None:
    if resp.status_code == 200:
        return
    if resp.status_code in (401, 403):
        raise NaverError(f"Naver rejected the request (HTTP {resp.status_code}). Check that the Client ID/Secret are "
                         f"copied exactly and that the {api} API is selected for the application in NAVER API HUB.")
    if resp.status_code == 429:
        raise NaverError("Naver's daily request limit was reached (HTTP 429). Try again tomorrow.")
    raise NaverError(f"Naver returned HTTP {resp.status_code}.")


def _months_back(end: date, months: int) -> date:
    y, m = end.year, end.month - months
    while m <= 0:
        m += 12
        y -= 1
    return date(max(y, 2016), m, 1) if y >= 2016 else date(2016, 1, 1)


def search_trend(terms: list[str], months: int = 24) -> dict:
    """Relative Naver search interest (0–100) for up to 5 terms."""
    headers = _headers()
    end = date.today()
    start = _months_back(end, months)
    unit = "month" if months >= 12 else "week"
    body = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "timeUnit": unit,
        "keywordGroups": [{"groupName": t, "keywords": [t]} for t in terms[:5]],
    }
    try:
        resp = _http.post(ENDPOINTS[_mode()]["trend"], json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise NaverError(f"could not reach Naver DataLab ({exc.__class__.__name__})") from exc
    _raise_for(resp, "검색어 트렌드 (search trend)")
    data = resp.json()
    series = []
    for group in data.get("results", []):
        points = [{"period": p["period"][:7] if unit == "month" else p["period"], "ratio": round(float(p["ratio"]), 1)}
                  for p in group.get("data", [])]
        series.append({"term": group.get("title", ""), "points": points})
    return {"start": start.isoformat(), "end": end.isoformat(), "unit": unit, "series": series}


_TAGS = re.compile(r"<[^>]+>")


def search_count(phrase: str, kind: str = "blog") -> dict:
    """Total Naver results for an exact phrase plus one short example snippet."""
    headers = _headers()
    try:
        resp = _http.get(ENDPOINTS[_mode()]["search"].format(kind=kind), params={"query": f'"{phrase}"', "display": 3, "sort": "sim"},
                         headers=headers)
    except httpx.HTTPError as exc:
        raise NaverError(f"could not reach Naver search ({exc.__class__.__name__})") from exc
    _raise_for(resp, "검색 (search)")
    data = resp.json()
    snippet = ""
    for item in data.get("items", []):
        text = _TAGS.sub("", item.get("description", "")).replace("&quot;", '"').replace("&amp;", "&").strip()
        if phrase.replace(" ", "") in text.replace(" ", ""):
            snippet = text
            break
    if len(snippet) > 140:
        pos = snippet.find(phrase)
        start = max(0, pos - 40) if pos >= 0 else 0
        snippet = ("…" if start else "") + snippet[start: start + 120].strip() + "…"
    return {"total": int(data.get("total", 0)), "snippet": snippet}
