"""Offline tests: no network, no Google Cloud credentials needed.

HTTP calls to the dictionaries and Naver are mocked with httpx.MockTransport,
and Gemini is replaced by a scripted fake, so the whole agent loop
(tool selection → execution → final answer) runs end to end.
"""

import json
import os

os.environ["USE_FIRESTORE"] = "false"
os.environ["KRDICT_API_KEY"] = "test-key"
os.environ["STDICT_API_KEY"] = "test-key"
os.environ["OPENDICT_API_KEY"] = "test-key"
os.environ["NAVER_CLIENT_ID"] = "id"
os.environ["NAVER_CLIENT_SECRET"] = "secret"

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from google.genai import types  # noqa: E402

from malmoi import agent, cache, config, dictionaries, llm, naver, tools  # noqa: E402

config.KRDICT_API_KEY = config.STDICT_API_KEY = config.OPENDICT_API_KEY = "test-key"
config.NAVER_CLIENT_ID, config.NAVER_CLIENT_SECRET = "id", "secret"

KRDICT_SEARCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<channel><total>1</total>
<item><target_code>12345</target_code><word>눈치</word><sup_no>0</sup_no><origin></origin>
<pronunciation>눈치</pronunciation><word_grade>중급</word_grade><pos>명사</pos>
<link>https://krdict.korean.go.kr/dicSearch/SearchView?ParaWordNo=12345</link>
<sense><sense_order>1</sense_order><definition>남의 마음을 그때그때 상황으로 미루어 알아내는 것.</definition>
<translation><trans_lang>영어</trans_lang><trans_word>tact; sense</trans_word><trans_dfn>The ability to grasp others' feelings from the situation.</trans_dfn></translation></sense>
<sense><sense_order>2</sense_order><definition>속으로 생각하는 것이 겉으로 드러나는 어떤 태도.</definition>
<translation><trans_lang>영어</trans_lang><trans_word>hint; sign</trans_word><trans_dfn>An attitude that reveals what one thinks.</trans_dfn></translation></sense>
</item></channel>"""

KRDICT_VIEW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<channel><item><word_info><word>눈치</word><pronunciation_info><pronunciation>눈치</pronunciation></pronunciation_info>
<sense_info><definition>...</definition>
<example_info><type>문장</type><example>그는 눈치가 빨라서 금방 알아챘다.</example></example_info>
<example_info><type>구</type><example>눈치가 없다</example></example_info>
</sense_info></word_info></item></channel>"""

KRDICT_ECONOMY_XML = """<channel><item><target_code>9</target_code><word>경제</word><origin>經濟</origin><pronunciation>경제</pronunciation>
<word_grade>중급</word_grade><pos>명사</pos><sense><definition>...</definition><translation><trans_word>economy</trans_word></translation></sense></item></channel>"""

KRDICT_EMPTY = "<channel><total>0</total></channel>"
KRDICT_ERROR = "<error><error_code>020</error_code><message>Unregistered key</message></error>"

STDICT_JSON = {"channel": {"total": 1, "item": [{"word": "망설-이다", "sup_no": "0", "target_code": "77",
                                                  "sense": {"definition": "이리저리 생각만 하고 태도를 결정하지 못하다.", "pos": "동사", "link": "https://stdict.korean.go.kr/x"}}]}}


def krdict_word(q: str) -> str:
    if q == "눈치":
        return KRDICT_SEARCH_XML
    if q == "경제":
        return KRDICT_ECONOMY_XML
    if q in ("경영", "경험"):
        return KRDICT_ECONOMY_XML.replace("경제", q).replace("經濟", "經營" if q == "경영" else "經驗")
    if q == "badkey":
        return KRDICT_ERROR
    return KRDICT_EMPTY


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    q = request.url.params.get("q", "")
    if "krdict.korean.go.kr/api/search" in url:
        return httpx.Response(200, text=krdict_word(q))
    if "krdict.korean.go.kr/api/view" in url:
        return httpx.Response(200, text=KRDICT_VIEW_XML)
    if "stdict.korean.go.kr/api/search" in url:
        return httpx.Response(200, json=STDICT_JSON) if q == "망설이다" else httpx.Response(200, text="")
    if "stdict.korean.go.kr/api/view" in url:
        return httpx.Response(200, json={"channel": {"item": {"word_info": {"pronunciation_info": [{"pronunciation": "망서리다"}]}}}})
    if "opendict" in url:
        return httpx.Response(200, json={"channel": {"total": 0, "item": []}})
    if "search-trend" in url or "datalab" in url:
        body = json.loads(request.content)
        results = [{"title": g["groupName"], "keywords": g["keywords"],
                    "data": [{"period": f"2025-{m:02d}-01", "ratio": r} for m, r in zip(range(1, 13), [5, 9, 20, 55, 100, 80, 60, 45, 40, 38, 35, 30])]}
                   for g in body["keywordGroups"]]
        return httpx.Response(200, json={"results": results})
    if "/search/v1/" in url or "openapi.naver.com/v1/search" in url:
        phrase = request.url.params.get("query", "").strip('"')
        total = 120000 if "내렸" in phrase else 30000
        return httpx.Response(200, json={"total": total, "items": [{"description": f"오늘 <b>{phrase}</b> 기로 했다"}]})
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def mock_http(monkeypatch):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(dictionaries, "_http", httpx.Client(transport=transport))
    monkeypatch.setattr(naver, "_http", httpx.Client(transport=transport))
    dictionaries._cache.clear()
    cache.clear_memory()
    tools._prefetched.clear()
    llm._background_paused_until = 0.0


# ---------------------------------------------------------------- parsers

def test_krdict_lookup_builds_entry():
    entry, errors, configured = dictionaries.lookup("눈치")
    assert configured and not errors
    assert entry["source"] == "krdict"
    assert entry["romanization"] == "nunchi"
    assert entry["senses"][0]["english_word"] == "tact; sense"
    assert entry["senses"][1]["n"] == 2
    assert entry["examples"] == ["그는 눈치가 빨라서 금방 알아챘다."]
    assert entry["level_en"] == "intermediate"


def test_fallback_to_stdict_and_headword_cleanup():
    entry, _, _ = dictionaries.lookup("망설이다")
    assert entry["source"] == "stdict"
    assert entry["word"] == "망설이다"
    assert entry["romanization"] == "mangseorida"


def test_krdict_error_is_actionable():
    with pytest.raises(dictionaries.DictError, match="API key rejected"):
        dictionaries.krdict_search("badkey")


def test_lookup_tool_handles_conjugated_form():
    ctx = tools.ToolContext(user="t@x.com")
    r = tools.lookup_word(ctx, "망설여져서", "계속 망설여져서 말을 못 했어")
    assert r["ok"] and r["word"] == "망설이다"
    assert "conjugated form" in r["looked_up_as"]
    assert r["context_sentence"]
    assert tools.store.get_word("t@x.com", "망설이다")["contexts"] == ["계속 망설여져서 말을 못 했어"]


def test_lookup_tool_rejects_english():
    r = tools.lookup_word(tools.ToolContext(user="u"), "awkward")
    assert r["ok"] is False and "find_korean_words" in r["hint"]


def test_naver_uses_api_hub_by_default(monkeypatch):
    seen = []
    def capture(request):
        seen.append((str(request.url), dict(request.headers)))
        return handler(request)
    monkeypatch.setattr(naver, "_http", httpx.Client(transport=httpx.MockTransport(capture)))
    naver.search_count("결정을 내렸어", "blog")
    url, headers = seen[0]
    assert url.startswith("https://naverapihub.apigw.ntruss.com/search/v1/blog")
    assert headers["x-ncp-apigw-api-key-id"] == "id"
    monkeypatch.setattr(config, "NAVER_API", "legacy")
    naver.search_count("결정을 내렸어", "news")
    assert seen[-1][0].startswith("https://openapi.naver.com/v1/search/news.json")


def test_word_trend_and_naturalness():
    ctx = tools.ToolContext(user="u")
    t = tools.word_trend(ctx, ["킹받다"], 12)
    assert t["ok"] and t["series"][0]["summary"]["peak_period"] == "2025-05"
    n = tools.check_naturalness(ctx, ["결정을 내렸어", "결정을 했어"])
    assert n["ok"] and n["results"][0]["phrase"] == "결정을 내렸어"
    assert n["results"][0]["share_pct"] == 80.0


# ---------------------------------------------------------- agent loop

class FakeModels:
    """Scripted Gemini: returns the next queued response each call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(list(contents))
        step = self.script.pop(0)
        if isinstance(step, tuple):  # function call(s)
            parts = [types.Part(function_call=types.FunctionCall(name=n, args=a, id=f"c{i}")) for i, (n, a) in enumerate(step)]
        else:
            parts = [types.Part(text=step)]
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def test_chat_endpoint_shape_and_sessions(monkeypatch):
    import app as app_module

    fake = FakeClient([
        ((("lookup_word", {"word": "눈치", "context_sentence": "걔는 눈치가 없어"}),)),
        "**눈치** here means sense 1: reading the room.",
        "Second session answer.",
    ])
    monkeypatch.setattr(llm, "client", lambda: fake)
    c = TestClient(app_module.app)
    r = c.post("/chat", json={"message": "What does 눈치 mean in 걔는 눈치가 없어?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"response", "session_id", "tool_calls"}
    assert body["tool_calls"][0]["name"] == "lookup_word"
    assert body["tool_calls"][0]["args"]["word"] == "눈치"
    assert body["tool_calls"][0]["result"]["ok"]
    assert "reading the room" in body["response"]

    # The function response was sent back to the model with the call id.
    second_call_history = fake.models.calls[1]
    assert second_call_history[-1].parts[0].function_response.id == "c0"

    # A new session doesn't see the first one's history.
    r2 = c.post("/chat", json={"message": "hi"})
    assert r2.json()["session_id"] != body["session_id"]
    assert len(fake.models.calls[2]) == 1

    # Word bank API sees the saved word.
    words = c.get("/api/words").json()["words"]
    assert any(w["word"] == "눈치" for w in words)
    assert c.get("/api/export.csv").status_code == 200
    assert c.get("/api/export.apkg").content[:2] == b"PK"


def test_drill_flow(monkeypatch):
    scenario = {"partner_message_ko": "어제 왜 연락 안 했어?", "situation_en": "Your friend is upset.",
                "task_en": "Tell her you felt hurt that she forgot your birthday.",
                "model_answer_ko": "솔직히 네가 내 생일 잊어서 좀 서운했어.", "hint_en": "Starts with 서."}
    judge = {"meaning_matches": True, "uses_acceptable_alternative": False, "feedback_en": "Nice.",
             "better_version_ko": "솔직히 좀 서운했어.", "partner_reply_ko": "헐 미안해 ㅠㅠ"}
    queue = [scenario, judge]
    monkeypatch.setattr(llm, "generate_json", lambda prompt, schema, system=None, thinking=None: queue.pop(0))
    # Drill tools are "terminal": no second model call after them.
    fake = FakeClient([
        ((("start_text_drill", {"mode": "produce", "word": "서운하다", "persona": "close friend"}),)),
        ((("check_drill_answer", {"reply": "솔직히 네가 생일 잊어서 서운했어"}),)),
    ])
    monkeypatch.setattr(llm, "client", lambda: fake)
    sid = None
    out = agent.run_turn("Start a new drill.", sid, "drill@x.com", "drill")
    sid = out["session_id"]
    start = out["tool_calls"][0]["result"]
    assert start["ok"] and start["answer_key"]["target_word"] == "서운하다"
    out2 = agent.run_turn("솔직히 네가 생일 잊어서 서운했어", sid, "drill@x.com", "drill")
    res = out2["tool_calls"][0]["result"]
    assert res["verdict"] == "correct", res
    assert res["speech_level_ok"] and res["mastery_after"] == 1
    assert "Minji" not in out["response"] and "민지" in out["response"]  # built from the tool result
    assert len(fake.models.calls) == 2  # one model call per turn


def test_failed_turn_is_rolled_back(monkeypatch):
    class Boom:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(**_):
                raise RuntimeError("PERMISSION_DENIED")

    monkeypatch.setattr(llm, "client", lambda: Boom())
    out = agent.run_turn("hello", None, "u@x.com", "ask")
    assert "Permission denied" in out["response"]


def test_generate_then_verify_tools(monkeypatch):
    ctx = tools.ToolContext(user="g@x.com")

    def fake_json(prompt, schema, system=None, thinking=None):
        if "hanja" in prompt:
            return {"roots": [{"hanja": "經", "reading": "경", "meaning_en": "manage; pass through",
                               "words": [{"word": "경영", "hanja": "經營", "english": "management"},
                                         {"word": "경험", "hanja": "經驗", "english": "experience"},
                                         {"word": "가짜단어", "hanja": "假", "english": "fake"}]},
                              {"hanja": "濟", "reading": "제", "meaning_en": "relieve", "words": []}]}
        if "exploring the topic" in prompt:
            return {"words": [{"word": "눈치", "english": "tact", "why_useful": "x", "example_ko": "눈치가 빠르다", "example_en": "y"},
                              {"word": "없는말", "english": "fake", "why_useful": "x", "example_ko": "-", "example_en": "-"}]}
        return {"candidates": [{"word": "눈치", "register": "neutral", "nuance": "n", "example_ko": "e", "example_en": "e"},
                               {"word": "뻘쭘하다", "register": "slang", "nuance": "n", "example_ko": "e", "example_en": "e"}]}

    monkeypatch.setattr(llm, "generate_json", fake_json)
    h = tools.hanja_family(ctx, "경제")
    assert h["ok"] and h["origin"] == "經濟"
    fam = [f["word"] for f in h["roots"][0]["family"]]
    assert fam == ["경영", "경험"] and h["roots"][0]["reading"] == "경"

    e = tools.explore_domain(ctx, "feelings", "advanced", 4)
    # Verified words first, then unverified ones fill the set, clearly labeled.
    assert [(w["word"], w["verified"]) for w in e["words"]] == [("눈치", True), ("없는말", False)]
    assert e["unverified_count"] == 1
    # "Show more": the same topic in the same session never repeats words.
    again = tools.explore_domain(ctx, "feelings", "advanced", 4)
    assert again["words"] == []

    f = tools.find_korean_words(ctx, "awkward")
    assert [c["verified"] for c in f["candidates"]] == [True, False]

    monkeypatch.setattr(llm, "grounded_answer", lambda p, system=None: (
        "Meaning: so annoying it's 'king'-level\nOrigin: internet slang\nRegister: very casual\n"
        "Example: 아 진짜 킹받네 — ugh, so annoying\nStill current?: mainstream", [{"title": "namu", "url": "https://x"}]))
    s = tools.search_slang(ctx, "킹받다")
    assert s["ok"] and s["web"]["Meaning"].startswith("so annoying") and s["sources"]
    assert tools.store.get_word("g@x.com", "킹받다")["source"] == "web"

    m = tools.mine_vocabulary(ctx, "그는 눈치가 빨라서 분위기를 금방 알아챘다. 경제 뉴스도 봤다.")
    assert m["ok"] and "눈치" in [w["word"] for w in m["words"]]


def test_naver_tools_hidden_without_keys(monkeypatch):
    monkeypatch.setattr(config, "NAVER_CLIENT_ID", "")
    tools_now = agent.active_tools("ask")
    assert "word_trend" not in tools_now and "check_naturalness" not in tools_now
    assert "lookup_word" in tools_now and "search_slang" in tools_now
    assert "word_trend" not in agent.system_prompt("ask").split("Tools available right now:")[1]
    monkeypatch.setattr(config, "NAVER_CLIENT_ID", "id")
    assert "word_trend" in agent.active_tools("ask")


def test_drill_word_sources(monkeypatch):
    ctx = tools.ToolContext(user="src@x.com")
    # Empty word bank → falls back to a random useful word and says so.
    word, gloss, note = tools._pick_drill_word(ctx, "word_bank", "", "advanced")
    assert word in dict(tools.STARTER_WORDS) and "empty" in note.lower()
    # Topic source → a dictionary-checked word for that topic.
    monkeypatch.setattr(llm, "generate_json", lambda prompt, schema, system=None, thinking=None: {"words": [
        {"word": "눈치", "english": "tact", "why_useful": "x", "example_ko": "x", "example_en": "x"}]})
    word, gloss, note = tools._pick_drill_word(ctx, "topic", "feelings", "advanced")
    assert word == "눈치" and gloss == "tact" and "feelings" in note
    # Recently drilled words are avoided by the random source when possible.
    ctx.session["recent_drill_words"] = [w for w, _ in tools.STARTER_WORDS[1:]]
    assert tools._pick_drill_word(ctx, "random", "", "advanced")[0] == tools.STARTER_WORDS[0][0]


def test_save_unverified_word_from_card():
    import app as app_module
    c = TestClient(app_module.app)
    r = c.post("/api/words", json={"word": "없는말", "gloss": "made-up word"})
    assert r.status_code == 200 and r.json()["verified"] is False
    assert r.json()["word"]["source"] == "ai" and r.json()["word"]["gloss"] == "made-up word"


def test_stream_endpoint_and_parallel_tools(monkeypatch):
    import app as app_module
    fake = FakeClient([
        ((("lookup_word", {"word": "눈치"}), ("lookup_word", {"word": "망설이다"}))),
        "Both looked up.",
    ])
    monkeypatch.setattr(llm, "client", lambda: fake)
    c = TestClient(app_module.app)
    with c.stream("POST", "/chat/stream", json={"message": "compare 눈치 and 망설이다"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert "Checking the dictionary for 눈치" in statuses
    done = events[-1]
    assert done["type"] == "done" and set(done) == {"type", "response", "session_id", "tool_calls"}
    assert [t["args"]["word"] for t in done["tool_calls"]] == ["눈치", "망설이다"]
    # Both function responses go back to the model in one message, in order.
    parts = fake.models.calls[1][-1].parts
    assert [p.function_response.id for p in parts] == ["c0", "c1"]


class NoModel:
    """Fails the test if the agent asks the model anything."""
    class models:  # noqa: N801
        @staticmethod
        def generate_content(**_):
            raise AssertionError("the model should not have been called")

        @staticmethod
        def generate_content_stream(**_):
            raise AssertionError("the model should not have been called")


def test_drill_buttons_run_tools_directly_with_prefetch(monkeypatch):
    import app as app_module

    scenario = {"partner_message_ko": "뭐해?", "situation_en": "s", "task_en": "t",
                "model_answer_ko": "좀 서운했어", "hint_en": "starts with 서"}
    built = []

    def fake_json(prompt, schema, system=None, thinking=None):
        if "KakaoTalk" in prompt:
            built.append(prompt)
            return dict(scenario)
        return {"meaning_matches": True, "uses_acceptable_alternative": False, "feedback_en": "Good.",
                "better_version_ko": "좀 서운했어", "partner_reply_ko": "미안!"}

    monkeypatch.setattr(llm, "generate_json", fake_json)
    monkeypatch.setattr(tools, "_pick_drill_word", lambda ctx, source, topic, level: ("서운하다", "hurt", "A random useful word"))
    monkeypatch.setattr(llm, "client", lambda: NoModel())
    c = TestClient(app_module.app)
    settings = {"mode": "produce", "persona": "close friend", "source": "random", "topic": "", "level": "advanced"}

    # Written ahead of time...
    assert c.post("/api/drill/prefetch", json=settings).json() == {"ready": True}
    assert len(built) == 1
    # ...so starting the drill uses it: no new scenario, and no model call to pick the tool.
    first = c.post("/chat", json={"message": "Start a new drill", "mode": "drill",
                                  "action": {"tool": "start_text_drill", "args": settings}}).json()
    assert set(first) == {"response", "session_id", "tool_calls"}
    assert first["tool_calls"][0]["name"] == "start_text_drill" and len(built) == 1
    assert first["tool_calls"][0]["result"]["answer_key"]["hint_en"] == "starts with 서"
    sid = first["session_id"]

    answer = c.post("/chat", json={"message": "좀 서운했어", "mode": "drill", "session_id": sid,
                                   "action": {"tool": "check_drill_answer", "args": {"reply": "좀 서운했어"}}}).json()
    assert answer["tool_calls"][0]["result"]["verdict"] == "correct"

    # Only tools that map to a button can be run directly.
    bad = c.post("/chat", json={"message": "x", "mode": "drill", "session_id": sid,
                                "action": {"tool": "update_word", "args": {"word": "x", "action": "delete"}}}).json()
    assert bad["tool_calls"] == [] and "can't be run directly" in bad["response"]

    # A new drill to test the instant reveal endpoint.
    second = c.post("/chat", json={"message": "Start", "mode": "drill", "session_id": sid,
                                   "action": {"tool": "start_text_drill", "args": settings}}).json()
    assert second["tool_calls"][0]["result"]["ok"]
    r = c.post("/api/drill/reveal", json={"session_id": sid}).json()
    assert r["verdict"] == "revealed" and r["answer_key"]["target_word"] == "서운하다"
    assert c.post("/api/drill/reveal", json={"session_id": sid}).status_code == 404  # already revealed


def test_repeated_first_question_is_answered_from_cache(monkeypatch):
    import app as app_module

    fake = FakeClient([((("lookup_word", {"word": "눈치"}),)), "**눈치** is reading the room."])
    monkeypatch.setattr(llm, "client", lambda: fake)
    c = TestClient(app_module.app)
    q = "What does 눈치 mean?"
    first = c.post("/chat", json={"message": q}).json()
    assert first["response"] == "**눈치** is reading the room."

    # Same question, new conversation, different user: tools run again, the model doesn't.
    monkeypatch.setattr(llm, "client", lambda: NoModel())
    second = c.post("/chat", json={"message": "  what does 눈치 mean? "},
                    headers={"x-goog-authenticated-user-email": "accounts.google.com:other@columbia.edu"}).json()
    assert second["response"] == first["response"] and second["session_id"] != first["session_id"]
    assert second["tool_calls"][0]["name"] == "lookup_word" and second["tool_calls"][0]["result"]["ok"]
    assert tools.store.get_word("other@columbia.edu", "눈치")  # saved to that user's word bank

    # A follow-up in that conversation goes to the model as usual.
    monkeypatch.setattr(llm, "client", lambda: FakeClient(["Sense 2 is a hint or sign."]))
    follow = c.post("/chat", json={"message": "and sense 2?", "session_id": second["session_id"]},
                    headers={"x-goog-authenticated-user-email": "accounts.google.com:other@columbia.edu"}).json()
    assert follow["response"] == "Sense 2 is a hint or sign."


def test_rate_limits_are_retried(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(1)
            if len(calls) < 3:
                err = RuntimeError("429 RESOURCE_EXHAUSTED")
                err.code = 429
                raise err
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text='{"ok": true}')]))])

    class C:
        models = Models()

    monkeypatch.setattr(llm, "client", lambda: C())
    assert llm.generate_json("x", {"type": "object"}) == {"ok": True}
    assert len(calls) == 3


def test_background_flag_carries_into_worker_threads():
    seen = []

    def work():
        seen.append(llm.BACKGROUND.get())

    def run():
        llm.BACKGROUND.set(True)
        tools._submit(tools._gen_pool, work).result()

    import contextvars
    contextvars.copy_context().run(run)
    tools._submit(tools._gen_pool, work).result()
    assert seen == [True, False]


def test_thinking_level_fallback(monkeypatch):
    seen = []

    class Models:
        def generate_content(self, model, contents, config):
            level = config.thinking_config.thinking_level if config.thinking_config else None
            seen.append(level)
            if level is not None:
                err = RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported for this model")
                err.code = 400
                raise err
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text='{"a": 1}')]))])

    class C:
        models = Models()

    monkeypatch.setattr(llm, "client", lambda: C())
    monkeypatch.setattr(llm, "_unsupported_levels", set())
    assert llm.generate_json("x", {"type": "object"}) == {"a": 1}
    assert [str(x).lower() if x else None for x in seen][-1] is None  # finally called without thinking
    seen.clear()
    llm.generate_json("y", {"type": "object"})
    assert seen == [None]  # remembers the model doesn't support it


class StreamingModels(FakeModels):
    """Like FakeModels, but also supports generate_content_stream, splitting text into chunks."""

    def generate_content_stream(self, model, contents, config):
        resp = self.generate_content(model, contents, config)
        parts = resp.candidates[0].content.parts
        if parts[0].text:
            words = parts[0].text.split(" ")
            for i, w in enumerate(words):
                piece = w + (" " if i < len(words) - 1 else "")
                yield types.GenerateContentResponse(candidates=[types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text=piece)]))])
        else:
            yield resp


def test_text_streams_as_it_is_written(monkeypatch):
    import app as app_module

    class C:
        models = StreamingModels(["**눈치** means reading the room."])

    monkeypatch.setattr(llm, "client", lambda: C())
    c = TestClient(app_module.app)
    with c.stream("POST", "/chat/stream", json={"message": "hi"}) as r:
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    deltas = [e["delta"] for e in events if e["type"] == "text"]
    assert len(deltas) > 1 and "".join(deltas) == "**눈치** means reading the room."
    assert events[-1]["response"] == "**눈치** means reading the room."
    # The streamed message is stored as ONE model message in the history.
    sid = events[-1]["session_id"]
    hist = [types.Content.model_validate_json(x) for x in agent.store.load_session(sid)["history"]]
    assert hist[-1].role == "model" and hist[-1].parts[0].text == "**눈치** means reading the room."


def test_explore_streams_batches_and_warm_cache_avoids_gemini(monkeypatch):
    import app as app_module
    calls = []

    def fake_json(prompt, schema, system=None, thinking=None):
        if "in hanja" in prompt:  # the 경제 sample-query warm-up
            return {"roots": []}
        calls.append(prompt)
        n = len(calls)
        return {"words": [
            {"word": w, "english": "e", "why_useful": "w", "example_ko": "x", "example_en": "y"}
            for w in (["눈치", "경제"] if n == 1 else ["망설이다"] if n == 2 else ["없는말"])]}

    monkeypatch.setattr(llm, "generate_json", fake_json)
    c = TestClient(app_module.app)
    r = c.post("/api/warm", json={"level": "advanced", "topics": ["feelings"]}).json()
    assert r["status"] == "ok" and r["ready"] == r["total"]
    assert len(calls) == 3  # three angles, generated in parallel

    calls.clear()
    events = []
    ctx = tools.ToolContext(user="warm@x.com", emit=events.append)
    out = tools.explore_domain(ctx, "feelings", "advanced", 4)
    assert calls == []  # served from the warmed pool: no Gemini calls
    assert {w["word"] for w in out["words"]} == {"눈치", "경제", "망설이다", "없는말"}
    assert any(e["type"] == "partial" for e in events)
    assert out["words"][-1]["verified"] is False  # unverified words come last


def test_warm_up_prepares_sample_answers(monkeypatch):
    import app as app_module

    fake = FakeClient([((("hanja_family", {"word": "경제"}),)), "**경제** comes from 經濟."])
    monkeypatch.setattr(llm, "client", lambda: fake)
    monkeypatch.setattr(llm, "generate_json", lambda prompt, schema, system=None, thinking=None: {"roots": []})
    c = TestClient(app_module.app)
    sample = "Break down 경제 into its hanja roots and show me related words."
    r = c.post("/api/warm", json={"level": "advanced", "topics": [], "samples": [sample]}).json()
    assert r["ready"] == r["total"] == 1

    monkeypatch.setattr(llm, "client", lambda: NoModel())  # now answered without the model
    out = c.post("/chat", json={"message": sample}).json()
    assert out["response"] == "**경제** comes from 經濟." and out["tool_calls"][0]["name"] == "hanja_family"
