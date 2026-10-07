# Malmoi 말모이

**A Korean vocabulary agent for advanced learners: people who are conversational or fluent but keep running into words a one-word translation doesn't explain.**

Duolingo stops being useful once you can hold a conversation. What's left is the long tail: a word in a text from a friend, a headline, a show, where the dictionary lists six senses and you need the one in front of you. Malmoi looks those words up in real dictionaries, explains the sense used in *your* sentence, handles slang the dictionaries haven't caught up with, saves everything to a word bank, and then makes you use the words by texting a simulated friend, coworker, or boss.

The name comes from *Malmoi* (말모이, "gathering words"), the first modern Korean dictionary manuscript, compiled word by word in the 1910s. This app does the same thing for one learner at a time.

## Try these three queries

Open the deployed app, stay on the **Ask** tab, and paste:

1. `What does 눈치 mean in this sentence: "걔는 눈치가 없어서 그 얘기를 또 꺼냈어"?`
   Looks the word up in the National Institute of Korean Language's Basic Korean Dictionary, picks the sense used in that sentence, and saves the word (with the sentence) to the word bank.
2. `How do I say "awkward" in Korean? I want casual and formal options.`
   English → Korean. Gemini proposes candidates; each one is checked against the dictionaries and labeled by register (neutral 어색하다, slang 뻘쭘하다, …).
3. `What does 킹받네 mean, and is it still popular?`
   Slang isn't in the dictionaries, so the agent runs a Google-Search-grounded lookup with source links, covering meaning, origin, register, and whether it's still current.

Then try the other tabs:

- **Text drill.** Choose "Explain in English" (no Korean typing needed) and press Start drill. You'll get a Korean text message that uses a word from your word bank; explain what it means and Malmoi grades you. "Reply in Korean" is the harder mode: you get a situation in English and must reply in Korean using the hidden word at the right politeness level.
- **Word bank.** Every word you've looked up, with the sentence you found it in, mastery tracking, and export to Anki or CSV.
- **Explore.** Pick a topic (finance, dating, workplace, internet slang…) for dictionary-verified vocabulary at your level.

Romanization appears above every Korean word (toggle it in the top bar), and every word has a play button for audio.

## Tools

The agent has 12 tools. Each one does something the model can't do reliably by itself: read a real dictionary, read live data, run morphological analysis, or keep state. When a tool uses Gemini internally, the output is verified before it's returned ("generate, then verify").

| Tool | What it does | Data |
|---|---|---|
| `lookup_word` | Dictionary lookup with numbered senses, English, pronunciation, level, hanja origin, examples. Accepts conjugated forms (망설여져서 → 망설이다) via morphological analysis. Takes the user's sentence so the agent can pick the right sense. Auto-saves to the word bank. | NIKL 한국어기초사전 → 표준국어대사전 → 우리말샘 (external APIs) |
| `find_korean_words` | English → Korean candidates with register and nuance; every candidate is checked against the dictionaries and labeled verified or not. | Gemini + NIKL dictionaries |
| `search_slang` | Slang and new words: meaning, origin, register, example, whether it's still current, with source links. | Gemini with Google Search grounding + 우리말샘 |
| `word_trend` | Naver search interest over time for up to five terms, with peak, current level, and direction. The UI draws the chart. | Naver search trend API (NAVER API HUB) |
| `check_naturalness` | Compares phrasings (결정을 내리다 vs 결정을 하다) by how often each exact phrase appears in Naver blogs and news, with a real example. | Naver search API (NAVER API HUB) |
| `mine_vocabulary` | Paste a Korean article; finds the intermediate/advanced words you don't know yet, in dictionary form. | Kiwi morphological analyzer + dictionary levels |
| `explore_domain` | Topic vocabulary at a chosen level, skipping words you already have; unverifiable suggestions are dropped. | Gemini + NIKL dictionaries |
| `hanja_family` | Splits a Sino-Korean word into its hanja roots (from the dictionary's origin field) and finds related words that share each root. | NIKL dictionaries + Gemini, verified |
| `start_text_drill` | Picks a word due for review and writes a realistic texting scenario with a persona whose relationship sets the expected speech level. | Word bank + Gemini |
| `check_drill_answer` | Detects the target word in any conjugation with Kiwi, checks speech level (반말 / 해요체 / 합쇼체) against the relationship, judges meaning and alternatives, and updates spaced-repetition mastery. | Kiwi + Gemini + Firestore |
| `get_word_bank` / `update_word` | Read, star, annotate, or delete saved words. | Firestore |

`word_trend` and `check_naturalness` are optional: they need NAVER API HUB keys, which Naver Cloud Platform currently issues only to accounts with a Korean business registration. When those keys aren't set, the app hides both tools from the model instead of letting them fail, and the other 10 tools work normally.

Errors are written for the model to act on. Each failed call returns `{"ok": false, "error": "...", "hint": "..."}`, for example "Not in any dictionary. If it looks like slang, call search_slang."

## How it works

```
Browser (static/)  ──POST /chat {message, session_id, mode}──▶  app.py (FastAPI)
                                                                   │
                                                     malmoi/agent.py: Gemini function-calling loop
                                                     (history per session, tools filtered by mode)
                                                                   │
              ┌──────────────────┬───────────────────┬─────────────┴─────┬────────────────┬──────────────────┐
       dictionaries.py       naver.py           korean.py            llm.py          storage.py
       NIKL krdict/stdict/   DataLab trend,     Kiwi morphology,     Gemini JSON +   Firestore word bank
       opendict              search counts      romanization,        grounded        + sessions
                                                speech level         search
```

- **`/chat`** keeps the starter's response shape: `response`, `session_id`, and `tool_calls` (each with `name`, `args`, `result`). The UI shows every call under each answer.
- **Sessions:** each `/chat` session has its own history, stored in Firestore (with an in-memory cache), and is bound to the signed-in user and the tab it came from, so sessions never leak into each other.
- **Users:** Identity-Aware Proxy passes the signed-in Columbia account in a header; each account gets its own word bank.
- **Romanization** is computed in code, never by the model. It follows pronunciation, not spelling (국물 → *gungmul*, 신라 → *silla*, 같이 → *gachi*), using the dictionary's pronunciation when available and common sound-change rules otherwise.
- **Audio** uses Google Cloud Text-to-Speech, falling back to the browser's Korean voice.

### Speed and rate limits

Most of the wait in an agent app is sequential model calls, so Malmoi keeps them few, short, and reliable:

- **Rate-limit handling.** Gemini on Vertex AI returns 429 (`RESOURCE_EXHAUSTED`) when requests arrive faster than the quota allows or shared capacity is busy. Every Gemini call retries with exponential backoff and jitter (up to ~15 s for what the user clicked), at most `GEMINI_MAX_CONCURRENT` calls run at once per instance, and background work is limited to 2 concurrent calls and pauses after any 429, so it never crowds out the user.
- **Typed questions go through the agent; single-tool buttons don't.** In Ask, the model reads the message and chooses tools. Buttons that map to exactly one tool (Start drill, sending a drill reply, picking an Explore topic) call that tool directly via the `action` field on `/chat`, skipping a model round trip that involved no judgment. The response keeps the same `/chat` shape, and the tool call is still shown. Grading a drill reply still uses Gemini to judge meaning.
- **Answer cache.** The first question in a new Ask conversation is cached with the tool calls the model chose and the answer it wrote (60 days). When anyone asks the same thing again, such as the sample questions or the README queries, the tools run again (so the word is saved to that user's word bank) but both model calls are skipped. Only tools that don't depend on who's asking are cached.
- **Warm-up.** When the site opens, the browser calls `/api/warm` once per level (Advanced first). It builds the word pool for every Explore topic and runs the sample questions through the agent at low priority, filling the caches above. Results are stored in Firestore (collection `cache`) for 60 days and shared across Cloud Run instances, so later visits only read the cache.
- **Drill prefetch.** Opening the drill tab, changing drill settings, or starting a drill asks `/api/drill/prefetch` to write the next drill ahead of time, so Start drill and Next word are usually instant. It builds inside that request rather than a background thread, because Cloud Run only reliably gives CPU to open requests. The hint is written with the scenario, and Show answer uses a direct endpoint with no model call.
- **Lower thinking depth.** The agent loop uses `thinking_level=low`; JSON generation inside tools uses `minimal`; grading uses `low`. If a model doesn't accept a level, the app falls back automatically.
- **Terminal tools and parallel tools.** In the drill and Explore tabs the tool result *is* the answer, so no extra "write a reply" call is made. When the model asks for several independent tools at once, they run concurrently.
- **Live progress.** `/chat/stream` runs the same turn as `/chat` but streams Server-Sent Events: progress (`status`), answer text as it's written (`text`), and Explore cards as they're verified (`partial`). Its final `done` event carries exactly the `/chat` payload. `/chat` itself is unchanged.
- **Other caches.** English → Korean candidates (14 days), hanja breakdowns (30 days), and web-grounded slang explanations (3 days) are cached in Firestore; dictionary responses in memory. Word banks are never cached or shared.

The server log prints `timing:` lines for every model call and tool, so you can see where time goes. To avoid the slow first request after Cloud Run scales to zero, run `gcloud run services update malmoi --region=us-central1 --min-instances=1` (about $13/month while on; set it back to `0` afterwards).

### Limitations

- Romanization of running text uses rules for the most common sound changes; rare exceptions (e.g. 의견란) may be off. Dictionary headwords use the dictionary's own pronunciation.
- Slang explanations come from web search and are labeled "Web-sourced" with links; they're less reliable than dictionary entries.
- Naver counts are estimated result totals, useful for comparing phrasings, not exact frequencies.

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and the gcloud CLI set up as in the course's GCP guide.

```bash
gcloud auth application-default login       # if you haven't recently
cp .env.example .env                        # then fill in your keys
uv run app.py                               # http://localhost:8080
uv run pytest                               # offline tests, no keys needed
```

Without dictionary keys the app still runs: definitions come from Gemini and are marked "Not dictionary-verified". Without Firestore it keeps the word bank in memory.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_MODEL` | no (default `gemini-3.8-flash`) | Gemini model ID on Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | no (default `global`) | Vertex AI location |
| `GEMINI_THINKING` | no (default `low`) | Thinking level for the agent loop: `minimal`, `low`, `medium`, `high`, or `off` |
| `GEMINI_TOOL_THINKING` | no (default `minimal`) | Thinking level for JSON generation inside tools |
| `GEMINI_MAX_CONCURRENT` | no (default `6`) | Max Gemini requests in flight at once per server instance |
| `GOOGLE_CLOUD_PROJECT` | no (auto-detected) | GCP project ID |
| `KRDICT_API_KEY` | recommended | 한국어기초사전 key: https://krdict.korean.go.kr/openApi/openApiInfo |
| `STDICT_API_KEY` | optional | 표준국어대사전 key: https://stdict.korean.go.kr/openapi/openApiInfo.do |
| `OPENDICT_API_KEY` | optional | 우리말샘 key: https://opendict.korean.go.kr/service/openApiInfo |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | optional | Client ID / Client Secret of a NAVER API HUB application (Naver Cloud Platform) with 검색 and 검색어 트렌드 selected |
| `NAVER_API` | no (default `hub`) | Set `legacy` only for old Naver Developers Center keys issued before 2026-07-31 |
| `USE_FIRESTORE` | no (default `auto`) | `auto`, `true`, or `false` |
| `TTS_VOICE` | no (default `ko-KR-Neural2-A`) | Cloud Text-to-Speech voice |
| `NIKL_SSL_VERIFY` | no (default `true`) | Set `false` only if dictionary calls fail with SSL errors |

## Deploy (Cloud Run, continuous deploy from GitHub)

Follow the course guide *Deploying to Cloud Run from GitHub* with this repo. Run the commands below once in your class project.

**1. Enable services and create the Firestore database**

```bash
PROJECT_ID=$(gcloud config get-value project)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com developerconnect.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com firestore.googleapis.com texttospeech.googleapis.com
gcloud firestore databases create --location=nam5
```

**2. Give the Cloud Run service account the roles it needs**

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
SA=$PROJECT_NUMBER-compute@developer.gserviceaccount.com
for ROLE in roles/aiplatform.user roles/datastore.user roles/secretmanager.secretAccessor roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$SA --role=$ROLE --condition=None
done
```

**3. Store API keys in Secret Manager** (one command per key you have)

```bash
printf '%s' 'PASTE_KRDICT_KEY'   | gcloud secrets create KRDICT_API_KEY --data-file=-
printf '%s' 'PASTE_STDICT_KEY'   | gcloud secrets create STDICT_API_KEY --data-file=-
printf '%s' 'PASTE_OPENDICT_KEY' | gcloud secrets create OPENDICT_API_KEY --data-file=-
printf '%s' 'PASTE_NAVER_ID'     | gcloud secrets create NAVER_CLIENT_ID --data-file=-
printf '%s' 'PASTE_NAVER_SECRET' | gcloud secrets create NAVER_CLIENT_SECRET --data-file=-
```

**4. Create the service** (course guide steps 3–7): Connect repository → Developer Connect → this repo, branch `main`, **buildpack** with entrypoint

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Require authentication with **Identity-Aware Proxy**, principal `columbia.edu`. Name the service `malmoi`.

**5. Set memory, environment variables, and secrets.** The Korean morphological analyzer needs about 0.5 GB, so the default 512 MiB is not enough. Use the region you picked in step 4, and include only the secrets you created.

```bash
gcloud run services update malmoi --region=us-east1 --memory=1Gi --timeout=300 \
  --set-env-vars=GEMINI_MODEL=gemini-3.8-flash,GOOGLE_CLOUD_LOCATION=global,GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  --set-secrets=KRDICT_API_KEY=KRDICT_API_KEY:latest,STDICT_API_KEY=STDICT_API_KEY:latest,OPENDICT_API_KEY=OPENDICT_API_KEY:latest,NAVER_CLIENT_ID=NAVER_CLIENT_ID:latest,NAVER_CLIENT_SECRET=NAVER_CLIENT_SECRET:latest
```

Later pushes to `main` redeploy automatically and keep these settings.

### Troubleshooting

- **First build fails with `developerconnect.gitRepositoryLinks.fetchReadToken` denied:** wait a minute, then Cloud Build → History → Rebuild (from the course guide).
- **Container crashes or restarts:** memory is still 512 MiB; rerun step 5.
- **Answers say the Gemini model isn't available:** set `GEMINI_MODEL` to the model the course starter uses.
- **Dictionary lookups time out from Cloud Run:** the NIKL servers are in Korea; redeploy the service in `asia-northeast3` (Seoul).
- **Word bank shows "Firestore isn't connected":** check step 1 (database exists) and step 2 (`roles/datastore.user`).

## Project layout

```
app.py              FastAPI app: /chat, word bank API, exports, romanization, TTS
malmoi/agent.py     Gemini function-calling loop, system prompts, sessions
malmoi/tools.py     The 12 tools, their declarations, and the executor
malmoi/             dictionaries.py, naver.py, korean.py, llm.py, storage.py, config.py
static/             index.html, app.css, app.js (no build step)
tests/              offline tests (mocked APIs and a scripted Gemini)
```
