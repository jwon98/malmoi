/* Malmoi frontend — no build step, no framework. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const HANGUL = /[가-힣]+/g;

/* ------------------------------------------------------------------ helpers */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function h(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

async function api(path, body, method) {
  const opts = { method: method || (body ? "POST" : "GET"), headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Couldn't reach the server. Check your connection and try again.");
  }
  const type = res.headers.get("content-type") || "";
  const data = type.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = typeof data === "object" ? data.detail : data;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${res.status}).`);
  }
  return data;
}

const store = {
  get(k, d) { try { const v = sessionStorage.getItem(k); return v ?? d; } catch { return d; } },
  set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* private mode */ } },
  pref(k, d) { try { const v = localStorage.getItem(k); return v ?? d; } catch { return d; } },
  setPref(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};

/* Minimal, safe markdown: escape first, then format. */
function md(src) {
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const lines = String(src || "").replace(/\r/g, "").split("\n");
  let out = "", para = [], list = null;
  const flushPara = () => { if (para.length) { out += `<p>${para.map(inline).join("<br>")}</p>`; para = []; } };
  const flushList = () => { if (list) { out += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const hd = line.match(/^#{1,4}\s+(.*)$/);
    if (hd) { flushPara(); flushList(); out += `<h4>${inline(hd[1])}</h4>`; }
    else if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
    } else if (!line.trim()) { flushPara(); flushList(); }
    else { flushList(); para.push(line); }
  }
  flushPara(); flushList();
  return out;
}

/* Topics shared by Explore and the drill's topic picker: [Korean, English label, prompt]. */
const TOPICS = [
  ["경제·금융", "Finance and money", "finance, investing, and the economy"],
  ["정치·사회", "Politics and news", "politics, policy, and news coverage"],
  ["과학·기술", "Science and tech", "science, technology, and research"],
  ["연애", "Dating", "dating and relationships"],
  ["직장", "Workplace", "office life and work communication"],
  ["감정", "Feelings and nuance", "subtle emotions and feelings that are hard to translate"],
  ["신조어", "Internet slang", "current internet slang and new words"],
  ["음식", "Food and cooking", "food, cooking, and taste"],
  ["건강", "Health and the body", "health, symptoms, and going to the doctor"],
  ["법률·계약", "Law and contracts", "leases, contracts, and legal language"],
];

let toastTimer = null;
function toast(message, kind = "") {
  const t = $("#toast");
  t.textContent = message;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === "error" ? 6000 : 3000);
}

/* Korean keyboards (IMEs) can fire Enter while a syllable is still being composed,
   which some browsers treat as a form submit. Ignore submits during composition. */
function guardIME(input) {
  input.addEventListener("compositionstart", () => { input.dataset.composing = "1"; });
  input.addEventListener("compositionend", () => { setTimeout(() => { delete input.dataset.composing; }, 40); });
}
const composing = (input) => input.dataset.composing === "1";

/* ------------------------------------------------------------- romanization */

const romCache = new Map();

async function annotate(root) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      if (!/[가-힣]/.test(n.nodeValue)) return NodeFilter.FILTER_REJECT;
      const p = n.parentElement;
      if (!p || p.closest("rt, ruby, pre, code, textarea, input, select, option, [data-noroman], .wongoji")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  if (!nodes.length) return;
  const words = new Set();
  nodes.forEach((n) => (n.nodeValue.match(HANGUL) || []).forEach((w) => words.add(w)));
  const missing = [...words].filter((w) => !romCache.has(w));
  for (let i = 0; i < missing.length; i += 400) {
    try {
      const r = await api("/api/romanize", { items: missing.slice(i, i + 400) });
      Object.entries(r.romanized || {}).forEach(([k, v]) => romCache.set(k, v));
    } catch { /* romanization is optional */ }
  }
  for (const n of nodes) {
    if (!n.parentNode) continue;
    const s = n.nodeValue;
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of s.matchAll(HANGUL)) {
      frag.append(s.slice(last, m.index));
      const ruby = document.createElement("ruby");
      ruby.append(m[0]);
      const rt = document.createElement("rt");
      rt.textContent = romCache.get(m[0]) || "";
      ruby.append(rt);
      frag.append(ruby);
      last = m.index + m[0].length;
    }
    frag.append(s.slice(last));
    n.replaceWith(frag);
  }
}

/* -------------------------------------------------------------------- audio */

let serverTTS = null; // null = untested, true/false after the first try
const audioCache = new Map();

async function speak(text) {
  text = String(text || "").trim();
  if (!text) return;
  if (serverTTS !== false) {
    try {
      let url = audioCache.get(text);
      if (!url) {
        const res = await fetch(`/api/tts?text=${encodeURIComponent(text)}`);
        if (!res.ok) throw new Error("tts");
        url = URL.createObjectURL(await res.blob());
        audioCache.set(text, url);
      }
      serverTTS = true;
      await new Audio(url).play();
      return;
    } catch {
      if (serverTTS === null) serverTTS = false;
    }
  }
  if ("speechSynthesis" in window) {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ko-KR";
    u.rate = 0.9;
    const voice = speechSynthesis.getVoices().find((v) => v.lang && v.lang.startsWith("ko"));
    if (voice) u.voice = voice;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }
}

const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9v6h4l5 4V5L8 9H4z" fill="currentColor"/><path d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>';
const ICON_STAR = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8L3.5 9.7l5.9-.9z" fill="currentColor"/></svg>';
const ICON_STAR_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8L3.5 9.7l5.9-.9z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';
const ICON_CHAT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5h14v10H10l-4 4v-4H5z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';
const ICON_TRASH = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_NOTE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 19l1-4L16 5l3 3L9 18z" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linejoin="round"/></svg>';

function playBtn(text, label = "Play pronunciation") {
  return `<button type="button" class="icon-btn play" data-say="${esc(text)}" aria-label="${esc(label)}" title="${esc(label)}">${ICON_PLAY}</button>`;
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-say]");
  if (b) { e.preventDefault(); speak(b.dataset.say); }
});

/* Headword in 원고지 cells (short words) or plain serif (long ones). */
function headword(word, cls = "headword") {
  const w = String(word || "");
  if ([...w].length > 6) return `<span class="${cls} plain" data-noroman lang="ko">${esc(w)}</span>`;
  const cells = [...w].map((ch) => (ch === " " ? '<span class="gap"></span>' : `<span>${esc(ch)}</span>`)).join("");
  return `<span class="wongoji ${cls}" lang="ko" aria-label="${esc(w)}">${cells}</span>`;
}

function masteryDots(n) {
  return `<span class="mastery" aria-label="Mastery ${n} of 5">${[0, 1, 2, 3, 4].map((i) => `<i class="${i < n ? "on" : ""}"></i>`).join("")}</span>`;
}

function sourceChip(r) {
  if (r.source === "ai" || r.verified === false) return '<span class="chip unverified">Not dictionary-verified</span>';
  if (r.source === "web") return '<span class="chip web">Web-sourced</span>';
  return `<span class="chip verified">${esc(shortSource(r.source_label || r.dictionary || ""))}</span>`;
}
function shortSource(label) {
  if (label.includes("Basic")) return "Basic Korean Dictionary";
  if (label.includes("Standard")) return "Standard Korean Dictionary";
  if (label.includes("Urimalsaem")) return "우리말샘";
  return label || "Dictionary";
}

/* -------------------------------------------------------------------- tabs */

const tabs = $$(".tabs [role=tab]");
const PANELS = ["home", "ask", "drill", "bank", "explore"];
function showTab(name, { push = true } = {}) {
  if (!PANELS.includes(name)) name = "home";
  tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === name)));
  $$("main > .panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  store.set("tab", name);
  if (push && location.hash !== `#${name}`) history.pushState(null, "", `#${name}`);
  if (name === "bank") loadBank();
  if (name === "home") refreshMe();
  if (name === "drill" && typeof prefetchDrill === "function" && !drill.active) prefetchDrill();
  window.scrollTo({ top: 0 });
}
tabs.forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
$("#brand-home").addEventListener("click", (e) => { e.preventDefault(); showTab("home"); });
window.addEventListener("popstate", () => showTab(location.hash.slice(1), { push: false }));
$(".tabs").addEventListener("keydown", (e) => {
  if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  const i = tabs.findIndex((t) => t === document.activeElement);
  const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
  next.focus(); showTab(next.dataset.tab);
});

/* Romanization toggle */
const romToggle = $("#roman-toggle");
romToggle.checked = store.pref("roman", "on") === "on";
document.body.classList.toggle("no-roman", !romToggle.checked);
romToggle.addEventListener("change", () => {
  document.body.classList.toggle("no-roman", !romToggle.checked);
  store.setPref("roman", romToggle.checked ? "on" : "off");
});

/* --------------------------------------------------------------- chat core */

// Session IDs live in memory: reloading the page starts fresh conversations,
// which matches what's on screen. The word bank is stored on the server either way.
const sessions = { ask: null, drill: null, explore: null };

/* Runs one agent turn. Uses /chat/stream to receive live progress ("Checking the
   dictionary for 눈치") and falls back to plain /chat if streaming isn't available.
   Either way the result has the same shape: { response, session_id, tool_calls }. */
async function chat(mode, message, handlers = {}, action = null) {
  if (typeof handlers === "function") handlers = { status: handlers };
  // `action` runs one tool directly (buttons like Start drill). Typed questions
  // leave it null and go through the agent, which decides which tools to use.
  const body = { message, session_id: sessions[mode], mode, ...(action ? { action } : {}) };
  let res;
  try {
    res = await fetch("/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new Error("Couldn't reach the server. Check your connection and try again.");
  }
  let data = null;
  if (res.ok && res.body) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, cut).split("\n").find((l) => l.startsWith("data: "));
        buf = buf.slice(cut + 2);
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "status") handlers.status?.(ev.text);
        else if (ev.type === "text") handlers.text?.(ev.delta);
        else if (ev.type === "text_reset") handlers.reset?.();
        else if (ev.type === "partial") handlers.partial?.(ev);
        else if (ev.type === "done") data = ev;
        else if (ev.type === "error") throw new Error(ev.detail);
      }
    }
    if (!data) throw new Error("The connection was interrupted. Please try again.");
  } else {
    data = await api("/chat", body);
  }
  sessions[mode] = data.session_id;
  return data;
}

const TOOL_NAMES = {
  lookup_word: "Dictionary lookup",
  find_korean_words: "English → Korean options",
  search_slang: "Slang search",
  word_trend: "Search trend",
  check_naturalness: "Usage comparison",
  mine_vocabulary: "Vocabulary scan",
  explore_domain: "Topic vocabulary",
  hanja_family: "Hanja roots",
  start_text_drill: "New drill",
  check_drill_answer: "Answer check",
  get_word_bank: "Word bank",
  update_word: "Word bank update",
};

function argSummary(args) {
  const v = Object.values(args || {}).find((x) => typeof x === "string" && x.trim()) ?? (Array.isArray(Object.values(args || {})[0]) ? Object.values(args)[0].join(", ") : "");
  const s = String(v || "");
  return s.length > 60 ? s.slice(0, 60) + "…" : s;
}

function traceEl(calls) {
  if (!calls || !calls.length) return null;
  const items = calls.map((c) => {
    const failed = c.result && c.result.ok === false;
    return `<li class="${failed ? "failed" : ""}"><details><summary>${esc(TOOL_NAMES[c.name] || c.name)}
      <span class="args" data-noroman>${esc(argSummary(c.args))}</span>${failed ? ' <span class="chip unverified">failed</span>' : ""}</summary>
      <pre data-noroman>${esc(`${c.name}(${JSON.stringify(c.args, null, 2)})\n\n→ ${JSON.stringify(c.result, null, 2)}`)}</pre></details></li>`;
  }).join("");
  return h(`<details class="trace"><summary>${calls.length} tool call${calls.length > 1 ? "s" : ""}</summary><ol class="trace-steps">${items}</ol></details>`);
}

/* ------------------------------------------------------------- tool cards */

const POS_EN = { "명사": "noun", "동사": "verb", "형용사": "adjective", "부사": "adverb", "관형사": "determiner",
  "감탄사": "interjection", "대명사": "pronoun", "수사": "numeral", "의존 명사": "bound noun", "보조 동사": "auxiliary verb",
  "보조 형용사": "auxiliary adjective", "조사": "particle", "어미": "ending", "접사": "affix", "품사 없음": "expression" };

function cardLookup(r) {
  if (!r.ok) return null;
  if (r.found === false) {
    return h(`<div class="card"><p><strong data-noroman>${esc(r.word)}</strong> isn't in the Basic, Standard, or 우리말샘 dictionaries${r.tried_forms && r.tried_forms.length > 1 ? ` (also tried ${esc(r.tried_forms.slice(1).join(", "))})` : ""}.</p></div>`);
  }
  const level = r.level_en && r.level_en !== "unrated" ? `<span class="chip">${esc(r.level_en)}</span>` : "";
  const origin = r.origin ? `<span class="chip hanja" data-noroman>${esc(r.origin)}</span>` : "";
  const pos = r.pos ? `<span class="chip" data-noroman>${esc(POS_EN[r.pos] || r.pos)}</span>` : "";
  const senses = (r.senses || []).map((s) => `<li>
      ${s.english_word ? `<span class="en">${esc(s.english_word)}</span>` : ""}
      ${s.english_definition ? `<div>${esc(s.english_definition)}</div>` : ""}
      ${s.definition_ko ? `<div class="ko" lang="ko">${esc(s.definition_ko)}</div>` : ""}
    </li>`).join("");
  const examples = (r.examples || []).slice(0, 3).map((e) => `<li>${playBtn(e, "Play example")}<span lang="ko">${esc(e)}</span></li>`).join("");
  let seen = "";
  if (r.context_sentence) {
    const ctx = esc(r.context_sentence);
    const stem = esc(r.word.replace(/다$/, ""));
    seen = `<p class="seen-in" lang="ko">${stem && ctx.includes(stem) ? ctx.replace(stem, `<mark>${stem}</mark>`) : ctx}</p>`;
  }
  const card = h(`<div class="card dict">
    <div class="card-head">
      ${headword(r.word)}
      <div class="head-meta">
        <div><span class="rom">${esc(r.romanization || "")}</span> ${playBtn(r.word)}</div>
        <div class="chips">${sourceChip(r)}${level}${pos}${origin}</div>
      </div>
    </div>
    ${seen}
    ${senses ? `<ol class="senses">${senses}</ol>` : ""}
    ${examples ? `<ul class="examples">${examples}</ul>` : ""}
    <div class="card-foot">
      ${r.saved_to_word_bank ? `<span>Saved to your word bank${r.times_looked_up > 1 ? `. Looked up ${r.times_looked_up} times` : ""}</span>` : ""}
      ${r.link ? `<a href="${esc(r.link)}" target="_blank" rel="noopener">Open in dictionary</a>` : ""}
    </div>
  </div>`);
  return card;
}

function cardOptions(r) {
  if (!r.ok || !r.candidates) return null;
  const rows = r.candidates.map((c) => `<li>
    <div class="word">${esc(c.word)} ${playBtn(c.word)}</div>
    <div>
      <div class="chips">
        <span class="chip">${esc(c.register)}</span>
        ${c.verified === true ? '<span class="chip verified">In dictionary</span>' : c.verified === false ? '<span class="chip unverified">Not in dictionaries</span>' : ""}
        ${c.level ? `<span class="chip">${esc(c.level)}</span>` : ""}
      </div>
      <p class="nuance">${esc(c.nuance)}</p>
      ${c.example_ko ? `<p class="ex"><span lang="ko">${esc(c.example_ko)}</span><br>${esc(c.example_en)}</p>` : ""}
    </div>
  </li>`).join("");
  return h(`<div class="card"><ul class="options">${rows}</ul></div>`);
}

function cardSlang(r) {
  if (!r.ok) return null;
  const web = r.web || {};
  const facts = Object.entries(web).map(([k, v]) => `<dt>${esc(k.replace("?", ""))}</dt><dd>${esc(v)}</dd>`).join("");
  const sources = (r.sources || []).map((s) => `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a></li>`).join("");
  return h(`<div class="card">
    <div class="card-head">
      ${headword(r.term)}
      <div class="head-meta">
        <div><span class="rom">${esc(r.romanization || "")}</span> ${playBtn(r.term)}</div>
        <div class="chips">${r.in_dictionary ? '<span class="chip verified">Also in a dictionary</span>' : ""}<span class="chip web">Web-sourced</span></div>
      </div>
    </div>
    ${facts ? `<dl class="facts">${facts}</dl>` : ""}
    ${sources ? `<ul class="sources">${sources}</ul>` : ""}
  </div>`);
}

const SERIES_COLORS = ["var(--pine)", "var(--seal)", "var(--ochre)", "var(--slate)", "var(--plum)"];

function cardTrend(r) {
  if (!r.ok || !r.series || !r.series.length) return null;
  const W = 640, H = 230, L = 34, R = 12, T = 14, B = 30;
  const n = Math.max(...r.series.map((s) => s.points.length));
  const x = (i) => L + (n <= 1 ? 0 : (i * (W - L - R)) / (n - 1));
  const y = (v) => T + (1 - v / 100) * (H - T - B);
  const periods = r.series[0].points.map((p) => p.period);
  const grid = [0, 50, 100].map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)" /><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="10" fill="var(--muted)">${v}</text>`).join("");
  const ticks = [0, Math.floor((n - 1) / 2), n - 1].filter((v, i, a) => a.indexOf(v) === i && periods[v])
    .map((i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="${i === 0 ? "start" : i === n - 1 ? "end" : "middle"}" font-size="10" fill="var(--muted)">${esc(periods[i])}</text>`).join("");
  const lines = r.series.map((s, si) => {
    const pts = s.points.map((p, i) => `${x(i).toFixed(1)},${y(p.ratio).toFixed(1)}`).join(" ");
    const peakIdx = s.points.findIndex((p) => p.period === (s.summary || {}).peak_period);
    const peak = peakIdx >= 0 ? `<circle cx="${x(peakIdx)}" cy="${y(s.points[peakIdx].ratio)}" r="3.5" fill="${SERIES_COLORS[si]}" />` : "";
    return `<polyline points="${pts}" fill="none" stroke="${SERIES_COLORS[si]}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" />${peak}`;
  }).join("");
  const legend = r.series.map((s, si) => `<span><i style="background:${SERIES_COLORS[si]}"></i><span lang="ko">${esc(s.term)}</span></span>`).join("");
  const sums = r.series.map((s) => {
    const m = s.summary || {};
    if (!m.peak_period) return "";
    return `<span lang="ko">${esc(s.term)}</span>: peaked ${esc(m.peak_period)}, now at ${esc(m.latest_vs_peak_pct)}% of its peak and ${esc(m.direction_last_3_periods)}.`;
  }).filter(Boolean).join(" ");
  const label = `Naver search interest for ${r.series.map((s) => s.term).join(", ")}, ${r.start} to ${r.end}`;
  return h(`<div class="card chart">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title>${grid}${lines}${ticks}</svg>
    <div class="legend">${legend}</div>
    <p class="summary">${sums} Relative Naver search interest (100 = highest point in this chart).</p>
  </div>`);
}

function cardNaturalness(r) {
  if (!r.ok || !r.results) return null;
  const max = Math.max(...r.results.map((x) => x.share_pct || 0), 1);
  const rows = r.results.map((x) => `<li>
    <div class="row"><span class="phrase" lang="ko">${esc(x.phrase)}</span><span>${x.error ? esc(x.error) : `${x.share_pct}%`}</span></div>
    <div class="meter"><span style="width:${((x.share_pct || 0) / max) * 100}%"></span></div>
    <p class="snippet">${x.error ? "" : `${(x.blog_results || 0).toLocaleString()} blog and ${(x.news_results || 0).toLocaleString()} news results`}${x.example ? `<br><span lang="ko">${esc(x.example)}</span>` : ""}</p>
  </li>`).join("");
  return h(`<div class="card"><ul class="bars">${rows}</ul></div>`);
}

function cardMined(r) {
  if (!r.ok || !r.words) return null;
  if (!r.words.length) return h(`<div class="card"><p class="muted">${esc(r.note || "No new words found.")}</p></div>`);
  const chips = r.words.map((w) => `<span class="word-chip">
    <span class="ko" data-noroman lang="ko">${esc(w.word)}</span>
    <span class="gl">${esc(w.gloss || w.level || "")}</span>
    ${playBtn(w.word)}
    <button type="button" class="btn small ${w.in_bank ? "" : "primary"}" data-save="${esc(w.word)}" data-gloss="${esc(w.gloss || "")}" ${w.in_bank ? "disabled" : ""}>${w.in_bank ? "Saved" : "Save"}</button>
  </span>`).join("");
  return h(`<div class="card"><div class="mined">${chips}</div></div>`);
}

function cardHanja(r) {
  if (!r.ok) return null;
  if (!r.has_hanja) return h(`<div class="card"><p>${headword(r.word)} <span class="muted">${esc(r.message)}</span></p></div>`);
  const roots = (r.roots || []).map((rt) => `<div class="root">
    <div><span class="char" lang="zh">${esc(rt.hanja)}</span><span class="reading" data-noroman>${esc(rt.reading)}</span></div>
    <div class="muted">${esc(rt.meaning)}</div>
    <ul>${(rt.family || []).map((f) => `<li><span class="ko" lang="ko">${esc(f.word)}</span><span>${esc(f.english)}</span></li>`).join("") || '<li class="muted">No verified related words.</li>'}</ul>
  </div>`).join("");
  const empty = roots ? "" : '<p class="muted">No related words could be verified for these roots.</p>';
  return h(`<div class="card">
    <div class="card-head">${headword(r.word)}<div class="head-meta"><div><span class="rom">${esc(r.romanization || "")}</span> ${playBtn(r.word)}</div><div class="chips"><span class="chip hanja" data-noroman>${esc(r.origin)}</span></div></div></div>
    ${roots ? `<div class="roots">${roots}</div>` : empty}
  </div>`);
}

function cardBank(r) {
  if (!r.ok || !r.words || !r.words.length) return null;
  const rows = r.words.slice(0, 12).map((w) => `<li><div class="word"><span lang="ko">${esc(w.word)}</span></div><div>${esc(w.gloss || "")} ${masteryDots(w.mastery || 0)}</div></li>`).join("");
  return h(`<div class="card"><ul class="options">${rows}</ul></div>`);
}

function wordCard(w) {
  const ko = [...w.word].length <= 5 ? headword(w.word) : `<span class="plain-ko" lang="ko" data-noroman>${esc(w.word)}</span>`;
  return h(`<div class="card word-card">
    <div class="top">${ko}${w.verified === false ? '<span class="chip unverified" title="Not found in the learner\'s dictionary. Suggested by Gemini; double-check before relying on it.">Not in dictionary</span>' : w.level ? `<span class="chip">${esc(w.level)}</span>` : ""}</div>
    <span class="rom">${esc(w.romanization || "")}</span>
    <span class="english">${esc(w.english || w.gloss || "")}</span>
    <p class="why">${esc(w.why_useful || "")}</p>
    ${w.example_ko ? `<p class="ex"><span lang="ko">${esc(w.example_ko)}</span><br><span class="muted">${esc(w.example_en || "")}</span></p>` : ""}
    <div class="actions">${playBtn(w.word)}<button type="button" class="btn small primary" data-save="${esc(w.word)}" data-gloss="${esc(w.english || "")}" data-example="${esc(w.example_ko || "")}">Save</button></div>
  </div>`);
}

function cardExplore(r) {
  if (!r.ok || !r.words) return null;
  const grid = h('<div class="word-grid"></div>');
  r.words.forEach((w) => grid.append(wordCard(w)));
  return grid;
}

const CARD_RENDERERS = {
  lookup_word: cardLookup,
  find_korean_words: cardOptions,
  search_slang: cardSlang,
  word_trend: cardTrend,
  check_naturalness: cardNaturalness,
  mine_vocabulary: cardMined,
  hanja_family: cardHanja,
  explore_domain: cardExplore,
  get_word_bank: cardBank,
};

/* Save buttons on cards (Explore, vocabulary scan) */
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-save]");
  if (!b || b.disabled) return;
  b.disabled = true;
  const label = b.textContent;
  b.textContent = "Saving…";
  try {
    const r = await api("/api/words", { word: b.dataset.save, gloss: b.dataset.gloss || "", example: b.dataset.example || "" });
    b.textContent = "Saved";
    b.classList.remove("primary");
    toast(r.verified === false ? `Saved ${b.dataset.save} (not in the dictionary, so marked unverified)` : `Saved ${b.dataset.save} to your word bank`);
    refreshMe();
  } catch (err) {
    b.textContent = label;
    b.disabled = false;
    toast(err.message, "error");
  }
});

/* -------------------------------------------------------------------- Ask */

const askLog = $("#ask-log");
const askInput = $("#ask-input");
const askForm = $("#ask-form");
let askBusy = false;

function scrollToEnd(el) {
  requestAnimationFrame(() => el.scrollIntoView({ behavior: "smooth", block: "end" }));
}

function thinkingEl(text = "Reading your message") {
  return h(`<div class="msg agent"><div class="thinking"><span class="dots"><span></span><span></span><span></span></span><span class="label">${esc(text)}</span></div></div>`);
}
function setStatus(el, text) {
  const label = el && el.querySelector(".label");
  if (label) label.textContent = text;
}

const askIntroTemplate = $("#ask-intro").cloneNode(true);

async function ask(message) {
  if (askBusy || !message.trim()) return;
  askBusy = true;
  $("#ask-send").disabled = true;
  $("#ask-intro")?.remove();
  $("#ask-tools").hidden = false;
  const userMsg = h(`<div class="msg user"><div class="bubble"></div></div>`);
  userMsg.firstElementChild.textContent = message;
  askLog.append(userMsg);
  annotate(userMsg);
  const wait = thinkingEl();
  askLog.append(wait);
  scrollToEnd(wait);
  // Text streams into `live` as the model writes it. If the model writes a little
  // and then decides to use a tool, that preamble is cleared.
  let live = null;
  let liveText = "";
  const clearLive = () => { live?.remove(); live = null; liveText = ""; wait.hidden = false; };
  try {
    const data = await chat("ask", message, {
      status: (t) => { if (liveText) clearLive(); setStatus(wait, t); },
      reset: clearLive,
      text: (delta) => {
        liveText += delta;
        if (!live) { live = h('<div class="msg agent"><div class="text"></div></div>'); wait.before(live); wait.hidden = true; }
        live.firstElementChild.innerHTML = md(liveText);
      },
    });
    live?.remove();
    wait.hidden = false;
    const msg = h(`<div class="msg agent"><div class="text">${md(data.response)}</div><div class="cards"></div></div>`);
    const cards = $(".cards", msg);
    (data.tool_calls || []).forEach((c) => {
      const fn = CARD_RENDERERS[c.name];
      const el = fn && c.result ? fn(c.result) : null;
      if (el) cards.append(el);
    });
    if (!cards.children.length) cards.remove();
    const trace = traceEl(data.tool_calls);
    if (trace) msg.append(trace);
    wait.replaceWith(msg);
    await annotate(msg);
    if ((data.tool_calls || []).some((c) => ["lookup_word", "search_slang", "update_word"].includes(c.name))) refreshMe();
  } catch (err) {
    live?.remove();
    const msg = h(`<div class="msg agent error"><div class="text"></div></div>`);
    msg.firstElementChild.textContent = err.message;
    wait.hidden = false;
    wait.replaceWith(msg);
  } finally {
    askBusy = false;
    $("#ask-send").disabled = false;
    askInput.focus();
  }
}

askForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = askInput.value;
  askInput.value = "";
  autosize();
  ask(q);
});
askInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); askForm.requestSubmit(); }
});
$("#ask-new").addEventListener("click", () => {
  sessions.ask = null;
  askLog.innerHTML = "";
  const intro = askIntroTemplate.cloneNode(true);
  askLog.append(intro);
  annotate(intro);
  $("#ask-tools").hidden = true;
  askInput.focus();
});
function autosize() { askInput.style.height = "auto"; askInput.style.height = Math.min(askInput.scrollHeight, 220) + "px"; }
askInput.addEventListener("input", autosize);
askLog.addEventListener("click", (e) => {
  const b = e.target.closest(".sample");
  if (b) ask(b.dataset.q);
});

/* ------------------------------------------------------------------ Drill */

const drill = { active: false, busy: false, mode: "produce", data: null };
// Mirrors PERSONAS in malmoi/tools.py so the phone header updates instantly.
const PERSONAS = {
  "close friend": { name: "민지", relationship: "your close friend", level: "Casual (반말)" },
  "coworker": { name: "김 대리", relationship: "a coworker you're friendly with", level: "Polite (해요체)" },
  "boss": { name: "박 팀장님", relationship: "your team lead at work", level: "Formal (합쇼체)" },
  "older relative": { name: "이모", relationship: "your aunt", level: "Polite (해요체)" },
  "crush": { name: "서준", relationship: "someone you've just started talking to (썸)", level: "Polite (해요체)" },
  "stranger": { name: "중고거래 판매자", relationship: "a seller on a secondhand marketplace app", level: "Polite (해요체)" },
};
function setPhoneHeader(personaKey, mode) {
  const p = PERSONAS[personaKey] || PERSONAS["close friend"];
  $("#phone-avatar").textContent = [...p.name][0];
  $("#phone-name").textContent = p.name;
  $("#phone-rel").textContent = p.relationship;
  const lvl = $("#phone-level");
  lvl.hidden = mode !== "produce";
  lvl.textContent = `Reply in ${p.level}`;
}
const thread = $("#drill-thread");
const coach = $("#coach-body");
const drillInput = $("#drill-input");

const drillSource = $("#drill-source");
const drillTopic = $("#drill-topic");
TOPICS.forEach(([, en, prompt]) => drillTopic.append(new Option(en, prompt)));
const SOURCE_HELP = {
  word_bank: "Words due for review come first. If your word bank is empty, Malmoi picks a useful word for you.",
  topic: "Malmoi picks a word about this topic at your level and checks it in the dictionary.",
  random: "Words fluent learners often half-know, like 서운하다 or 눈치.",
};
function syncDrillSource() {
  const src = drillSource.value;
  $("#drill-topic-field").hidden = src !== "topic";
  $("#drill-level-field").hidden = src !== "topic";
  $("#drill-source-help").textContent = SOURCE_HELP[src];
}
drillSource.addEventListener("change", syncDrillSource);
syncDrillSource();
["#drill-source", "#drill-topic", "#drill-level", "#drill-persona"].forEach((sel) => $(sel).addEventListener("change", prefetchDrill));
$$("input[name=drill-mode]").forEach((r) => r.addEventListener("change", prefetchDrill));

$$("input[name=drill-mode]").forEach((r) => r.addEventListener("change", () => {
  $("#drill-mode-help").textContent = r.value === "produce"
    ? "You get a situation in English and reply in Korean using the hidden word."
    : "You get a Korean text that uses the word and explain in English what it means.";
}));

function setDrillControls(on) {
  drillInput.disabled = !on || drill.busy;
  $("#drill-send").disabled = !on || drill.busy;
  $("#drill-hint").disabled = !on || drill.busy;
  $("#drill-reveal").disabled = !on || drill.busy;
  $("#drill-next").disabled = !drill.data || drill.busy;
  drillInput.placeholder = drill.mode === "produce" ? "Reply in Korean…" : "Explain it in English…";
  drillInput.lang = drill.mode === "produce" ? "ko" : "en";
}

function bubble(text, who) {
  const row = h(`<div class="bub-row ${who}"><div class="bub ${who}"></div></div>`);
  row.firstElementChild.textContent = text;
  if (who === "them" && /[가-힣]/.test(text)) row.insertAdjacentHTML("beforeend", playBtn(text, "Play message"));
  thread.append(row);
  annotate(row);
  thread.scrollTop = thread.scrollHeight;
}

function sys(text) {
  const el = h(`<div class="sys"></div>`);
  el.textContent = text;
  thread.append(el);
  thread.scrollTop = thread.scrollHeight;
}

function setCoach(text, extra) {
  coach.innerHTML = "";
  if (text) coach.append(h(`<div class="coach-text">${md(text)}</div>`));
  if (extra) coach.append(extra);
  annotate(coach);
}

const VERDICTS = {
  correct: ["Correct", "정답"],
  alternative: ["Different word, still works", "통과"],
  close: ["Close", "아깝다"],
  miss: ["Not yet", ""],
  partial: ["Partly right", "아깝다"],
  revealed: ["Answer shown", ""],
};

function answerBlock(key) {
  if (!key) return "";
  return `<div class="answer">
    <div class="label">The word</div>
    <div class="ko-line">${headword(key.target_word, "headword")} ${playBtn(key.target_word)}</div>
    <div class="rom">${esc(key.target_romanization || "")}</div>
    ${key.gloss ? `<div>${esc(key.gloss)}</div>` : ""}
    ${key.model_answer_ko ? `<div class="label" style="margin-top:.5rem">A natural reply</div><div class="ko-line"><span lang="ko">${esc(key.model_answer_ko)}</span>${playBtn(key.model_answer_ko)}</div>` : ""}
    ${key.translation_en ? `<div class="label" style="margin-top:.5rem">Translation</div><div>${esc(key.translation_en)}</div>` : ""}
  </div>`;
}

function verdictEl(r) {
  const [title, sealText] = VERDICTS[r.verdict] || [r.verdict, ""];
  const checks = r.verdict === "revealed" ? "" : `<ul class="checks">
    ${r.used_target_word !== undefined ? `<li class="${r.used_target_word ? "yes" : "no"}">${r.used_target_word ? "Used the target word" : "Didn't use the target word"}${r.alternative_word ? ` (used ${esc(r.alternative_word)})` : ""}</li>` : ""}
    ${r.expected_speech_level ? `<li class="${r.speech_level_ok ? "yes" : "no"}">Speech level: ${esc(r.detected_speech_level)}${r.speech_level_ok ? "" : `, expected ${esc(r.expected_speech_level)}`}</li>` : ""}
  </ul>`;
  const better = r.better_version_ko ? `<div class="answer"><div class="label">More natural</div><div class="ko-line"><span lang="ko">${esc(r.better_version_ko)}</span>${playBtn(r.better_version_ko)}</div></div>` : "";
  return h(`<div class="verdict">
    <div class="verdict-head">
      ${sealText ? `<span class="seal big stamp" data-noroman aria-hidden="true">${esc(sealText)}</span>` : ""}
      <div><div class="verdict-title">${esc(title)}</div>
      <div class="verdict-sub">Mastery ${masteryDots(r.mastery_after || 0)}</div></div>
    </div>
    ${checks}
    ${r.feedback_en && r.verdict !== "revealed" ? `<p>${esc(r.feedback_en)}</p>` : ""}
    ${better}
    ${answerBlock(r.answer_key)}
  </div>`);
}

async function drillSend(message, { echo = null, action = null } = {}) {
  if (drill.busy) return;
  drill.busy = true;
  setDrillControls(drill.active);
  if (echo) bubble(echo, "me");
  const typing = h('<div class="bub-row them"><div class="bub them thinking"><span class="dots"><span></span><span></span><span></span></span></div></div>');
  thread.append(typing);
  thread.scrollTop = thread.scrollHeight;
  const status = h('<div class="thinking coach-status"><span class="dots"><span></span><span></span><span></span></span><span class="label">Working on it</span></div>');
  coach.prepend(status);
  try {
    const data = await chat("drill", message, (t) => setStatus(status, t), action);
    typing.remove();
    let verdict = null;
    for (const c of data.tool_calls || []) {
      const r = c.result || {};
      if (c.name === "start_text_drill" && r.ok) { startDrillUI(r); if (r.word_source) sys(r.word_source); }
      if (c.name === "check_drill_answer" && r.ok) {
        verdict = verdictEl(r);
        if (r.partner_reply_ko) bubble(r.partner_reply_ko, "them");
        if (r.drill_status && r.drill_status !== "active") { drill.active = false; sys(r.verdict === "revealed" ? "Answer shown" : "Drill complete"); }
        refreshMe();
      }
      if (c.result && c.result.ok === false) sys(c.result.error);
    }
    setCoach(data.response, verdict);
    const trace = traceEl(data.tool_calls);
    if (trace) coach.append(trace);
  } catch (err) {
    typing.remove();
    status.remove();
    setCoach("", h(`<p class="muted">${esc(err.message)}</p>`));
  } finally {
    drill.busy = false;
    setDrillControls(drill.active);
    if (drill.active) drillInput.focus();
  }
}

function startDrillUI(r) {
  drill.active = true;
  drill.data = r;
  drill.mode = r.mode;
  drill.hintShown = false;
  thread.innerHTML = "";
  setPhoneHeader(r.persona, r.mode);
  $("#drill-task").hidden = false;
  $("#drill-task-text").textContent = r.task_en;
  $("#drill-situation").textContent = r.situation_en;
  (r.partner_message_ko || "").split(/\n+/).filter(Boolean).forEach((m) => bubble(m, "them"));
  prefetchDrill(); // write the next one while this one is being answered
}

/* Ask the server to write a drill for the current settings ahead of time, so
   "Start drill" and "Next word" find one ready. Fire-and-forget. */
function drillSettings() {
  const source = drillSource.value;
  return {
    mode: $("input[name=drill-mode]:checked").value,
    persona: $("#drill-persona").value,
    source,
    topic: source === "topic" ? drillTopic.value : "",
    level: $("#drill-level").value,
  };
}
let prefetchTimer = null;
function prefetchDrill() {
  if ($("#drill-word").value.trim()) return; // a specific word is built on demand
  clearTimeout(prefetchTimer);
  prefetchTimer = setTimeout(() => {
    api("/api/drill/prefetch", { ...drillSettings(), session_id: sessions.drill }).catch(() => {});
  }, 400);
}

$("#drill-start").addEventListener("click", () => startDrill());
$("#drill-next").addEventListener("click", () => startDrill());
async function startDrill(word) {
  const { mode, persona, source, topic, level } = drillSettings();
  const w = (word ?? $("#drill-word").value).trim();
  drill.mode = mode;
  drill.active = false;
  drill.data = null;
  thread.innerHTML = "";
  $("#drill-task").hidden = true;
  setPhoneHeader(persona, mode); // show who you're texting right away
  setCoach("", null);
  const label = w ? `Start a drill with ${w}` : `Start a new drill (${mode === "produce" ? "reply in Korean" : "explain in English"}, ${persona}, words from ${source.replace("_", " ")})`;
  await drillSend(label, { action: { tool: "start_text_drill", args: { mode, persona, source, topic, level, word: w } } });
}
guardIME(drillInput);
$("#drill-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (composing(drillInput)) return;
  const v = drillInput.value.trim();
  if (!v || !drill.active) return;
  drillInput.value = "";
  drillSend(v, { echo: v, action: { tool: "check_drill_answer", args: { reply: v } } });
});
// The hint is written together with the scenario, so it shows instantly.
$("#drill-hint").addEventListener("click", () => {
  const hint = drill.data && drill.data.answer_key && drill.data.answer_key.hint_en;
  if (!hint) { drillSend("Give me a hint, but don't tell me the word."); return; }
  if (!drill.hintShown) { sys(`Hint: ${hint}`); drill.hintShown = true; }
  else toast(hint);
});

// "Show answer" is also instant: the answer key is already here. The server call
// just records the miss for spaced repetition and returns the updated mastery.
$("#drill-reveal").addEventListener("click", async () => {
  if (!drill.data || !drill.active) return;
  const key = drill.data.answer_key;
  drill.active = false;
  setDrillControls(false);
  sys("Answer shown");
  const local = { verdict: "revealed", answer_key: key, mastery_after: 0, feedback_en: "" };
  setCoach("Here's the answer. This word will come back sooner in your reviews.", verdictEl(local));
  try {
    const r = await api("/api/drill/reveal", { session_id: sessions.drill });
    setCoach("Here's the answer. This word will come back sooner in your reviews.", verdictEl(r));
    refreshMe();
  } catch { /* the answer is already on screen */ }
  setDrillControls(false);
});
$("#bank-practice").addEventListener("click", () => {
  drillSource.value = "word_bank";
  syncDrillSource();
  showTab("drill");
  startDrill("");
});

/* ------------------------------------------------------------------- Bank */

const wordList = $("#word-list");
let bankTimer = null;

async function loadBank() {
  const view = $("input[name=bank-view]:checked").value;
  const q = $("#bank-search").value.trim();
  try {
    const data = await api(`/api/words?view=${encodeURIComponent(view)}&q=${encodeURIComponent(q)}`);
    renderStats(data.stats);
    renderWords(data.words, view, q);
  } catch (err) {
    wordList.innerHTML = `<li class="empty">${esc(err.message)}</li>`;
  }
}

function renderStats(s) {
  $("#bank-stats").innerHTML = `
    <div class="stat"><b>${s.total}</b><span>words</span></div>
    <div class="stat"><b>${s.due}</b><span>due for review</span></div>
    <div class="stat"><b>${s.mastered}</b><span>mastered</span></div>`;
  $("#bank-count").textContent = s.total ? String(s.total) : "";
}

const EMPTY_TEXT = {
  recent: "No words yet. Ask about a word and it's saved here automatically, along with the sentence you found it in.",
  due: "Nothing is due for review. Words come back here on a schedule after you practice them.",
  starred: "No starred words. Star the ones you want to focus on.",
  struggling: "No struggling words. Words you miss in drills show up here.",
  mastered: "No mastered words yet. Answer a word correctly in drills a few times to master it.",
  alphabetical: "No words yet.",
};

function fmtDate(iso) {
  if (!iso || iso.startsWith("9999")) return "";
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function wordDetails(w) {
  const rows = [];
  if (w.definition) rows.push(["Meaning", esc(w.definition)]);
  if (w.definition_ko) rows.push(["In Korean", `<span lang="ko">${esc(w.definition_ko)}</span>`]);
  if (w.example) rows.push(["Example", `<span lang="ko">${esc(w.example)}</span> ${playBtn(w.example, "Play example")}`]);
  if ((w.contexts || []).length) rows.push(["Seen in", `<ul>${w.contexts.map((c) => `<li lang="ko">${esc(c)}</li>`).join("")}</ul>`]);
  const review = (w.mastery || 0) >= 5 ? "Marked as known" : w.reviews && fmtDate(w.next_review) ? `Next review ${fmtDate(w.next_review)}` : "";
  rows.push(["Progress", `${masteryDots(w.mastery || 0)} ${w.reviews ? `${w.reviews} drill${w.reviews > 1 ? "s" : ""}` : "Not practiced yet"}${review ? `. ${review}` : ""}`]);
  if (fmtDate(w.created_at)) rows.push(["Added", fmtDate(w.created_at)]);
  return `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
}

function renderWords(words, view, q) {
  wordList.innerHTML = "";
  if (!words.length) {
    wordList.append(h(`<li class="empty"><span class="wongoji" aria-hidden="true"><span></span><span></span><span></span></span><p>${esc(q ? `No saved words match "${q}".` : EMPTY_TEXT[view])}</p>${view === "recent" && !q ? '<button class="btn primary" data-goto="ask">Look up a word</button>' : ""}</li>`));
    return;
  }
  for (const w of words) {
    const mastered = (w.mastery || 0) >= 4;
    const li = h(`<li class="word-row">
      <div class="word-main" role="button" tabindex="0" aria-expanded="false" title="Show details">
        <span class="word-ko" lang="ko" data-noroman>${esc(w.word)}</span>
        <div class="word-info">
          <div><span class="rom">${esc(w.romanization || "")}</span></div>
          <div class="gloss">${esc(w.gloss || w.definition || "")}</div>
          <div class="chips">${w.level ? `<span class="chip">${esc(w.level)}</span>` : ""}${w.source === "web" ? '<span class="chip web">Web-sourced</span>' : w.source === "ai" ? '<span class="chip unverified">Unverified</span>' : ""}${w.origin ? `<span class="chip hanja" data-noroman>${esc(w.origin)}</span>` : ""}</div>
          ${(w.contexts || [])[0] ? `<p class="seen" lang="ko" title="${esc(w.contexts[0])}">${esc(w.contexts[0])}</p>` : ""}
          ${w.note ? `<p class="note">${esc(w.note)}</p>` : ""}
        </div>
      </div>
      <div class="word-side">
        ${mastered ? '<span class="seal" data-noroman title="Mastered">익힘</span>' : masteryDots(w.mastery || 0)}
        ${playBtn(w.word)}
        <button type="button" class="icon-btn" data-practice="${esc(w.word)}" aria-label="Practice ${esc(w.word)} in a text drill" title="Practice in a text drill">${ICON_CHAT}</button>
        <button type="button" class="icon-btn ${w.starred ? "on" : ""}" data-star="${esc(w.word)}" aria-pressed="${!!w.starred}" aria-label="${w.starred ? "Unstar" : "Star"} ${esc(w.word)}" title="${w.starred ? "Starred" : "Star"}">${w.starred ? ICON_STAR : ICON_STAR_OFF}</button>
        <button type="button" class="icon-btn" data-note="${esc(w.word)}" aria-label="Edit note for ${esc(w.word)}" title="Add a note">${ICON_NOTE}</button>
        <button type="button" class="icon-btn" data-delete="${esc(w.word)}" aria-label="Delete ${esc(w.word)}" title="Delete">${ICON_TRASH}</button>
      </div>
      <div class="word-details" hidden>${wordDetails(w)}</div>
    </li>`);
    wordList.append(li);
  }
  annotate(wordList);
}

function toggleDetails(main) {
  const details = main.closest(".word-row").querySelector(".word-details");
  const open = details.hidden;
  details.hidden = !open;
  main.setAttribute("aria-expanded", String(open));
}
wordList.addEventListener("keydown", (e) => {
  const main = e.target.closest(".word-main");
  if (main && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggleDetails(main); }
});

wordList.addEventListener("click", async (e) => {
  const practice = e.target.closest("[data-practice]");
  if (practice) { showTab("drill"); startDrill(practice.dataset.practice); return; }
  const main = e.target.closest(".word-main");
  if (main && !e.target.closest("button")) { toggleDetails(main); return; }
  const star = e.target.closest("[data-star]");
  const del = e.target.closest("[data-delete]");
  const note = e.target.closest("[data-note]");
  try {
    if (star) {
      const on = star.getAttribute("aria-pressed") !== "true";
      await api(`/api/words/${encodeURIComponent(star.dataset.star)}`, { starred: on }, "PATCH");
      loadBank();
    } else if (del) {
      if (!confirm(`Delete ${del.dataset.delete} from your word bank?`)) return;
      await api(`/api/words/${encodeURIComponent(del.dataset.delete)}`, undefined, "DELETE");
      toast(`Deleted ${del.dataset.delete}`);
      loadBank();
    } else if (note) {
      const text = prompt(`Note for ${note.dataset.note}:`);
      if (text === null) return;
      await api(`/api/words/${encodeURIComponent(note.dataset.note)}`, { note: text }, "PATCH");
      loadBank();
    }
  } catch (err) {
    toast(err.message, "error");
  }
});
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-goto]");
  if (g) { showTab(g.dataset.goto); if (g.dataset.goto === "ask") askInput.focus(); }
});
$$("input[name=bank-view]").forEach((r) => r.addEventListener("change", loadBank));
$("#bank-search").addEventListener("input", () => { clearTimeout(bankTimer); bankTimer = setTimeout(loadBank, 250); });

/* ---------------------------------------------------------------- Explore */

const topicsEl = $("#topics");
TOPICS.forEach(([ko, en, prompt]) => {
  const b = h(`<button type="button" class="topic" aria-pressed="false"><span class="ko" lang="ko" data-noroman>${esc(ko)}</span><span class="en">${esc(en)}</span></button>`);
  b.addEventListener("click", () => {
    $$(".topic").forEach((t) => t.setAttribute("aria-pressed", "false"));
    b.setAttribute("aria-pressed", "true");
    explore(prompt, en);
  });
  topicsEl.append(b);
});
guardIME($("#topic-input"));
$("#topic-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (composing($("#topic-input"))) return;
  const v = $("#topic-input").value.trim();
  if (v) { $$(".topic").forEach((t) => t.setAttribute("aria-pressed", "false")); explore(v, v); }
});

let exploreBusy = false;

async function explore(category, label, { more = false } = {}) {
  if (exploreBusy) return;
  exploreBusy = true;
  const level = more ? explore.current.level : $("input[name=explore-level]:checked").value;
  explore.current = { category, label, level };
  const out = $("#explore-results");
  if (!more) {
    out.innerHTML = "";
    out.append(h('<div class="explore-intro"></div>'), h('<div class="word-grid"></div>'));
  }
  $(".explore-more", out)?.remove();
  const grid = $(".word-grid", out);
  const wait = thinkingEl(`Finding ${more ? "more " : ""}${level} words for ${label.toLowerCase()}`);
  grid.after(wait);
  const onScreen = new Set($$(".word-card", grid).map((c) => c.dataset.word));
  const addCards = (words) => words.forEach((w) => {
    if (onScreen.has(w.word)) return;
    onScreen.add(w.word);
    const card = wordCard(w);
    card.dataset.word = w.word;
    grid.append(card);
    annotate(card);
  });
  try {
    const msg = more
      ? `Show me more words. category="${category}"; level=${level}; count=9`
      : `Explore vocabulary. category="${category}"; level=${level}; count=9`;
    const data = await chat("explore", msg, {
      status: (t) => setStatus(wait, t),
      partial: (ev) => { if (ev.tool === "explore_domain") addCards(ev.words || []); },
    }, { tool: "explore_domain", args: { category, level, count: 9 } });
    wait.remove();
    const call = (data.tool_calls || []).find((c) => c.name === "explore_domain");
    const result = call && call.result && call.result.ok ? call.result : null;
    if (result) addCards(result.words); // anything that didn't arrive as a partial
    if (!more) {
      const intro = $(".explore-intro", out);
      intro.innerHTML = md(data.response);
      annotate(intro);
    }
    const count = result ? result.words.length : 0;
    const unverified = result ? result.unverified_count || 0 : 0;
    const footer = h(`<div class="explore-more">
      ${count ? '<button type="button" class="btn">Show more words</button>' : ""}
      <p>${!result ? esc(data.response) : !count ? "No new words left for this topic at this level. Try another level or topic." : unverified ? `${unverified} of these weren't in the learner's dictionary, so they're marked “Not in dictionary.” Double-check those before relying on them.` : "Every word here was checked in the dictionary."}</p>
    </div>`);
    $("button", footer)?.addEventListener("click", () => explore(category, label, { more: true }));
    grid.after(footer);
    const trace = traceEl(data.tool_calls);
    if (trace) {
      let traces = $(".explore-traces", out);
      if (!traces) { traces = h('<div class="explore-traces"></div>'); out.append(traces); }
      traces.append(trace);
    }
  } catch (err) {
    wait.remove();
    grid.after(h(`<p class="muted explore-more">${esc(err.message)}</p>`));
  } finally {
    exploreBusy = false;
  }
}

/* Pre-build Explore word lists for every level, and the answers to the sample
   questions, so clicking them is fast. The server caches results in Firestore for
   60 days, so after the first visit these requests just read the cache. Levels go
   one at a time, starting with whichever one is selected. */
const SAMPLES = $$(".sample").map((b) => b.dataset.q);
const warmed = new Set();
function warmTopics(level) {
  if (warmed.has(level)) return Promise.resolve();
  warmed.add(level);
  return api("/api/warm", { level, topics: TOPICS.map((t) => t[2]), samples: level === "advanced" ? SAMPLES : [] })
    .catch(() => warmed.delete(level));
}
async function warmEverything() {
  const selected = $("input[name=explore-level]:checked").value;
  const order = ["advanced", selected, "upper-intermediate", "native-level"];
  for (const level of [...new Set(order)]) await warmTopics(level);
}
$$("input[name=explore-level]").forEach((r) => r.addEventListener("change", () => warmTopics(r.value)));

/* ---------------------------------------------------------------- status */

async function refreshMe() {
  try {
    const me = await api("/api/me");
    $("#bank-count").textContent = me.stats.total ? String(me.stats.total) : "";
    renderHomeProgress(me.stats);
    const notes = [];
    const f = me.features;
    if (!f.krdict && !f.stdict && !f.opendict) notes.push("No dictionary API keys are configured, so definitions come from Gemini and are marked unverified. Add KRDICT_API_KEY to enable dictionary lookups.");
    const n = $("#setup-notice");
    n.hidden = !notes.length;
    n.textContent = notes.join(" ");
    const sn = $("#storage-note");
    sn.hidden = me.storage !== "memory";
    sn.textContent = "Firestore isn't connected, so this word bank only lasts until the server restarts.";
  } catch { /* status is best-effort */ }
}

function renderHomeProgress(s) {
  const el = $("#home-progress");
  if (!s.total) {
    el.innerHTML = `<p>Your word bank is empty. Start with a word you ran into today: paste the sentence on the Ask tab, or try a drill with random useful words.</p>
      <button class="btn primary" data-goto="ask">Look up your first word</button>`;
    return;
  }
  const action = s.due
    ? `<button class="btn primary" id="home-review">Review ${s.due} due word${s.due > 1 ? "s" : ""}</button>`
    : '<button class="btn" data-goto="bank">Open your word bank</button>';
  el.innerHTML = `<div class="stats">
      <div class="stat"><b>${s.total}</b><span>words saved</span></div>
      <div class="stat"><b>${s.due}</b><span>due for review</span></div>
      <div class="stat"><b>${s.mastered}</b><span>mastered</span></div>
    </div>${action}`;
  $("#home-review")?.addEventListener("click", () => $("#bank-practice").click());
}

/* ------------------------------------------------------------------ init */

showTab(location.hash.slice(1) || store.get("tab", "home"), { push: false });
warmEverything();
annotate($("#samples"));
annotate($(".home-notes"));
annotate($(".steps"));
refreshMe();
if ("speechSynthesis" in window) speechSynthesis.getVoices();
