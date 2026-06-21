"""
Prometheus continuation engine (Milestone 2).

Sits inside the proxy. For each chat/completions request from Hermes:
  - call upstream (Gonka) with stream=false (easy to inspect finish_reason/tool_calls),
  - if the response is truncated (finish_reason == "length"):
      * truncated write_file tool-call  -> reconstruct the FULL file content via
        schema-aware content continuation (Approach A), then synthesize one complete
        write_file tool-call,
      * truncated plain text            -> text continuation (anchor + dedup),
  - re-emit the assembled result to Hermes as a compliant OpenAI SSE stream
    (Hermes always consumes via the OpenAI SDK with stream=True).

Dynamic cap: we never hardcode the provider cap. We simply loop while
finish_reason == "length"; the cap (whatever it currently is — 4096 today) is
irrelevant to correctness.
"""
import ast
import json
import re
import time
import uuid

EOF_MARKER = "<<<PROM_EOF>>>"
MAX_ROUNDS = 80          # safety ceiling on continuation rounds
MAX_TOTAL_CHARS = 4_000_000

# Per-chunk read timeout for upstream streams. Set by app at startup (None =
# the httpx client default, which is unbounded). A finite value fails fast on a
# stalled node while never cutting a healthy generation (read resets per chunk).
DEFAULT_TIMEOUT = None


# ─────────────────────────── upstream helper ───────────────────────────
async def call_upstream_json(client, url, headers, body, timeout=None):
    """Call upstream with stream=TRUE (mirroring Hermes) and REASSEMBLE the SSE
    into a non-streaming response dict. stream=true gives consistent NATIVE
    tool_calls; stream=false yields messy textual <tool_call>/python-fence formats.
    Returns (status, response_dict) where response_dict has the usual shape:
      {"choices":[{"index":0,"message":{content,tool_calls?},"finish_reason":..}],"usage":..}
    """
    b = dict(body)
    b["stream"] = True
    b["stream_options"] = {"include_usage": True}
    to = timeout if timeout is not None else DEFAULT_TIMEOUT

    content_parts = []
    tool_acc = {}          # index -> {"id","name","args"}
    finish = None
    usage = None
    try:
        async with client.stream("POST", url, headers=headers, json=b, timeout=to) as r:
            if r.status_code != 200:
                raw = await r.aread()
                try:
                    return r.status_code, json.loads(raw)
                except Exception:
                    return r.status_code, {"_raw": raw.decode("utf-8", "replace")}
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    continue
                try:
                    obj = json.loads(d)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                c0 = choices[0]
                delta = c0.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index") if tc.get("index") is not None else 0
                    e = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        e["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        e["name"] = fn["name"]
                    if fn.get("arguments"):
                        e["args"] += fn["arguments"]
                if c0.get("finish_reason"):
                    finish = c0["finish_reason"]
    except Exception as e:
        return 599, {"_error": repr(e)}

    tool_calls = None
    if tool_acc:
        tool_calls = []
        for i in sorted(tool_acc):
            e = tool_acc[i]
            tool_calls.append({
                "id": e["id"] or ("call_" + uuid.uuid4().hex[:20]),
                "type": "function",
                "function": {"name": e["name"], "arguments": e["args"]},
            })
    msg = {"role": "assistant", "content": "".join(content_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    resp = {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage or {}}
    return 200, resp


# ─────────────────────── partial-JSON arg parsing ──────────────────────
def _decode_partial_json_string(after_quote: str) -> str:
    """Decode a JSON string value that may be truncated mid-way.

    `after_quote` is everything after the opening quote of the value.
    We strip a trailing incomplete escape, close the quote, and json.loads it;
    on failure we trim from the end until it decodes.
    """
    s = after_quote
    # remove a trailing incomplete \uXXXX
    s = re.sub(r'\\u[0-9a-fA-F]{0,3}$', '', s)
    # remove a dangling odd backslash (incomplete escape)
    m = re.search(r'\\+$', s)
    if m and (len(m.group(0)) % 2 == 1):
        s = s[:-1]
    try:
        return json.loads('"' + s + '"')
    except Exception:
        for cut in range(1, min(12, len(s)) + 1):
            try:
                return json.loads('"' + s[:-cut] + '"')
            except Exception:
                continue
    return ""


def find_truncated_write_file(tool_calls):
    """Return a write_file tool_call whose arguments are INVALID JSON (truncated),
    regardless of finish_reason. Providers sometimes report finish=tool_calls on a
    tool call that was actually cut at the output cap, leaving unparseable args."""
    for tc in (tool_calls or []):
        fn = tc.get("function") or {}
        if fn.get("name") == "write_file":
            try:
                json.loads(fn.get("arguments") or "")
            except Exception:
                return tc
    return None


def is_degenerate(text, recent_input=""):
    """Heuristic detector for model degeneration (repetition collapse / language
    leak) so the proxy can RETRY instead of relaying garbage to the client.

    Two signals: (1) a flood of CJK characters in the output while the request
    itself wasn't CJK (the classic Kimi/Qwen 'Chinese garbage' leak), and
    (2) a collapsed unique-token ratio over many tokens (a repeated-word loop)."""
    t = text or ""
    if len(t) < 80:
        return False
    cjk = sum(1 for x in t if "一" <= x <= "鿿")
    ri = recent_input or ""
    in_frac = (sum(1 for x in ri if "一" <= x <= "鿿") / len(ri)) if ri else 0.0
    if cjk >= 20 and cjk / len(t) > 0.20 and in_frac < 0.10:
        return True
    toks = t.split()
    if len(toks) >= 60 and len(set(toks)) / len(toks) < 0.18:
        return True
    return False


def extract_tool_name(s: str):
    m = re.search(r'"name"\s*:\s*"([^"]+)"', s or "")
    return m.group(1) if m else None


def find_textual_toolcall(content: str):
    """Return the JSON block after a (possibly truncated) <tool_call> tag, or None."""
    i = (content or "").find("<tool_call>")
    if i == -1:
        return None, 0
    block = content[i + len("<tool_call>"):]
    j = block.find("</tool_call>")
    if j != -1:
        block = block[:j]
    return block.strip(), i


def parse_write_file_args(args_str: str):
    """Return (path, content, complete). `complete` True if args parsed cleanly."""
    try:
        d = json.loads(args_str)
        return d.get("path"), d.get("content", "") or "", True
    except Exception:
        pass
    path = None
    mp = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', args_str)
    if mp:
        try:
            path = json.loads('"' + mp.group(1) + '"')
        except Exception:
            path = mp.group(1)
    content = ""
    mc = re.search(r'"content"\s*:\s*"', args_str)
    if mc:
        content = _decode_partial_json_string(args_str[mc.end():])
    return path, content, False


# ─────────────────────────── splice helpers ────────────────────────────
def _strip_code_fences(text: str) -> str:
    """If the whole piece is wrapped in a single ``` fence, unwrap it."""
    t = text.strip("\n")
    if t.startswith("```"):
        # drop first fence line
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t


def dedup_overlap(full: str, piece: str, max_overlap: int = 4000) -> str:
    """Trim the largest prefix of `piece` that duplicates the tail of `full`."""
    if not full or not piece:
        return piece
    tail = full[-max_overlap:]
    maxk = min(len(tail), len(piece))
    for k in range(maxk, 0, -1):
        if tail[-k:] == piece[:k]:
            return piece[k:]
    return piece


def verify_python(path: str, content: str):
    """Best-effort AST check for .py; returns (ok, msg)."""
    if not (path or "").endswith(".py"):
        return True, "not-python"
    try:
        ast.parse(content)
        return True, "AST OK"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


# ───────────────────── content continuation (Approach A) ───────────────
async def continue_file_content(client, url, headers, base_body, path, partial):
    """Continue raw file content until complete. Returns (content, finished, rounds, log)."""
    full = partial or ""
    log = []
    finished = False
    rounds = 0
    empty_streak = 0
    model = base_body.get("model")
    while rounds < MAX_ROUNDS and len(full) < MAX_TOTAL_CHARS:
        rounds += 1
        # Give the model the FULL content written so far (not just a short anchor)
        # so it KNOWS what it already wrote and won't redeclare/restart earlier code.
        # For very large files, keep the head (declarations) + a generous tail.
        if len(full) <= 120_000:
            context = full
        else:
            context = full[:8_000] + "\n...[earlier content omitted for brevity]...\n" + full[-90_000:]
        sys_msg = {
            "role": "system",
            "content": (
                "You are a file-content continuation engine. You are given the BEGINNING "
                "of a file that was cut off mid-output. Output ONLY the raw text that comes "
                "AFTER what is shown — the missing remainder. Your first character must be "
                "the one that immediately follows the last shown character. Do NOT repeat or "
                "restart ANY content already shown (no re-declaring variables/functions, no "
                "re-emitting earlier lines). Do NOT add explanations or markdown code fences. "
                f"When the file is fully complete, append the exact marker {EOF_MARKER}."
            ),
        }
        usr_msg = {
            "role": "user",
            "content": (
                f"File path: {path}\n"
                f"--- FILE CONTENT SO FAR (do NOT repeat any of this) ---\n"
                f"{context}\n"
                f"--- END OF PARTIAL CONTENT ---\n"
                "Output only the remaining content that continues from exactly the last "
                f"character above. Finish with {EOF_MARKER} when the whole file is done."
            ),
        }
        body = {"model": model, "messages": [sys_msg, usr_msg],
                "temperature": 0, "max_tokens": 100000}
        status, resp = await call_upstream_json(client, url, headers, body)
        ch = (resp.get("choices") or [{}])[0]
        piece = ((ch.get("message") or {}).get("content")) or ""
        fr = ch.get("finish_reason")
        piece = _strip_code_fences(piece)
        had_eof = EOF_MARKER in piece
        if had_eof:
            piece = piece.split(EOF_MARKER)[0]
        piece = dedup_overlap(full, piece)
        # An empty piece means the upstream round errored or stalled — retry a few
        # times instead of silently returning a truncated result.
        if not piece.strip() and not had_eof:
            empty_streak += 1
            log.append(f"round {rounds}: EMPTY (finish={fr}, status={status}) retry {empty_streak}/3")
            if empty_streak >= 3:
                break  # finished stays False — caller sees an incomplete reconstruction
            continue
        empty_streak = 0
        full += piece
        log.append(f"round {rounds}: +{len(piece)} chars finish={fr} eof={had_eof}")
        if had_eof:
            finished = True
            break
        if fr != "length":
            finished = True
            break
    return full, finished, rounds, log


# ───────────────────────── text continuation ──────────────────────────
async def continue_text(client, url, headers, base_body, partial):
    """Continue a truncated plain-text answer. Returns (text, finished, rounds, log)."""
    full = partial or ""
    log = []
    finished = False
    rounds = 0
    model = base_body.get("model")
    msgs = list(base_body.get("messages") or [])
    while rounds < MAX_ROUNDS and len(full) < MAX_TOTAL_CHARS:
        rounds += 1
        cont_msgs = msgs + [
            {"role": "assistant", "content": full},
            {"role": "user", "content": (
                "Your previous response was truncated by the output length limit. "
                "Continue EXACTLY where you left off. Do not restart or repeat prior "
                "text. Finish the answer directly."
            )},
        ]
        body = {"model": model, "messages": cont_msgs, "temperature": 0,
                "max_tokens": 100000}
        status, resp = await call_upstream_json(client, url, headers, body)
        ch = (resp.get("choices") or [{}])[0]
        piece = ((ch.get("message") or {}).get("content")) or ""
        fr = ch.get("finish_reason")
        piece = dedup_overlap(full, piece)
        full += piece
        log.append(f"round {rounds}: +{len(piece)} chars finish={fr}")
        if fr != "length":
            finished = True
            break
    return full, finished, rounds, log


async def ask_path(client, url, headers, base_body):
    """Best-effort recovery of the intended file path when it truncated before
    being emitted (content-first truncation)."""
    model = base_body.get("model")
    msgs = list(base_body.get("messages") or [])
    q = {"role": "user", "content": (
        "Reply with ONLY the absolute filesystem path of the file you are creating "
        "in your write_file call — nothing else, no quotes, no explanation."
    )}
    body = {"model": model, "messages": msgs + [q], "temperature": 0, "max_tokens": 200}
    try:
        status, resp = await call_upstream_json(client, url, headers, body)
        txt = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        m = re.search(r'(/[^\s"\']+)', txt)
        return m.group(1) if m else (txt or None)
    except Exception:
        return None


# ─────────────────────────── SSE synthesis ─────────────────────────────
def _sse(obj) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")


def synth_sse(content, tool_calls, finish_reason, usage, model):
    """Yield a compliant OpenAI chat.completion.chunk SSE sequence."""
    cid = "chatcmpl-prom-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}

    # role delta
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    yield _sse(first)

    if content:
        c = dict(base)
        c["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
        yield _sse(c)

    if tool_calls:
        tdelta = []
        for i, tc in enumerate(tool_calls):
            tdelta.append({
                "index": i,
                "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:20]),
                "type": "function",
                "function": {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments", ""),
                },
            })
        t = dict(base)
        t["choices"] = [{"index": 0, "delta": {"tool_calls": tdelta}, "finish_reason": None}]
        yield _sse(t)

    fin = dict(base)
    fin["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
    yield _sse(fin)

    if usage:
        u = dict(base)
        u["choices"] = []
        u["usage"] = usage
        yield _sse(u)

    yield b"data: [DONE]\n\n"


def synth_json(content, tool_calls, finish_reason, usage, model):
    """Non-streaming JSON response (when Hermes did not request streaming)."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = content or None
    return {
        "id": "chatcmpl-prom-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": usage or {},
    }
