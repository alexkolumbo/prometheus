"""
Prometheus continuation proxy — OpenAI-compatible shim between Hermes and Gonka.

Milestone 2 (corrected): CONTINUATION over stream=true upstream.

For POST /v1/chat/completions the proxy:
  - calls upstream with stream=true (mirroring Hermes) and reassembles the SSE into
    a response with consistent NATIVE tool_calls (engine.call_upstream_json),
  - on finish_reason == "length":
      * truncated write_file tool-call (native OR textual <tool_call>) -> schema-aware
        content continuation (Approach A) -> one complete native write_file call,
      * truncated plain text -> text continuation,
  - for streaming requests, relays content tokens to the client LIVE as they arrive
    (real streaming) while buffering tool-call deltas so a truncated write_file can be
    rebuilt into one clean call; on a length-truncated plain-text answer it continues
    and streams the appended remainder into the same SSE.

The provider's hard output cap is never hardcoded — we loop while finish == "length".
Everything non-chat is transparently forwarded.
"""
import os
import json
import uuid
import time
import asyncio
import datetime

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

import engine

UPSTREAM = os.environ.get("UPSTREAM_BASE_URL", "https://proxy.gonka.gg").rstrip("/")
PORT = int(os.environ.get("PORT", "8780"))
LOGDIR = os.environ.get("LOGDIR", "/log")
HEARTBEAT_SECS = float(os.environ.get("HEARTBEAT_SECS", "15"))

# ── dynamic output + timeout (all env-configurable) ──────────────────────
# PROM_MAX_TOKENS: how many output tokens we REQUEST upstream per call. We ask
# for a large value on purpose so the provider returns as much as it can in ONE
# call — the only real limiter becomes the provider's CURRENT output cap. Today
# that cap is ~4096 (continuation stitches the rest seamlessly); if/when it is
# raised (e.g. to 16k), a whole ≤16k answer comes back in a single call with
# finish_reason=stop and is no longer split. No cap is ever hardcoded. A small
# client max_tokens never shrinks this — delivering the full answer is the
# proxy's whole job. Set PROM_MAX_TOKENS=0 to disable the boost (pure passthrough).
PROM_MAX_TOKENS = int(os.environ.get("PROM_MAX_TOKENS", "32000"))
# Upstream timeouts (seconds). read = max silence BETWEEN streamed chunks: a big
# but healthy generation streams steadily and is never cut (total time scales
# with output on its own), while a stalled node fails fast so Hermes can retry.
PROM_CONNECT_TIMEOUT = float(os.environ.get("PROM_CONNECT_TIMEOUT", "15"))
PROM_READ_TIMEOUT = float(os.environ.get("PROM_READ_TIMEOUT", "300"))
PROM_WRITE_TIMEOUT = float(os.environ.get("PROM_WRITE_TIMEOUT", "60"))

# ── degeneration guard ───────────────────────────────────────────────────
# Models occasionally collapse into a repetition / CJK-garbage loop on a hard
# prompt (intermittent — the SAME request is usually fine on a re-roll). The
# proxy detects that and RETRIES; retries also nudge sampling to break the loop.
# For streaming we buffer a short prefix and check it before committing, so the
# client never sees the garbage. Set PROM_GUARD_RETRIES=0 to disable.
PROM_GUARD_RETRIES = int(os.environ.get("PROM_GUARD_RETRIES", "2"))
PROM_GUARD_PREFIX = int(os.environ.get("PROM_GUARD_PREFIX_CHARS", "400"))
PROM_RETRY_FREQ_PENALTY = float(os.environ.get("PROM_RETRY_FREQ_PENALTY", "0.5"))
PROM_RETRY_PRESENCE_PENALTY = float(os.environ.get("PROM_RETRY_PRESENCE_PENALTY", "0.5"))
PROM_RETRY_TEMPERATURE = float(os.environ.get("PROM_RETRY_TEMPERATURE", "0.7"))
# Pause between retries so a brief provider blip (a dropped connection for a
# second) is ridden out instead of burning all retries instantly. And show a
# friendly line — not a raw exception — if every attempt still fails.
PROM_RETRY_BACKOFF = float(os.environ.get("PROM_RETRY_BACKOFF", "1.0"))
PROM_ERROR_TEXT = os.environ.get(
    "PROM_ERROR_TEXT", "The model is momentarily unavailable. Please try again in a moment.")

# ── fallback provider (e.g. Grok / xAI) ──────────────────────────────────
# Tried after the PRIMARY upstream has exhausted its retries (provider down,
# overloaded, or still degenerate). Keeps the bot answering when Gonka is out.
# Needs PROM_FALLBACK_KEY to be enabled; the fallback's own output still goes
# through the degeneration guard.
PROM_FALLBACK_URL = os.environ.get("PROM_FALLBACK_URL", "https://api.x.ai/v1").rstrip("/")
PROM_FALLBACK_KEY = os.environ.get("PROM_FALLBACK_KEY", "")
PROM_FALLBACK_MODEL = os.environ.get("PROM_FALLBACK_MODEL", "grok-4")
PROM_FALLBACK_TRIES = int(os.environ.get("PROM_FALLBACK_TRIES", "2"))
# Auth for the fallback: a static key (PROM_FALLBACK_KEY) OR reuse the OAuth token
# the agent already holds. With PROM_FALLBACK_AUTH=oauth we read the current
# access_token out of the agent's auth file (mounted read-only). The agent keeps
# that token refreshed, so the proxy never has to do the OAuth dance itself.
PROM_FALLBACK_AUTH = os.environ.get("PROM_FALLBACK_AUTH", "")   # "" | "oauth"
PROM_FALLBACK_AUTH_FILE = os.environ.get("PROM_FALLBACK_AUTH_FILE", "/hermes-auth/auth.json")
PROM_FALLBACK_PROVIDER = os.environ.get("PROM_FALLBACK_PROVIDER", "xai-oauth")
os.makedirs(LOGDIR, exist_ok=True)


def _upstream_timeout():
    return httpx.Timeout(connect=PROM_CONNECT_TIMEOUT, read=PROM_READ_TIMEOUT,
                         write=PROM_WRITE_TIMEOUT, pool=PROM_CONNECT_TIMEOUT)


def _recent_input(data):
    for m in reversed(data.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c[-2000:]
            if isinstance(c, list):
                return " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))[-2000:]
    return ""


def _build_up_body(data, attempt, is_fallback=False):
    """Upstream body for attempt N. A non-fallback retry (attempt>0) nudges
    sampling to break a degeneration loop; the fallback provider gets a clean
    body (e.g. Grok rejects the penalty params)."""
    b = dict(data)
    client_max = int(data.get("max_tokens") or 0)
    eff_max = max(client_max, PROM_MAX_TOKENS) if PROM_MAX_TOKENS else (client_max or None)
    if eff_max:
        b["max_tokens"] = eff_max
    if attempt > 0 and not is_fallback:
        b["frequency_penalty"] = PROM_RETRY_FREQ_PENALTY
        b["presence_penalty"] = PROM_RETRY_PRESENCE_PENALTY
        b["temperature"] = PROM_RETRY_TEMPERATURE
    return b, eff_max, client_max


def _read_oauth_token():
    """Current access_token for the OAuth fallback provider, from the agent's
    auth file (which the agent keeps refreshed). None if unreadable/missing."""
    try:
        d = json.load(open(PROM_FALLBACK_AUTH_FILE))
        cred = (d.get("providers") or {}).get(PROM_FALLBACK_PROVIDER) or {}
        return (cred.get("tokens") or {}).get("access_token") or cred.get("access_token")
    except Exception as e:
        log(f"oauth fallback token read failed: {e!r}")
        return None


def _fallback_enabled():
    return bool(PROM_FALLBACK_KEY) or PROM_FALLBACK_AUTH == "oauth"


def _total_attempts():
    return (PROM_GUARD_RETRIES + 1) + (PROM_FALLBACK_TRIES if _fallback_enabled() else 0)


def _attempt_target(attempt, headers):
    """(url, headers, model_override, is_fallback) for attempt N: the primary
    upstream first, then the fallback provider once the primary's tries run out."""
    if attempt < PROM_GUARD_RETRIES + 1 or not _fallback_enabled():
        return f"{UPSTREAM}/v1/chat/completions", headers, None, False
    if PROM_FALLBACK_AUTH == "oauth":
        tok = _read_oauth_token()
        if not tok:                       # no token -> just retry the primary
            return f"{UPSTREAM}/v1/chat/completions", headers, None, False
        key = tok
    else:
        key = PROM_FALLBACK_KEY
    fh = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    return f"{PROM_FALLBACK_URL}/chat/completions", fh, PROM_FALLBACK_MODEL, True


app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(None))
# Apply the finite per-chunk read timeout to every upstream stream call.
engine.DEFAULT_TIMEOUT = _upstream_timeout()

HOP = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "te", "trailer", "upgrade", "accept-encoding",
}


def log(msg: str) -> None:
    line = f"[{datetime.datetime.utcnow().isoformat()}Z] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(LOGDIR, "proxy.log"), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def fwd_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "upstream": UPSTREAM, "role": "prometheus-proxy", "milestone": 2}


async def orchestrate(data: dict, headers: dict):
    """Run the (possibly continuing) request. Returns (content, tool_calls, finish, usage, error)."""
    url = f"{UPSTREAM}/v1/chat/completions"
    # Dynamic output + degeneration guard: request a large max_tokens (so the
    # provider returns all it can per call), and if the reply comes back as a
    # repetition/CJK-garbage collapse, retry (a re-roll is almost always clean).
    recent = _recent_input(data)
    resp = ch = msg = None
    total = _total_attempts()
    for attempt in range(total):
        turl, theaders, tmodel, is_fb = _attempt_target(attempt, headers)
        up_body, eff_max, client_max = _build_up_body(data, attempt, is_fb)
        if tmodel:
            up_body["model"] = tmodel
        log(f"upstream call attempt={attempt}{' FALLBACK' if is_fb else ''} "
            f"max_tokens={eff_max} (client asked {client_max or 'none'})")
        status, resp = await engine.call_upstream_json(client, turl, theaders, up_body,
                                                       timeout=_upstream_timeout())
        if status != 200:
            log(f"upstream non-200 ({status}) attempt={attempt}")
            if attempt < total - 1:
                await asyncio.sleep(PROM_RETRY_BACKOFF * (attempt + 1))
                continue
            return PROM_ERROR_TEXT, [], "stop", {}, None
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        if content and engine.is_degenerate(content, recent) and attempt < total - 1:
            log("GUARD degenerate output — retry")
            continue
        url, headers = turl, theaders            # winner — used by any continuation below
        if tmodel:
            data = {**data, "model": tmodel}
        break

    fr = ch.get("finish_reason")
    usage = resp.get("usage") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    if fr == "length":
        ct = (usage or {}).get("completion_tokens")
        if ct:
            log(f"OBSERVED provider per-call output cap ~= {ct} tokens "
                f"(finish=length; will continue)")

    async def reconstruct_write_file(args_or_block):
        path, partial, _ = engine.parse_write_file_args(args_or_block)
        log(f"CONTINUATION write_file path={path!r} partial_chars={len(partial)} — starting")
        full, finished, rounds, clog = await engine.continue_file_content(
            client, url, headers, data, path, partial
        )
        if not path:
            path = await engine.ask_path(client, url, headers, data)
            log(f"  recovered path={path!r}")
        ok, vmsg = engine.verify_python(path, full)
        log(f"CONTINUATION write_file DONE rounds={rounds} finished={finished} "
            f"total_chars={len(full)} verify={vmsg} | {clog}")
        return {
            "id": "call_prom_" + uuid.uuid4().hex[:20],
            "type": "function",
            "function": {"name": "write_file",
                         "arguments": json.dumps({"path": path, "content": full}, ensure_ascii=False)},
        }

    # Detect a truncated write_file by INVALID JSON args, even if finish=tool_calls
    # (providers mislabel a cut tool call as complete).
    wf_trunc = engine.find_truncated_write_file(tool_calls)
    if wf_trunc is not None:
        idx = tool_calls.index(wf_trunc)
        log("truncated write_file detected via invalid-JSON args "
            f"(finish_reason={fr}) — reconstructing")
        tool_calls[idx] = await reconstruct_write_file(
            (wf_trunc.get("function") or {}).get("arguments") or ""
        )
        fr = "tool_calls"
        return content, tool_calls, fr, usage, None

    if fr == "length":
        handled = False
        if tool_calls:
            last = tool_calls[-1]
            name = (last.get("function") or {}).get("name")
            if name == "write_file":
                tool_calls[-1] = await reconstruct_write_file(
                    (last.get("function") or {}).get("arguments") or ""
                )
                fr = "tool_calls"
                handled = True
            else:
                log(f"truncated native tool-call '{name}' (not write_file) — passthrough")
        if not handled and "<tool_call>" in (content or ""):
            block, pos = engine.find_textual_toolcall(content)
            name = engine.extract_tool_name(block or "")
            log(f"textual <tool_call> detected, name={name!r}")
            if name == "write_file":
                preamble = content[:pos].strip()
                tool_calls = [await reconstruct_write_file(block)]
                content = preamble or None
                fr = "tool_calls"
                handled = True
        if not handled and not tool_calls:
            log(f"CONTINUATION text partial_chars={len(content)} — starting")
            full, finished, rounds, clog = await engine.continue_text(
                client, url, headers, data, content
            )
            content = full
            fr = "stop"
            log(f"CONTINUATION text DONE rounds={rounds} finished={finished} "
                f"total_chars={len(full)} | {clog}")

    return content, tool_calls, fr, usage, None


def _chunk(base, delta, finish=None):
    o = dict(base)
    o["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    return engine._sse(o)


def _tool_calls_delta(base, tool_calls):
    tdelta = []
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        tdelta.append({"index": i, "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:20]),
                       "type": "function",
                       "function": {"name": fn.get("name"), "arguments": fn.get("arguments", "")}})
    return _chunk(base, {"tool_calls": tdelta})


async def stream_chat(data: dict, headers: dict, model: str):
    """Live streaming with a degeneration guard and tail-continuation.

    A short content prefix is buffered and checked for degeneration BEFORE we
    commit to the live stream, so a garbage roll is caught and retried without
    the client ever seeing it. Once a clean prefix is committed, tokens flow
    live. Tool-call deltas are buffered so a truncated write_file is rebuilt; on
    a length-truncated plain-text answer we continue and stream the remainder."""
    url = f"{UPSTREAM}/v1/chat/completions"
    recent = _recent_input(data)
    base = {"id": "chatcmpl-prom-" + uuid.uuid4().hex[:24], "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model}
    content_parts, tool_acc, finish, usage = [], {}, None, None
    committed = False

    yield _chunk(base, {"role": "assistant"})

    win = None
    for attempt in range(_total_attempts()):
        turl, theaders, tmodel, is_fb = _attempt_target(attempt, headers)
        last = attempt >= _total_attempts() - 1
        up_body, eff_max, client_max = _build_up_body(data, attempt, is_fb)
        up_body["stream"] = True
        up_body["stream_options"] = {"include_usage": True}
        if tmodel:
            up_body["model"] = tmodel
        log(f"STREAM chat attempt={attempt}{' FALLBACK' if is_fb else ''} "
            f"max_tokens={eff_max} (client asked {client_max or 'none'})")
        if not committed:
            content_parts, tool_acc, finish, usage = [], {}, None, None
        prefix, degenerate = [], False
        try:
            async with client.stream("POST", turl, headers=theaders, json=up_body,
                                     timeout=_upstream_timeout()) as r:
                if r.status_code != 200:
                    raw = (await r.aread()).decode("utf-8", "replace")
                    log(f"STREAM upstream non-200 ({r.status_code}) attempt={attempt} {raw[:160]}")
                    if not committed and not last:
                        await asyncio.sleep(PROM_RETRY_BACKOFF * (attempt + 1))
                        continue
                    if not committed:
                        yield _chunk(base, {"content": PROM_ERROR_TEXT})
                    yield _chunk(base, {}, "stop")
                    yield b"data: [DONE]\n\n"
                    return
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
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        if committed:
                            yield _chunk(base, {"content": piece})            # live
                        else:
                            prefix.append(piece)
                            if sum(len(p) for p in prefix) >= PROM_GUARD_PREFIX:
                                joined = "".join(prefix)
                                if engine.is_degenerate(joined, recent):
                                    degenerate = True
                                    break
                                committed = True
                                yield _chunk(base, {"content": joined})       # flush clean prefix
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
        except Exception as ex:
            log(f"STREAM error: {ex!r} attempt={attempt}")
            if not committed and not last:
                await asyncio.sleep(PROM_RETRY_BACKOFF * (attempt + 1))
                continue
            if not committed:
                yield _chunk(base, {"content": PROM_ERROR_TEXT})
            yield _chunk(base, {}, "stop")
            yield b"data: [DONE]\n\n"
            return

        if degenerate:
            log(f"STREAM GUARD degenerate prefix — retry {attempt + 1}")
            continue
        if not committed:
            short = "".join(content_parts)
            no_output = (not short) and (not tool_acc)
            if not last and (no_output or (short and engine.is_degenerate(short, recent))):
                log(f"STREAM retry attempt {attempt + 1} (empty or degenerate)")
                continue
            committed = True
            if short:
                yield _chunk(base, {"content": short})
        win = (turl, theaders, tmodel)
        break

    if win:
        url, headers = win[0], win[1]
        if win[2]:
            data = {**data, "model": win[2]}
    elif not committed:
        yield _chunk(base, {"content": PROM_ERROR_TEXT})
        yield _chunk(base, {}, "stop")
        yield b"data: [DONE]\n\n"
        return

    content = "".join(content_parts)
    tool_calls = []
    for i in sorted(tool_acc):
        e = tool_acc[i]
        tool_calls.append({"id": e["id"] or ("call_" + uuid.uuid4().hex[:20]), "type": "function",
                           "function": {"name": e["name"], "arguments": e["args"]}})

    async def reconstruct(args_or_block):
        path, partial, _ = engine.parse_write_file_args(args_or_block)
        log(f"STREAM CONTINUATION write_file path={path!r} partial={len(partial)} — starting")
        full, finished, rounds, clog = await engine.continue_file_content(
            client, url, headers, data, path, partial)
        if not path:
            path = await engine.ask_path(client, url, headers, data)
        log(f"STREAM CONTINUATION write_file DONE rounds={rounds} finished={finished} chars={len(full)}")
        return {"id": "call_prom_" + uuid.uuid4().hex[:20], "type": "function",
                "function": {"name": "write_file",
                             "arguments": json.dumps({"path": path, "content": full}, ensure_ascii=False)}}

    wf = engine.find_truncated_write_file(tool_calls)
    if wf is not None:
        tool_calls[tool_calls.index(wf)] = await reconstruct((wf.get("function") or {}).get("arguments") or "")
        yield _tool_calls_delta(base, tool_calls)
        finish = "tool_calls"
    elif finish == "length":
        if tool_calls and (tool_calls[-1].get("function") or {}).get("name") == "write_file":
            tool_calls[-1] = await reconstruct((tool_calls[-1].get("function") or {}).get("arguments") or "")
            yield _tool_calls_delta(base, tool_calls)
            finish = "tool_calls"
        elif tool_calls:
            yield _tool_calls_delta(base, tool_calls)
            finish = "tool_calls"
        elif "<tool_call>" in content and engine.extract_tool_name(
                engine.find_textual_toolcall(content)[0] or "") == "write_file":
            block, _pos = engine.find_textual_toolcall(content)
            yield _tool_calls_delta(base, [await reconstruct(block)])
            finish = "tool_calls"
        else:
            log(f"STREAM CONTINUATION text partial={len(content)} — starting")
            full, finished, rounds, clog = await engine.continue_text(client, url, headers, data, content)
            tail = full[len(content):]
            if tail:
                yield _chunk(base, {"content": tail})
            finish = "stop"
            log(f"STREAM CONTINUATION text DONE rounds={rounds} chars={len(full)}")
    else:
        if tool_calls:
            yield _tool_calls_delta(base, tool_calls)

    yield _chunk(base, {}, finish or "stop")
    if usage:
        u = dict(base); u["choices"] = []; u["usage"] = usage
        yield engine._sse(u)
    yield b"data: [DONE]\n\n"


async def handle_chat(data: dict, headers: dict):
    wants_stream = bool(data.get("stream"))
    model = data.get("model")

    if not wants_stream:
        content, tool_calls, fr, usage, err = await orchestrate(data, headers)
        if err is not None:
            return JSONResponse(err, status_code=502)
        return JSONResponse(engine.synth_json(content, tool_calls, fr, usage, model))

    return StreamingResponse(stream_chat(data, headers, model),
                             media_type="text/event-stream")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    body = await request.body()
    is_chat = request.method == "POST" and path.endswith("chat/completions")

    if is_chat and body:
        try:
            data = json.loads(body)
            log(f"REQ chat model={data.get('model')} stream={bool(data.get('stream'))} "
                f"n_messages={len(data.get('messages') or [])} n_tools={len(data.get('tools') or [])}")
            with open(os.path.join(LOGDIR, "last_request.json"), "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return await handle_chat(data, fwd_headers(request.headers))
        except Exception as e:
            log(f"handle_chat error: {e!r} — falling back to passthrough")

    url = f"{UPSTREAM}/{path}"
    headers = fwd_headers(request.headers)
    r = await client.request(request.method, url, headers=headers, content=body)
    log(f"REQ {request.method} /{path} -> {r.status_code} (passthrough)")
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))
