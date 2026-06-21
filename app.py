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
os.makedirs(LOGDIR, exist_ok=True)


def _upstream_timeout():
    return httpx.Timeout(connect=PROM_CONNECT_TIMEOUT, read=PROM_READ_TIMEOUT,
                         write=PROM_WRITE_TIMEOUT, pool=PROM_CONNECT_TIMEOUT)


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
    # Dynamic output: request a large max_tokens so the provider returns all it
    # can in one call. Only the provider's real cap limits us — when it rises,
    # answers stop being split. (PROM_MAX_TOKENS=0 disables the boost.)
    up_body = dict(data)
    client_max = int(data.get("max_tokens") or 0)
    eff_max = max(client_max, PROM_MAX_TOKENS) if PROM_MAX_TOKENS else (client_max or None)
    if eff_max:
        up_body["max_tokens"] = eff_max
    log(f"first upstream call: max_tokens={eff_max} (client asked {client_max or 'none'})")
    status, resp = await engine.call_upstream_json(client, url, headers, up_body,
                                                   timeout=_upstream_timeout())
    if status != 200:
        log(f"upstream non-200 ({status})")
        return None, None, "error", {}, resp

    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
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
    """Live streaming pass-through with tail-continuation.

    Content tokens are relayed to the client AS they arrive (real streaming).
    Tool-call deltas are buffered (not streamed) so a truncated write_file can be
    rebuilt into one clean call before emitting. On finish_reason==length for
    plain text we continue and stream the appended remainder into the same SSE."""
    url = f"{UPSTREAM}/v1/chat/completions"
    up_body = dict(data)
    up_body["stream"] = True
    up_body["stream_options"] = {"include_usage": True}
    client_max = int(data.get("max_tokens") or 0)
    eff_max = max(client_max, PROM_MAX_TOKENS) if PROM_MAX_TOKENS else (client_max or None)
    if eff_max:
        up_body["max_tokens"] = eff_max
    log(f"STREAM chat: max_tokens={eff_max} (client asked {client_max or 'none'})")

    base = {"id": "chatcmpl-prom-" + uuid.uuid4().hex[:24], "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model}
    content_parts, tool_acc, finish, usage = [], {}, None, None

    yield _chunk(base, {"role": "assistant"})

    try:
        async with client.stream("POST", url, headers=headers, json=up_body,
                                 timeout=_upstream_timeout()) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode("utf-8", "replace")
                log(f"STREAM upstream non-200 ({r.status_code})")
                yield _chunk(base, {"content": f"[prometheus upstream {r.status_code}] {raw[:400]}"})
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
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    yield _chunk(base, {"content": delta["content"]})   # LIVE
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
        log(f"STREAM error: {ex!r}")
        yield _chunk(base, {"content": f"\n[prometheus stream error] {ex!r}"})
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
