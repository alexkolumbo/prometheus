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
  - re-emits to Hermes as a compliant OpenAI SSE stream, emitting `: ping` heartbeats
    while the (possibly multi-minute) continuation runs so Hermes' 120s stream read
    timeout never fires.

The provider's hard output cap is never hardcoded — we loop while finish == "length".
Everything non-chat is transparently forwarded.
"""
import os
import json
import uuid
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
os.makedirs(LOGDIR, exist_ok=True)

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(None))

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
    status, resp = await engine.call_upstream_json(client, url, headers, data)
    if status != 200:
        log(f"upstream non-200 ({status})")
        return None, None, "error", {}, resp

    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    fr = ch.get("finish_reason")
    usage = resp.get("usage") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

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


async def handle_chat(data: dict, headers: dict):
    wants_stream = bool(data.get("stream"))
    model = data.get("model")

    if not wants_stream:
        content, tool_calls, fr, usage, err = await orchestrate(data, headers)
        if err is not None:
            return JSONResponse(err, status_code=502)
        return JSONResponse(engine.synth_json(content, tool_calls, fr, usage, model))

    async def gen():
        task = asyncio.create_task(orchestrate(data, headers))
        result = None
        while result is None:
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_SECS)
            except asyncio.TimeoutError:
                yield b": ping\n\n"        # SSE comment — keeps Hermes' read timer alive
        content, tool_calls, fr, usage, err = result
        if err is not None:
            for c in engine.synth_sse(f"[prometheus-proxy upstream error] {err}", None, "stop", {}, model):
                yield c
            return
        for c in engine.synth_sse(content, tool_calls, fr, usage, model):
            yield c

    return StreamingResponse(gen(), media_type="text/event-stream")


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
