# prometheus

A little proxy that you put in front of an OpenAI-compatible inference endpoint to get around a hard limit on output length.

It's one of the pieces in [hermes-stack](https://github.com/alexkolumbo/hermes-stack), a one-script setup that wires it up behind Hermes alongside the memory layer and the dashboard — but it works on its own just as well.

Some providers cap how many tokens a single response can contain. Gonka, which is what I built this for, cuts every response off at 4096 tokens. For chat that's fine. The problem shows up the moment an agent tries to write a real file.

## why this exists

When an agent writes a file through a tool call, the entire file ends up inside the `arguments` of that call as a JSON string. So a 600-line source file is one giant `write_file({"path": "...", "content": "...the whole file..."})`. If the file is longer than ~4096 tokens (somewhere around 300-400 lines of code), the JSON gets chopped off in the middle. The agent can't parse it, retries once, gives up, and you end up with a zero-byte file and an "incomplete" error.

So out of the box you basically can't use Gonka for coding agents or anything that produces a large structured output. That's the whole reason this thing exists.

The continuation that some agent frameworks do for plain text doesn't help here, because it only kicks in for ordinary message text, not for tool calls. Tool-call truncation gets one retry and then a refusal.

## what it does

You point your client's `base_url` at the proxy instead of at the provider. Nothing else in your setup changes. For every chat request it:

1. Calls the provider with `stream=true` (same as your client would) and reassembles the stream into one response, so it always works with clean native tool calls.
2. Watches for a truncated answer. That means either `finish_reason == "length"`, or a `write_file` call whose `arguments` aren't valid JSON because they got cut. The second case is sneaky: the provider sometimes still labels it `tool_calls` even though it's broken.
3. If it's truncated, it rebuilds the full output. It hands the model everything written so far and asks it to continue, looping until the file is actually done, then stitches the pieces together (with a check for duplicated text at the seams and an AST parse for Python).
4. Returns a single complete tool call to your client, re-emitted as a normal OpenAI SSE stream. It sends keep-alive pings while it works so a long reconstruction doesn't trip your client's read timeout.

The provider's cap is never written down anywhere in the code. The proxy just keeps going while the output keeps truncating, so if the cap changes (Gonka already moved it from 3k to 4096 at some point) nothing here needs touching. It also doesn't care which model you use, since it only deals with the OpenAI wire format.

One thing worth knowing: clients often send a small fixed `max_tokens` (mine was pinned at 4096 to match the old cap), and if the proxy just forwarded that, it would split answers at 4096 even on a provider that could return far more in one shot. So on the first call the proxy asks for a large output instead (`PROM_MAX_TOKENS`, 32000 by default). The only thing that actually limits a single call is then the provider's real cap. Today that's still 4096 and the continuation does its job; the day the cap goes up to 16k, a 12k answer just comes back whole and nothing gets stitched. Whenever the output does get cut you'll see the observed cap logged, so you can tell what the provider is currently doing.

The upstream timeout is a per-chunk read timeout (`PROM_READ_TIMEOUT`, 300s) rather than a total one. A healthy long generation keeps streaming tokens so it's never cut, while a node that goes silent fails fast and lets the client retry instead of hanging forever.

Rough shape of it:

```
your client  ->  prometheus  ->  provider
                    |
            detects the cut, continues
            the output, returns one
            complete response
```

## running it

You need docker, a running container that talks to an OpenAI-compatible endpoint, and python3 on the host.

```
cd prometheus
./install.sh
```

The installer builds the image, runs the proxy on the same docker network as your target container, backs up the target's config, and switches its `base_url` over to the proxy. It figures out the upstream from whatever the target was already pointing at.

Defaults can be overridden with env vars (`HERMES_CONTAINER`, `PROXY_PORT`, `UPSTREAM_BASE_URL`, and so on). There's a basic health check at `:8780/healthz`, and the runtime log lands in `log/proxy.log` so you can watch reconstructions happen.

To undo everything:

```
./uninstall.sh
```

That points the target back at the provider and removes the proxy container. Config backups are kept next to the original.

## what it's been tested against

Gonka's endpoint (`proxy.gonka.gg/v1`), across Kimi-K2.6, Qwen3-235B and MiniMax-M2.7. The clearest test: asking a coding agent to write a single-file Tetris game. Without the proxy it returns zero bytes. With it you get the whole 500-ish line file, syntactically valid, and the agent moves on like nothing happened.

## things it doesn't handle yet

It only knows how to reconstruct truncated `write_file` calls. If a model decides to express a tool call as a python code fence instead of a real tool call (some do, sometimes), that path falls back to plain text continuation rather than being properly rebuilt. The de-duplication at the seams catches an exact repeat of the last line but won't catch a model that decides to restart a section from scratch. For very large files, slot-filling (write the skeleton first, then fill bodies) would be more reliable than straight continuation. All of that is on the list.

## license

MIT. See LICENSE.
