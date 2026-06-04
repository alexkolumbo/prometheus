#!/usr/bin/env bash
#
# Prometheus continuation proxy — installer.
#
# Deploys a transparent OpenAI-compatible proxy between a Hermes Agent and its
# upstream LLM provider, then repoints Hermes through it. The proxy detects
# output-cap truncation (finish_reason=length OR a write_file tool-call with
# invalid-JSON arguments) and transparently reconstructs the full output via
# content-continuation, so Hermes can write files larger than the provider's
# hard output cap. Model-agnostic (works on the OpenAI wire protocol).
#
# Requirements: docker, a running Hermes container, bash, python3.
# Safe to re-run (idempotent). Creates config backups. See README.md.
#
# Configurable via env vars (with sensible defaults):
#   HERMES_CONTAINER   name of the Hermes container          (default: hermes)
#   PROXY_NAME         name for the proxy container          (default: prometheus-proxy)
#   PROXY_PORT         host/container port for the proxy     (default: 8780)
#   FIX_WRITE_FILE     1 = also apply the Hermes toolset fix (default: auto)
#                      (adds api_server toolset + terminal.backend=local if
#                       write_file is missing from the API platform)
#
set -euo pipefail

HC="${HERMES_CONTAINER:-hermes}"
PROXY="${PROXY_NAME:-prometheus-proxy}"
PORT="${PROXY_PORT:-8780}"
FIX_WRITE_FILE="${FIX_WRITE_FILE:-auto}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker not found"
docker inspect "$HC" >/dev/null 2>&1 || die "Hermes container '$HC' not found (set HERMES_CONTAINER)"

# ── 0. Locate Hermes network + config ───────────────────────────────────────
say "Inspecting Hermes container '$HC'"
NET=$(docker inspect "$HC" --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}' | head -1)
[ -n "$NET" ] || die "could not determine Hermes docker network"
HHOME=$(docker exec "$HC" printenv HERMES_HOME 2>/dev/null || echo /opt/data)
CFG_HOST=$(docker inspect "$HC" --format "{{range .Mounts}}{{if eq .Destination \"$HHOME\"}}{{.Source}}{{end}}{{end}}")
[ -n "$CFG_HOST" ] && CFG="$CFG_HOST/config.yaml" || die "could not find host path for $HHOME (config volume)"
[ -f "$CFG" ] || die "config not found at $CFG"
echo "network=$NET  config=$CFG"

# Current upstream base_url -> proxy upstream root (strip one trailing /v1).
# Parsed with grep/sed to avoid a host pyyaml dependency.
CUR_BASE=$(grep -m1 -E '^[[:space:]]*base_url:[[:space:]]*[^[:space:]]' "$CFG" \
  | sed -E 's/^[[:space:]]*base_url:[[:space:]]*//' | tr -d '"' | tr -d "'" | tr -d '\r\n')
[ -n "$CUR_BASE" ] || die "could not read model.base_url from $CFG"
PROXY_URL="http://$PROXY:$PORT/v1"
if [ "$CUR_BASE" = "$PROXY_URL" ]; then
  echo "Hermes already points at the proxy; reusing upstream from the existing proxy container."
  UPSTREAM=$(docker inspect "$PROXY" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^UPSTREAM_BASE_URL=//p' | head -1)
  [ -n "$UPSTREAM" ] || die "proxy already targeted but UPSTREAM unknown; run ./uninstall.sh first, then re-install"
else
  UPSTREAM="${CUR_BASE%/v1}"
fi
echo "upstream=$UPSTREAM  proxy_url=$PROXY_URL"

# ── 1. Build + (re)run the proxy on the Hermes network ───────────────────────
say "Building proxy image"
export DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"
mkdir -p "$DOCKER_CONFIG" 2>/dev/null || { export DOCKER_CONFIG=/tmp/.docker; mkdir -p "$DOCKER_CONFIG"; }
( cd "$HERE" && DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}" docker build -t "$PROXY:latest" . >/dev/null )

say "Starting proxy container '$PROXY' on network '$NET' (port $PORT)"
mkdir -p "$HERE/log"
docker rm -f "$PROXY" >/dev/null 2>&1 || true
docker run -d --name "$PROXY" \
  --network "$NET" --restart unless-stopped \
  -p "$PORT:8780" \
  -e UPSTREAM_BASE_URL="$UPSTREAM" -e LOGDIR=/log \
  -v "$HERE/log:/log" \
  "$PROXY:latest" >/dev/null
sleep 3
curl -fsS "http://localhost:$PORT/healthz" >/dev/null || die "proxy healthcheck failed"
echo "proxy healthy"

# ── 2. Repoint Hermes base_url -> proxy ──────────────────────────────────────
if [ "$CUR_BASE" != "$PROXY_URL" ]; then
  say "Repointing Hermes base_url -> $PROXY_URL"
  cp "$CFG" "$CFG.prometheus.bak.$(date +%Y%m%d_%H%M%S)"
  python3 - "$CFG" "$CUR_BASE" "$PROXY_URL" <<'PY'
import sys
cfg, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(cfg).read().replace(old, new)
open(cfg, "w").write(s)
print("replaced all occurrences of base_url")
PY
fi

# ── 3. (Optional) ensure write_file is exposed on the API platform ───────────
ensure_tools() {
  cp "$CFG" "$CFG.prometheus.bak.$(date +%Y%m%d_%H%M%S)"
  python3 - "$CFG" <<'PY'
import sys, re
cfg = sys.argv[1]; s = open(cfg).read()
# (a) ensure the api_server platform inherits the cli toolset list
if "api_server:" not in re.sub(r'platforms.*', '', s) and "  telegram: *id001\n" in s:
    s = s.replace("  telegram: *id001\n", "  telegram: *id001\n  api_server: *id001\n", 1)
# (b) terminal docker backend can fail to init at startup, dropping the file+terminal
#     toolsets; local backend always passes the availability gate.
s = re.sub(r'(\nterminal:\n(?:.*\n)*?  backend: )docker', r'\1local', s, count=1)
open(cfg, "w").write(s)
print("applied write_file/terminal toolset fix")
PY
}

# ── 4. Restart Hermes + verify ───────────────────────────────────────────────
say "Restarting Hermes"
docker restart "$HC" >/dev/null
for i in $(seq 1 40); do
  H=$(docker inspect "$HC" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' 2>/dev/null || echo "?")
  [ "$H" = "healthy" ] || [ "$H" = "running" ] && { echo "Hermes: $H"; break; }
  sleep 3
done

check_write_file() {
  local KEY; KEY=$(docker exec "$HC" printenv HERMES_API_KEY 2>/dev/null | tr -d '\r\n' || true)
  curl -s "http://localhost:8642/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"model":"hermes-agent","messages":[{"role":"user","content":"hi"}],"stream":false}' \
    -o /dev/null 2>/dev/null || true
  docker exec "$PROXY" sh -lc 'cat /log/last_request.json 2>/dev/null' \
    | python3 -c 'import sys,json
try:
  d=json.load(sys.stdin); n=[ (t.get("function") or {}).get("name") for t in (d.get("tools") or []) ]
  print("write_file" in n)
except Exception: print("False")' 2>/dev/null || echo False
}

say "Verifying write_file is exposed to the model"
HAS=$(check_write_file)
echo "write_file present: $HAS"
if [ "$HAS" != "True" ] && { [ "$FIX_WRITE_FILE" = "1" ] || [ "$FIX_WRITE_FILE" = "auto" ]; }; then
  say "write_file missing — applying Hermes toolset fix and restarting"
  ensure_tools
  docker restart "$HC" >/dev/null
  for i in $(seq 1 40); do
    H=$(docker inspect "$HC" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' 2>/dev/null || echo "?")
    [ "$H" = "healthy" ] || [ "$H" = "running" ] && break
    sleep 3
  done
  HAS=$(check_write_file)
  echo "write_file present after fix: $HAS"
fi

say "DONE"
echo "Proxy:    $PROXY on $NET (http://localhost:$PORT)  upstream=$UPSTREAM"
echo "Hermes:   base_url -> $PROXY_URL  (config: $CFG, backups: $CFG.prometheus.bak.*)"
echo "write_file exposed: $HAS"
echo "Rollback: ./uninstall.sh   (restores base_url, removes the proxy)"
