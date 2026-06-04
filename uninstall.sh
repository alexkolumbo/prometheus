#!/usr/bin/env bash
#
# Rollback for install.sh: repoint Hermes back to the upstream provider directly
# and remove the proxy container. Does NOT revert the optional toolset fix
# (api_server toolset / terminal.backend=local) since that is harmless/beneficial;
# restore a config.yaml.prometheus.bak.* manually if you want it gone.
#
set -euo pipefail
HC="${HERMES_CONTAINER:-hermes}"
PROXY="${PROXY_NAME:-prometheus-proxy}"
PORT="${PROXY_PORT:-8780}"

HHOME=$(docker exec "$HC" printenv HERMES_HOME 2>/dev/null || echo /opt/data)
CFG_HOST=$(docker inspect "$HC" --format "{{range .Mounts}}{{if eq .Destination \"$HHOME\"}}{{.Source}}{{end}}{{end}}")
CFG="$CFG_HOST/config.yaml"
PROXY_URL="http://$PROXY:$PORT/v1"
UPSTREAM=$(docker inspect "$PROXY" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^UPSTREAM_BASE_URL=//p' | head -1)

if [ -n "$UPSTREAM" ] && grep -q "$PROXY_URL" "$CFG" 2>/dev/null; then
  cp "$CFG" "$CFG.prometheus.bak.$(date +%Y%m%d_%H%M%S)"
  python3 - "$CFG" "$PROXY_URL" "$UPSTREAM/v1" <<'PY'
import sys
cfg, old, new = sys.argv[1:4]
open(cfg,"w").write(open(cfg).read().replace(old, new))
print(f"restored base_url -> {new}")
PY
  docker restart "$HC" >/dev/null && echo "Hermes restarted"
else
  echo "Hermes config does not point at the proxy (nothing to repoint)."
fi

docker rm -f "$PROXY" >/dev/null 2>&1 && echo "removed proxy container '$PROXY'" || true
echo "DONE."
