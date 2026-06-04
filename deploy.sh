#!/bin/bash
set -e
# ZimaOS root is read-only; point docker CLI config at a writable dir.
export DOCKER_CONFIG=/DATA/.docker
mkdir -p "$DOCKER_CONFIG"
cd /DATA/Prometheus/proxy
mkdir -p log
echo "=== build image ==="
DOCKER_BUILDKIT=0 docker build -t prometheus-proxy:0.1 . 2>&1 | tail -8
echo
echo "=== (re)create container on hermes_v3_default ==="
docker rm -f prometheus-proxy 2>/dev/null || true
docker run -d --name prometheus-proxy \
  --network hermes_v3_default \
  --restart unless-stopped \
  -p 8780:8780 \
  -e UPSTREAM_BASE_URL="https://proxy.gonka.gg" \
  -e LOGDIR="/log" \
  -v /DATA/Prometheus/proxy/log:/log \
  prometheus-proxy:0.1
echo
sleep 3
echo "=== container status ==="
docker ps --filter name=prometheus-proxy --format '{{.Names}} {{.Status}} {{.Ports}}'
echo
echo "=== healthz (from host) ==="
curl -s http://localhost:8780/healthz; echo
echo
echo "=== proxy reachable from hermes container? (proxied /v1/models) ==="
docker exec hermes sh -lc 'curl -s -o /dev/null -w "via-proxy /v1/models http=%{http_code}\n" http://prometheus-proxy:8780/v1/models -H "Authorization: Bearer $GONKA_API_KEY"'
