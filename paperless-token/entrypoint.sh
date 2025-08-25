#!/bin/sh
set -eux

# Defaults (can be overridden via env)
DB_PATH="${DB_PATH:-/usr/src/paperless/data/db.sqlite3}"
TOKEN_PATH="${TOKEN_PATH:-/bootstrap/paperless_token.txt}"
USER="${PAPERLESS_ADMIN_USER:-admin}"
TIMEOUT="${BOOTSTRAP_TIMEOUT:-360}"

# Ensure we're at the Django project root so manage.py is available
cd /usr/src/paperless/src

echo "[token-init] Waiting for Paperless DB at ${DB_PATH} ..."
start="$(date +%s)"
while [ ! -s "${DB_PATH}" ]; do
  now="$(date +%s)"
  [ $((now-start)) -ge "${TIMEOUT}" ] && break || true
  sleep 3
done

if [ ! -s "${DB_PATH}" ]; then
  echo "[token-init] DB not found within timeout (${TIMEOUT}s). Writing PENDING."
  printf 'PENDING' > "${TOKEN_PATH}"
  exit 0
fi

echo "[token-init] Creating/retrieving API token for user: ${USER} ..."
start="$(date +%s)"
while :; do
  # Try DRF management command first
  out="$(python3 manage.py drf_create_token "${USER}" || true)"
  echo "[token-init] manage.py output: ${out}" || true
  # Extract first long hex-like token from output
  token="$(printf '%s\n' "${out}" | tr -d '\r' | grep -Eio '[0-9a-f]{32,}' | head -n1 || true)"
  if [ -n "${token}" ] && [ "${token}" != "${USER}" ]; then
    printf '%s' "${token}" > "${TOKEN_PATH}"
    echo "[token-init] Token saved: ${TOKEN_PATH}"
    exit 0
  fi
  # Fallback to direct Django script
  token="$(python3 /bootstrap/get_token.py | tr -d '\r\n' || true)"
  if [ -n "${token}" ]; then
    printf '%s' "${token}" > "${TOKEN_PATH}"
    echo "[token-init] Token saved via fallback: ${TOKEN_PATH}"
    exit 0
  fi
  now="$(date +%s)"
  [ $((now-start)) -ge "${TIMEOUT}" ] && break
  sleep 3
done

printf 'PENDING' > "${TOKEN_PATH}"
echo '[token-init] Token pending (timeout reached)'
exit 0
