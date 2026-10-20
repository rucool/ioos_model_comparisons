#!/usr/bin/env bash
# Switch baffin back to the release that was live before the last deploy.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/deploy.conf"
ssh "$DEPLOY_HOST" bash -s -- "$DEPLOY_APP_ROOT" "$DEPLOY_VASSAL_INI" <<'REMOTE'
set -euo pipefail
ROOT="$1"; INI="$2"
PREV="$(cat "$ROOT/previous_release" 2>/dev/null || true)"
[ -n "$PREV" ] && [ -d "$PREV" ] || { echo "no previous release recorded"; exit 1; }
CUR="$(readlink "$ROOT/current")"
ln -sfn "$PREV" "$ROOT/current.tmp" && mv -T "$ROOT/current.tmp" "$ROOT/current"
echo "$CUR" > "$ROOT/previous_release"
touch "$INI"
echo "    live: $PREV (was $CUR)"
REMOTE
curl -fsS "$HEALTH_URL" && echo
