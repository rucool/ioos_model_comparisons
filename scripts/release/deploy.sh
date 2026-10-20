#!/usr/bin/env bash
# Deploy the web app zip to baffin and switch to it atomically.
#
#   scripts/release/deploy.sh [dist/model_comparisons-<version>.zip]
#   (no argument: builds one from the current commit first)
#
# Server layout under $DEPLOY_APP_ROOT (created on first deploy):
#   releases/<version>/   unpacked zips          shared/.env   secrets (yours)
#   venvs/<req hash>/     reused across releases  current ->   live release
#   previous_release      what rollback.sh returns to
# One-time: point the uwsgi vassal at  chdir=$DEPLOY_APP_ROOT/current,
#   virtualenv=$DEPLOY_APP_ROOT/current/venv, module=wsgi:app
# and put the app's environment (MONGODB_URI, FLASK_SECRET_KEY,
# DISCOVERY_BACKEND, ...) in $DEPLOY_APP_ROOT/shared/.env.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/deploy.conf" ] || { echo "missing $HERE/deploy.conf (copy deploy.conf.example)"; exit 2; }
# shellcheck source=/dev/null
source "$HERE/deploy.conf"
case "$DEPLOY_APP_ROOT$HEALTH_URL" in *"<"*|*/path/to/*) echo "fill in deploy.conf first"; exit 2;; esac

ZIP="${1:-}"
if [ -z "$ZIP" ]; then
  ZIP="$(python "$HERE/build_webapp.py" | sed -n 's/^built \(.*\.zip\) .*/\1/p')"
fi
[ -f "$ZIP" ] || { echo "no such zip: $ZIP"; exit 2; }
NAME="$(basename "$ZIP" .zip)"            # model_comparisons-<version>
VERSION="${NAME#model_comparisons-}"
case "$VERSION" in *dirty*) echo "refusing to deploy a dirty build ($VERSION)"; exit 2;; esac

echo "==> uploading $NAME to $DEPLOY_HOST"
scp -q "$ZIP" "$DEPLOY_HOST:/tmp/$NAME.zip"

echo "==> installing on $DEPLOY_HOST"
ssh "$DEPLOY_HOST" bash -s -- "$DEPLOY_APP_ROOT" "$NAME" "$DEPLOY_PYTHON" "$DEPLOY_VASSAL_INI" "${KEEP_RELEASES:-5}" <<'REMOTE'
set -euo pipefail
ROOT="$1"; NAME="$2"; PY="$3"; INI="$4"; KEEP="$5"
mkdir -p "$ROOT/releases" "$ROOT/venvs" "$ROOT/shared"
cd "$ROOT/releases"
rm -rf "$NAME"; unzip -q "/tmp/$NAME.zip"; rm -f "/tmp/$NAME.zip"
REQ="$NAME/requirements.txt"; [ -f "$NAME/requirements.lock" ] && REQ="$NAME/requirements.lock"
H="$(sha256sum "$REQ" | cut -c1-12)"
if [ ! -x "$ROOT/venvs/$H/bin/python" ]; then
  echo "    new dependency set -> building venv $H"
  "$PY" -m venv "$ROOT/venvs/$H"
  "$ROOT/venvs/$H/bin/pip" install -q --upgrade pip
  "$ROOT/venvs/$H/bin/pip" install -q -r "$REQ"
fi
ln -sfn "$ROOT/venvs/$H" "$NAME/venv"
[ -f "$ROOT/shared/.env" ] && ln -sfn "$ROOT/shared/.env" "$NAME/.env"
# Fail before switching if the app can't even import with this venv.
( cd "$NAME" && venv/bin/python -c "import wsgi" ) || { echo "import check failed"; exit 1; }
[ -L "$ROOT/current" ] && readlink "$ROOT/current" > "$ROOT/previous_release"
ln -sfn "$ROOT/releases/$NAME" "$ROOT/current.tmp" && mv -T "$ROOT/current.tmp" "$ROOT/current"
touch "$INI"
ls -1dt "$ROOT"/releases/model_comparisons-* | tail -n +$((KEEP + 1)) | xargs -r rm -rf
echo "    live: $(readlink "$ROOT/current")"
REMOTE

echo "==> waiting for $HEALTH_URL to report $VERSION"
for _ in $(seq 1 20); do
  got="$(curl -fsS "$HEALTH_URL" 2>/dev/null | python -c 'import json,sys; print(json.load(sys.stdin)["version"])' 2>/dev/null || true)"
  if [ "$got" = "$VERSION" ]; then echo "==> deployed $VERSION"; exit 0; fi
  sleep 3
done
echo "!! $HEALTH_URL never reported $VERSION (last: ${got:-no response}) — rolling back"
"$HERE/rollback.sh"
exit 1
