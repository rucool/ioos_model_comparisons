#!/usr/bin/env bash
# Install a model_comparisons release ON THE SERVER (baffin). No ssh, no git:
# everything happens locally under the app root.
#
#   install_release.sh model_comparisons-<version>.zip   install + switch + verify
#   install_release.sh --rollback                        back to the previous release
#   install_release.sh --status                          show live/previous releases
#   install_release.sh --adopt /path/to/old/app          one time: take over an existing
#                                                        hand-installed deployment
#
# The script ships inside every release zip, so a fresh copy is always at hand:
#   unzip -o -j model_comparisons-<version>.zip '*/install_release.sh' -d /tmp
#   bash /tmp/install_release.sh model_comparisons-<version>.zip
#
# Settings: command-line flags win, then environment variables, then
# <app root>/install.conf (KEY=value lines, written once by the server admin):
#   --app-root   APP_ROOT     where releases/, venvs/, shared/, current live
#   --vassal-ini VASSAL_INI   uwsgi vassal file to touch so uwsgi reloads
#   --python     PYTHON       interpreter for release venvs        (default python3)
#   --health-url HEALTH_URL   .../healthz to verify the new version (optional)
#   --keep       KEEP         releases to keep                      (default 5)
#
# Layout under APP_ROOT (created on first install):
#   releases/model_comparisons-<version>/   unpacked releases
#   venvs/<hash>/                           package environments, reused while
#                                           requirements are unchanged
#   shared/.env                             MONGODB_URI, FLASK_SECRET_KEY,
#                                           DISCOVERY_BACKEND ... (admin-owned)
#   current -> releases/<live release>      what uwsgi serves
#   previous_release                        what --rollback returns to
#
# Switching over from a hand-installed deployment (once):
#   1. install_release.sh --app-root <APP_ROOT> --adopt <old app dir>
#      registers the running app as release "model_comparisons-adopted" and
#      copies its .env to shared/.env. Changes nothing that is serving.
#   2. In the vassal ini set   chdir      = <APP_ROOT>/current
#                              virtualenv = <APP_ROOT>/current/venv
#                              module     = wsgi:app
#      and touch it: the same old code is now served through `current`.
#   3. install_release.sh <zip>; --rollback returns to the adopted app.
set -euo pipefail

usage() { sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

MODE=install; ZIP=""; ADOPT=""
F_APP_ROOT=""; F_VASSAL_INI=""; F_PYTHON=""; F_HEALTH_URL=""; F_KEEP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --app-root) F_APP_ROOT="$2"; shift 2;;
    --vassal-ini) F_VASSAL_INI="$2"; shift 2;;
    --python) F_PYTHON="$2"; shift 2;;
    --health-url) F_HEALTH_URL="$2"; shift 2;;
    --keep) F_KEEP="$2"; shift 2;;
    --rollback) MODE=rollback; shift;;
    --status) MODE=status; shift;;
    --adopt) MODE=adopt; ADOPT="$2"; shift 2;;
    -h|--help) usage 0;;
    -*) echo "unknown option $1"; usage 2;;
    *) ZIP="$1"; shift;;
  esac
done

APP_ROOT="${F_APP_ROOT:-${APP_ROOT:-}}"
[ -n "$APP_ROOT" ] || { echo "set --app-root (or APP_ROOT)"; exit 2; }
CONF="$APP_ROOT/install.conf"
conf() {  # conf KEY -> value from install.conf, or empty
  [ -f "$CONF" ] && sed -n "s/^$1=//p" "$CONF" | tail -1 | sed 's/^"\(.*\)"$/\1/' || true
}
VASSAL_INI="${F_VASSAL_INI:-${VASSAL_INI:-$(conf VASSAL_INI)}}"
PYTHON="${F_PYTHON:-${PYTHON:-$(conf PYTHON)}}"; PYTHON="${PYTHON:-python3}"
HEALTH_URL="${F_HEALTH_URL:-${HEALTH_URL:-$(conf HEALTH_URL)}}"
KEEP="${F_KEEP:-${KEEP:-$(conf KEEP)}}"; KEEP="${KEEP:-5}"

say() { echo "==> $*"; }

switch_to() {  # atomically point current at $1, then make uwsgi reload
  ln -sfn "$1" "$APP_ROOT/current.tmp"
  "$PYTHON" -c 'import os, sys; os.replace(sys.argv[1], sys.argv[2])' \
      "$APP_ROOT/current.tmp" "$APP_ROOT/current"
  if [ -n "$VASSAL_INI" ]; then
    touch "$VASSAL_INI" && say "reloaded uwsgi ($VASSAL_INI)"
  else
    echo "!! no VASSAL_INI set: reload uwsgi yourself"
  fi
}

wait_for_version() {  # wait_for_version VERSION -> 0 when /healthz reports it
  [ -n "$HEALTH_URL" ] || { echo "!! no HEALTH_URL set: skipping the live check"; return 0; }
  say "waiting for $HEALTH_URL to report $1"
  local got=""
  for _ in $(seq 1 20); do
    got="$(curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null \
           | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["version"])' 2>/dev/null || true)"
    [ "$got" = "$1" ] && { say "live: $1"; return 0; }
    sleep 3
  done
  echo "!! $HEALTH_URL never reported $1 (last: ${got:-no response})"
  return 1
}

rollback() {
  local prev cur
  prev="$(cat "$APP_ROOT/previous_release" 2>/dev/null || true)"
  [ -n "$prev" ] && [ -d "$prev" ] || { echo "!! no previous release recorded"; return 1; }
  cur="$(readlink "$APP_ROOT/current" || true)"
  switch_to "$prev"
  [ -n "$cur" ] && echo "$cur" > "$APP_ROOT/previous_release"
  say "rolled back: live is $(basename "$prev") (was $(basename "${cur:-none}"))"
  if [ -f "$prev/VERSION" ]; then
    wait_for_version "$(cat "$prev/VERSION")" || true
  else
    echo "    (pre-release-system app: no /healthz to check; open the site to confirm)"
  fi
}

case "$MODE" in
  status)
    echo "live:     $(readlink "$APP_ROOT/current" 2>/dev/null || echo none)"
    echo "previous: $(cat "$APP_ROOT/previous_release" 2>/dev/null || echo none)"
    ls -1dt "$APP_ROOT"/releases/model_comparisons-* 2>/dev/null || true
    exit 0;;
  rollback)
    rollback; exit $?;;
  adopt)
    [ -f "$ADOPT/app.py" ] || { echo "!! $ADOPT has no app.py: not the old app folder"; exit 2; }
    [ -e "$APP_ROOT/current" ] && { echo "!! $APP_ROOT/current already exists: already set up"; exit 2; }
    mkdir -p "$APP_ROOT/releases" "$APP_ROOT/venvs" "$APP_ROOT/shared"
    OLD="$(cd "$ADOPT" && pwd)"
    ln -sfn "$OLD" "$APP_ROOT/releases/model_comparisons-adopted"
    if [ -f "$OLD/.env" ] && [ ! -f "$APP_ROOT/shared/.env" ]; then
      cp -p "$OLD/.env" "$APP_ROOT/shared/.env" && chmod 600 "$APP_ROOT/shared/.env"
      say "copied $OLD/.env to shared/.env"
    fi
    [ -d "$OLD/venv" ] || echo "!! $OLD has no venv/ folder: set the vassal virtualenv by hand until the first install"
    ln -sfn "$APP_ROOT/releases/model_comparisons-adopted" "$APP_ROOT/current"
    say "adopted $OLD as the live release (nothing reloaded)"
    echo "    next: point the vassal at $APP_ROOT/current (see the header of this script), touch it,"
    echo "    check the site, then install a release zip."
    exit 0;;
esac

# ---------------------------------------------------------------- install
[ -n "$ZIP" ] && [ -f "$ZIP" ] || { echo "usage: $0 model_comparisons-<version>.zip"; exit 2; }
if [ -n "$VASSAL_INI" ] && [ ! -w "$VASSAL_INI" ]; then
  echo "!! cannot write $VASSAL_INI (needed to reload uwsgi): run as its owner, or with sudo"; exit 2
fi
NAME="$(unzip -Z1 "$ZIP" | head -1 | cut -d/ -f1)"
case "$NAME" in model_comparisons-*) ;; *) echo "!! $ZIP is not a release zip (top folder: $NAME)"; exit 2;; esac
VERSION="$(unzip -p "$ZIP" "$NAME/VERSION" | tr -d '[:space:]')"
case "$VERSION" in *dirty*) echo "!! refusing a dirty build ($VERSION)"; exit 2;; esac

mkdir -p "$APP_ROOT/releases" "$APP_ROOT/venvs" "$APP_ROOT/shared"
REL="$APP_ROOT/releases/$NAME"
if [ "$(readlink "$APP_ROOT/current" 2>/dev/null || true)" = "$REL" ]; then
  echo "!! $VERSION is already live; nothing to do"; exit 0
fi
say "unpacking $VERSION into $REL"
rm -rf "$REL"
unzip -q "$ZIP" -d "$APP_ROOT/releases"

REQ="$REL/requirements.txt"; [ -f "$REL/requirements.lock" ] && REQ="$REL/requirements.lock"
HASH="$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:12])' "$REQ")"
VENV="$APP_ROOT/venvs/$HASH"
if [ -x "$VENV/bin/python" ]; then
  say "reusing package environment $HASH (requirements unchanged)"
else
  say "requirements changed: building package environment $HASH (a few minutes)"
  rm -rf "$VENV.tmp" "$VENV"   # leftovers of an interrupted build
  "$PYTHON" -m venv "$VENV.tmp"
  "$VENV.tmp/bin/pip" install -q --upgrade pip
  "$VENV.tmp/bin/pip" install -q -r "$REQ"
  mv "$VENV.tmp" "$VENV"
fi
ln -sfn "$VENV" "$REL/venv"
if [ -f "$APP_ROOT/shared/.env" ]; then
  ln -sfn "$APP_ROOT/shared/.env" "$REL/.env"
else
  echo "!! $APP_ROOT/shared/.env not found: the app will run without MongoDB/sign-in settings"
fi

say "checking the release imports"
( cd "$REL" && venv/bin/python -c "import wsgi" ) || { echo "!! import failed: live site unchanged"; exit 1; }

CUR="$(readlink "$APP_ROOT/current" 2>/dev/null || true)"
if [ "$CUR" != "$REL" ]; then
  [ -n "$CUR" ] && echo "$CUR" > "$APP_ROOT/previous_release"
fi
switch_to "$REL"

# keep the newest $KEEP releases, never the live or previous one
PREV="$(cat "$APP_ROOT/previous_release" 2>/dev/null || true)"
ls -1dt "$APP_ROOT"/releases/model_comparisons-* | tail -n +$((KEEP + 1)) | while read -r old; do
  [ "$old" = "$REL" ] || [ "$old" = "$PREV" ] || rm -rf "$old"
done

if ! wait_for_version "$VERSION"; then
  echo "!! rolling back"
  rollback || true
  exit 1
fi
say "done: $VERSION is live"
