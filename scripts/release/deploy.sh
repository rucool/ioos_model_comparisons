#!/usr/bin/env bash
# Deploy from your machine: build the release zip, copy it and the installer
# to baffin, and run the installer there over ssh. Needs ssh access to baffin
# and write access to the app root and vassal ini. Without that, hand the zip
# to the server admin, who runs install_release.sh on baffin; it's the same
# installer and the same steps.
#
#   scripts/release/deploy.sh [dist/model_comparisons-<version>.zip]
#   (no argument: builds one from the current commit first)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/deploy.conf" ] || { echo "missing $HERE/deploy.conf (copy deploy.conf.example)"; exit 2; }
# shellcheck source=/dev/null
source "$HERE/deploy.conf"
case "$DEPLOY_APP_ROOT$HEALTH_URL" in *"<"*|*/path/to/*) echo "fill in deploy.conf first"; exit 2;; esac

ZIP="${1:-}"
if [ -z "$ZIP" ]; then
  ZIP="$(python "$HERE/build_webapp.py" | sed -n 's/^built \(.*\.zip\) .*/\1/p')"
fi
[ -f "$ZIP" ] || { echo "no such zip: $ZIP"; exit 2; }
NAME="$(basename "$ZIP")"

echo "==> copying $NAME to $DEPLOY_HOST"
scp -q "$ZIP" "$HERE/install_release.sh" "$DEPLOY_HOST:/tmp/"
status=0
ssh "$DEPLOY_HOST" bash "/tmp/install_release.sh" \
    --app-root "$DEPLOY_APP_ROOT" --vassal-ini "$DEPLOY_VASSAL_INI" \
    --python "${DEPLOY_PYTHON:-python3}" --health-url "$HEALTH_URL" \
    --keep "${KEEP_RELEASES:-5}" "/tmp/$NAME" || status=$?
ssh "$DEPLOY_HOST" rm -f "/tmp/$NAME"
exit $status
