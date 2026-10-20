#!/usr/bin/env bash
# Return baffin to the release that was live before the last install (runs
# install_release.sh --rollback there over ssh). The server admin can run
# `install_release.sh --app-root <root> --rollback` on baffin directly instead.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/deploy.conf"
scp -q "$HERE/install_release.sh" "$DEPLOY_HOST:/tmp/"
ssh "$DEPLOY_HOST" bash /tmp/install_release.sh \
    --app-root "$DEPLOY_APP_ROOT" --vassal-ini "$DEPLOY_VASSAL_INI" \
    --python "${DEPLOY_PYTHON:-python3}" --health-url "$HEALTH_URL" --rollback
