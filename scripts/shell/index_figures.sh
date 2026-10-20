#!/bin/bash
# Figure catalog indexer: registers published plots in MongoDB (metadata only)
# so the portal can find them without HTTP probing. Plot scripts register
# their own figures as they save them; this is the safety net for anything
# they missed, and the source of profile positions from locations.json.
#
#   */30 * * * *  /home/hurricaneadm/scripts/figure_catalog/scripts/shell/index_figures.sh
#   30 4 * * 0    /home/hurricaneadm/scripts/figure_catalog/scripts/shell/index_figures.sh --mark-missing
#
# (figure_catalog = a clone used only for this; it runs against its own code,
# so the toolbox checkout the plot crons use doesn't need updating.)
# With no arguments it indexes the last 2 months; pass any index_figures.py
# arguments to override (one-time backfill: --full; preview: --survey).
# Python: conda env model_comps by default (found under ~/miniconda3,
# ~/anaconda3, ~/miniforge3, ... or $CONDA_BASE); or give the exact
# interpreter with PYTHON=/path/to/envs/model_comps/bin/python in the crontab
# line. It needs pymongo, pyyaml and pydantic; the log records which Python ran.
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_comparisons/index_figures
logfile=$logdir/index_figures-$(date --utc +%Y%m%d).log
mkdir -p "$logdir"

# Overlapping runs (a slow --full while the */30 job fires) would just
# duplicate work; skip if one is already running.
exec 9>"$logdir/.lock"
if ! flock -n 9; then
    echo "$(date) previous run still active; skipping" >> $logfile
    exit 0
fi

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

# cron doesn't read ~/.bash_profile; secrets come from this file (see
# gulf_stream_fronts.sh for how to create it).
envfile=$HOME/.config/ioos_model_comparisons.env
[ -f "$envfile" ] && . "$envfile"
if [ -z "$MONGODB_URI" ]; then
    echo "ERROR: MONGODB_URI is not set ($envfile)" >> $logfile
    exit 2
fi

# Which Python: $PYTHON if set (full path, e.g. .../envs/model_comps/bin/python),
# otherwise activate conda env ${CONDA_ENV:-model_comps} from wherever conda
# is installed ($CONDA_BASE, or the usual locations).
if [ -z "$PYTHON" ]; then
    for base in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                "$HOME/mambaforge" /opt/conda /opt/miniconda3 /opt/anaconda3; do
        if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
            . "$base/etc/profile.d/conda.sh"; break
        fi
    done
    conda activate "${CONDA_ENV:-model_comps}" >> $logfile 2>&1 \
        || echo "ERROR: could not activate conda env ${CONDA_ENV:-model_comps}" >> $logfile
    PYTHON=python
fi
echo "python: $(command -v "$PYTHON")" >> $logfile
if ! "$PYTHON" -c "import pymongo, yaml, pydantic" >> $logfile 2>&1; then
    echo "ERROR: $PYTHON is missing pymongo/pyyaml/pydantic. Set PYTHON=<env>/bin/python" >> $logfile
    echo "       or CONDA_BASE/CONDA_ENV in the crontab line." >> $logfile
    exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
# Use this checkout's package, not whatever version is installed in the conda
# env, so a separate catalog-only clone works without touching the checkout
# the plot crons run from.
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
args=("$@")
[ ${#args[@]} -eq 0 ] && args=(--recent-months 2)
"$PYTHON" $repo/scripts/tools/index_figures.py "${args[@]}" >> $logfile 2>&1
status=$?

echo "exit status: $status" >> $logfile
echo ---------------- End ------------------------ >> $logfile
exit $status
