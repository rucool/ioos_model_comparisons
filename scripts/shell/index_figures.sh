#!/bin/bash
# Figure catalog indexer: registers published plots in MongoDB (metadata only)
# so the portal can find them without HTTP probing. Plot scripts register
# their own figures as they save them; this is the safety net for anything
# they missed, and the source of profile positions from locations.json.
#
#   */30 * * * *  /home/hurricaneadm/scripts/ioos_model_comparisons/scripts/shell/index_figures.sh
#   30 4 * * 0    /home/hurricaneadm/scripts/ioos_model_comparisons/scripts/shell/index_figures.sh --mark-missing
#
# With no arguments it indexes the last 2 months; pass any index_figures.py
# arguments to override (one-time backfill: --full; preview: --survey).
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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
# Use this checkout's package, not whatever version is installed in the conda
# env, so a separate catalog-only clone works without touching the checkout
# the plot crons run from.
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
args=("$@")
[ ${#args[@]} -eq 0 ] && args=(--recent-months 2)
python $repo/scripts/tools/index_figures.py "${args[@]}" >> $logfile 2>&1
status=$?

conda deactivate
echo "exit status: $status" >> $logfile
echo ---------------- End ------------------------ >> $logfile
exit $status
