#!/bin/bash
# Nightly Gulf Stream front digitizer.
#
#   0 13 * * *  /home/hurricaneadm/scripts/ioos_model_comparisons/scripts/shell/gulf_stream_fronts.sh
#
# Writes GeoJSON/PNG to outputs/gulf_stream_fronts AND to MongoDB, which is
# what the website and the next day's QC read.
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_comparisons/gulf_stream_fronts
log_file_name=gulf_stream_fronts-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}
mkdir -p "$logdir"

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

# cron does NOT read ~/.bash_profile — it runs a non-login, non-interactive
# shell. Anything exported there is invisible here, so secrets are loaded from
# a dedicated file instead. Create it once, readable only by this user:
#
#   umask 077
#   printf 'export MONGODB_URI="%s"\n' 'mongodb://USER:PASS@host:27017' \
#       > $HOME/.config/ioos_model_comparisons.env
#
envfile=$HOME/.config/ioos_model_comparisons.env
if [ -f "$envfile" ]; then
    . "$envfile"
else
    echo "WARNING: $envfile not found" >> $logfile
fi

# Fail loudly rather than degrade. Without MONGODB_URI the run still writes
# files and exits 0, so the only symptom is a website that quietly stops
# updating — which is exactly the failure that goes unnoticed for a week.
if [ -z "$MONGODB_URI" ]; then
    echo "ERROR: MONGODB_URI is not set — the digitizer would write files only" >> $logfile
    echo "       and the website would silently serve a stale wall." >> $logfile
    echo "       Populate $envfile (see the comment in this script)." >> $logfile
    echo "---------------- End [aborted] --------------" >> $logfile
    exit 2
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg

# Derive the repo from this script's own location rather than hardcoding it,
# so a test checkout (e.g. ~/scripts/temp/...) runs against itself instead of
# silently driving the production tree.
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
echo "repo: $repo" >> $logfile
python $repo/scripts/fronts/run_digitizer_goes19.py >> $logfile 2>&1
status=$?

# Catch-up: idempotent, so if a previous night could not reach MongoDB this
# loads whatever is on disk but missing from the database.
python $repo/scripts/tools/backfill_fronts_to_mongo.py >> $logfile 2>&1

conda deactivate

echo "digitizer exit status: $status" >> $logfile
echo ---------------- End ------------------------ >> $logfile
exit $status
