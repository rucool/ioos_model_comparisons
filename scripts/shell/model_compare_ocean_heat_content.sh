#!/bin/bash
PATH=/bin:/usr/bin
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"

logdir=$HOME/logs/model_comparisons/ocean_heat_content
log_file_name=ocean_heat_content-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}
lockfile="/tmp/ocean_heat_content.lock"

# Exit if lock file exists and process is running
if [ -f "$lockfile" ]; then
    pid=$(cat "$lockfile")
    if ps -p $pid > /dev/null 2>&1; then
        echo "[$(date)] Script is already running (PID $pid), exiting." >> $logfile
        exit 1
    else
        echo "[$(date)] Stale lock file found, removing." >> $logfile
        rm -f "$lockfile"
    fi
fi

# Write current PID to lock file
echo $$ > "$lockfile"

# Run the script
{
    echo ---------------- Start ----------------------
    date

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate model_comps
    export PYPROJ_GLOBAL_CONTEXT=ON
    export MPLBACKEND=agg
    python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/maps/models/synchronous/ocean_heat_content.py
    conda deactivate

    echo ---------------- End ------------------------
    date
} >> $logfile 2>&1

# Clean up
rm -f "$lockfile"

