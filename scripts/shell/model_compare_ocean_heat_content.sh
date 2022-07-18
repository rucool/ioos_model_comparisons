#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_comparisons/ocean_heat_content
log_file_name=ocean_heat_content-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg
python /home/hurricaneadm/scripts/hurricanes/scripts/maps/models/synchronous/ocean_heat_content.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
