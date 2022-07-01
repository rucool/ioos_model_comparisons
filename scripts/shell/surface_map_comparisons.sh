#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_map_comparisons
log_file_name=model_map_comparisons-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
#python /home/hurricaneadm/scripts/hurricanes/scripts/surface_maps/surface_map_comparisons.py >> $logfile
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg
python /home/hurricaneadm/scripts/hurricanes/scripts/maps/models/model_map_comparisons.py
conda deactivate

echo ---------------- End ------------------------ >> $logfile
