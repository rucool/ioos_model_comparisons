#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_map_comparisons/rtofs_vs_copernicus
log_file_name=model_map_comparisons-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg
python /home/hurricaneadm/scripts/hurricanes/scripts/maps/models/synchronous/rtofs-cmems.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
