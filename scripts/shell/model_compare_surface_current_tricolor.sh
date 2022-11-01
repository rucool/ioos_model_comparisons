#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_comparisons/surface_current_tricolor
log_file_name=surface_current_tricolor-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
# export PYPROJ_GLOBAL_CONTEXT=ON
# export MPLBACKEND=agg
python /home/hurricaneadm/scripts/ioos_model_comparisons/scripts/maps/models/synchronous/surface_current_contour_gom.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
