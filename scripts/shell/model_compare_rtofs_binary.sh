#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_comparisons/rtofs_binary
log_file_name=rtofs_binary-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}
mkdir -p $logdir

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_comps
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg
python /home/hurricaneadm/scripts/ioos_model_comparisons/scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py \
    --data-dir /home/hurricaneadm/data/rtofs_archv \
    --regions guam hawaii >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
