#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/argo_profile_model_comparisons_rtofs_binary
log_file_name=argo_profile_model_comparisons_rtofs_binary-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}
mkdir -p $logdir

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_comps
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg
python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/profiles/argo/synchronous/argo_profile_model_comparisons_rtofs_binary.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
