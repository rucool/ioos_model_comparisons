#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/argo_profile_model_comparisons
log_file_name=argo_profile_model_comparisons-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
python /home/hurricaneadm/scripts/hurricanes/scripts/profiles/argo/synchronous/argo_profile_model_comparisons.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
