#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/Daily_glider_profiles_vs_models
log_file_name=Daily_glider_profiles_vs_models-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
python /home/hurricaneadm/scripts/Daily_glider_models_comparisons/realtime/Daily_glider_profiles_vs_models.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
