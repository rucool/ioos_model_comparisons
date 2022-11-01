#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/grab_rtofs
log_file_name=grab_rtofs-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricane
# python /home/hurricaneadm/scripts/grab_rtofs.py >> $logfile
python /home/hurricaneadm/scripts/ioos_model_comparisons/scripts/harvest/grab_rtofs.py >> $logfile

conda deactivate

echo ---------------- End ------------------------ >> $logfile
