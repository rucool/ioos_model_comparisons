#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/Daily_map_North_Atlantic_gliders_in_DAC
log_file_name=Daily_map_North_Atlantic_gliders_in_DAC-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricane
python /home/hurricaneadm/scripts/Daily_glider_models_comparisons/realtime/Daily_map_North_Atlantic_gliders_in_DAC.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
