#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_maps
log_file_name=model_maps_rtofs-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hurricanes
#python /home/hurricaneadm/scripts/hurricanes/scripts/surface_maps/surface_maps_rtofs.py >> $logfile
python /home/hurricaneadm/scripts/hurricanes/scripts/maps/models/model_maps_rtofs.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
