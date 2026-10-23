#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/model_comparisons/rtofs-gofs-cmems-amseas
log_file_name=map_comparisons-rtofs-gofs-cmems-amseas-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

# Regions to plot — edit this list to add/remove regions
regions=(mab sab gom caribbean west_florida_shelf tropical_western_atlantic)

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_comps
export PYPROJ_GLOBAL_CONTEXT=ON
export MPLBACKEND=agg
python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/maps/models/synchronous/rtofs-gofs-cmems-amseas.py --no-overwrite --espc --cmems --regions "${regions[@]}" >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile

