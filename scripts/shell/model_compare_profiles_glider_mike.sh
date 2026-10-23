#!/bin/bash
PATH=/bin:/usr/bin;
PATH="/home/hurricaneadm/miniconda3/bin:$PATH"
logdir=$HOME/logs/Daily_glider_profiles_vs_models
log_file_name=Daily_glider_profiles_vs_models-$(date --utc +%Y%m%d).log
logfile=$logdir/${log_file_name}

echo ---------------- Start ---------------------- >> $logfile
date >> $logfile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate model_comps
python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/profiles/gliders/synchronous/daily_glider_profiles_comparisons_400m.py >> $logfile
python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/profiles/gliders/synchronous/daily_glider_profiles_comparisons_1000m.py >> $logfile
python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/profiles/gliders/synchronous/daily_glider_profiles_comparisons_400m_pacific.py >> $logfile
# daily_glider_profiles_comparisons_1000m_pacific.py was never added to git (only an
# old 2025 copy exists in ~/scripts/ioos_model_comparisons), so this line failed
# every hour. Re-enable once the script is in the repo.
# python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/profiles/gliders/synchronous/daily_glider_profiles_comparisons_1000m_pacific.py >> $logfile
python /home/hurricaneadm/scripts/temp/ioos_model_comparisons/scripts/profiles/gliders/synchronous/daily_glider_profiles_comparisons_400m_guam.py >> $logfile
conda deactivate

echo ---------------- End ------------------------ >> $logfile
