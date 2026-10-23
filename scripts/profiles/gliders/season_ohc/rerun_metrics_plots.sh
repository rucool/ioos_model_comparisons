#!/usr/bin/env bash
# Re-run steps 2 (metrics) and 3 (plots) for every region/model with paired data. No downloads.
# usage: ./rerun_metrics_plots.sh [end date for plot titles, default today]
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n model_comps python"
ROOT=$($PY -c 'from season_config import ROOT; print(ROOT)')
T0=$($PY -c 'from season_config import T0; print(T0.date())')
T1=${1:-$(date +%Y-%m-%d)}
LOGS=$ROOT/logs; mkdir -p "$LOGS"
for rm in "Caribbean:cmems:caribbean" "Caribbean:rtofs:caribbean" "Hawaii:cmems:hawaii" "Hawaii:rtofs:hawaii" \
          "Guam:cmems:guam" "Guam:rtofs:guam" "Mexico Pacific:cmems:mexico_pacific" "Mexico Pacific:rtofs:mexico_pacific"; do
    IFS=: read -r title model folder <<< "$rm"
    ls "$ROOT/$folder"/*/*_"$model".csv >/dev/null 2>&1 || continue
    $PY step2_calculate_metrics.py "$folder" "$model" > "$LOGS/p2_${folder}_${model}.log" 2>&1
    $PY step3_plot_hexbin.py "$title" "$model" "$T0" "$T1" "$folder" > "$LOGS/p3_${folder}_${model}.log" 2>&1
    echo "$(date '+%T') $title/$model: $(grep -a -e 'pass qc' -e 'Traceback' "$LOGS/p3_${folder}_${model}.log" | head -1)"
done
