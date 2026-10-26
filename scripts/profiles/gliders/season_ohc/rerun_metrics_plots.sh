#!/usr/bin/env bash
# Re-run steps 2 (metrics), 3 (model plots) and 4 (NESDIS OHC plots) for every region/model with paired
# data. No downloads. SKIP_STEP2=1 replots without recomputing metrics.
# usage: [SEASON_PLATFORM=argo] [SKIP_STEP2=1] ./rerun_metrics_plots.sh [end date for plot titles, default today]
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n model_comps python"
ROOT=$($PY -c 'from season_config import ROOT; print(ROOT)')
T0=$($PY -c 'from season_config import T0; print(T0.date())')
T1=${1:-$(date +%Y-%m-%d)}
LOGS=$ROOT/logs; mkdir -p "$LOGS"
for rm in "Caribbean:rtofs:caribbean" "Caribbean:cmems:caribbean" "Caribbean:espc:caribbean" \
          "Gulf of Mexico:rtofs:gom" "Gulf of Mexico:cmems:gom" "Gulf of Mexico:espc:gom" "Hawaii:rtofs:hawaii" "Hawaii:cmems:hawaii" "Hawaii:espc:hawaii" \
          "Guam:rtofs:guam" "Guam:cmems:guam" "Guam:espc:guam" "Mexico Pacific:rtofs:mexico_pacific" "Mexico Pacific:cmems:mexico_pacific" "Mexico Pacific:espc:mexico_pacific"; do
    IFS=: read -r title model folder <<< "$rm"
    [ -n "$(find "$ROOT/$folder" -maxdepth 2 -name "*_${model}.csv" -print -quit 2>/dev/null)" ] || continue   # find, not ls: too many files for one glob
    [ -n "$SKIP_STEP2" ] || $PY step2_calculate_metrics.py "$folder" "$model" > "$LOGS/p2_${folder}_${model}.log" 2>&1
    $PY step3_plot_hexbin.py "$title" "$model" "$T0" "$T1" "$folder" > "$LOGS/p3_${folder}_${model}.log" 2>&1
    echo "$(date '+%T') $title/$model: $(grep -a -e 'pass qc' -e 'Traceback' "$LOGS/p3_${folder}_${model}.log" | head -1)"
done
for rf in "Caribbean:caribbean" "Gulf of Mexico:gom" "Hawaii:hawaii" "Guam:guam" "Mexico Pacific:mexico_pacific"; do
    IFS=: read -r title folder <<< "$rf"
    [ -n "$(find "$ROOT/$folder" -path "*/calculated/*.csv" -print -quit 2>/dev/null)" ] || continue
    $PY step4_nesdis_ohc.py "$title" "$T0" "$T1" "$folder" > "$LOGS/p4_${folder}_nesdis.log" 2>&1
    echo "$(date '+%T') $title/NESDIS: $(grep -a -e 'pairs with a NESDIS value' -e 'Traceback' "$LOGS/p4_${folder}_nesdis.log" | head -1)"
done
