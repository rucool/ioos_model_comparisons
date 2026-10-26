#!/usr/bin/env bash
# Hurricane-season glider (or Argo: SEASON_PLATFORM=argo) vs model comparison, unattended: download -> pair -> metrics -> plots.
# Caribbean, Hawaii, Mexico Pacific with CMEMS and RTOFS; Guam CMEMS here, Guam RTOFS via run_guam_rtofs.sh.
# Step 1 is re-run up to 5 passes per job (each pass resumes), so network blips only cost a retry.
# Run under caffeinate so the Mac does not sleep:  caffeinate -i ./run_all.sh [end date]
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n model_comps python"
ROOT=$($PY -c 'from season_config import ROOT; print(ROOT)')
T0=$($PY -c 'from season_config import T0; print(T0.date())')
PLATFORM=$($PY -c 'from season_config import PLATFORM; print(PLATFORM)')
T1=${1:-$(date +%Y-%m-%d)}
LOGS=$ROOT/logs; mkdir -p "$LOGS"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOGS/run_all.log"; }

pair() {  # region model workers
    local region=$1 model=$2 workers=$3 f="$LOGS/p1_$1_$2.log"
    for pass in 1 2 3 4 5; do
        $PY step1_pair_model_profiles.py "$region" "$model" "$workers" > "$f" 2>&1
        local todo=$(grep -o '[0-9]* to do' "$f" | grep -o '^[0-9]*')
        local wrote=$(grep -o 'wrote [0-9]*' "$f" | grep -o '[0-9]*$')
        log "step1 $region/$model pass $pass: ${todo:-?} to do, wrote ${wrote:-?}"
        [ "${todo:-1}" = 0 ] && return
        grep -q DONE "$f" && [ "${wrote:-0}" = "${todo:-x}" ] && return
        [ "$pass" -gt 1 ] && [ "${wrote:-0}" = 0 ] && { log "step1 $region/$model: no progress, giving up"; return; }
    done
}

log "start ($T0 to $T1, root $ROOT)"
for region in caribbean gom hawaii guam mexico_pacific; do
    $PY step0_download_${PLATFORM}s.py "$region" "$T1" > "$LOGS/p0_$region.log" 2>&1
    log "step0 $region: $(grep -a -e 'profiles kept' -e ' profiles$' "$LOGS/p0_$region.log" | tail -1)"
done

pair caribbean cmems 6 &
pair gom cmems 6 &
pair hawaii cmems 6 &
pair guam cmems 6 &
pair mexico_pacific cmems 6 &
pair caribbean rtofs 2 &       # RTOFS: production Rutgers THREDDS, keep it gentle
pair gom rtofs 2 &
pair hawaii rtofs 2 &
pair mexico_pacific rtofs 2 &
wait
log "step1 finished"

"$(dirname "$0")/rerun_metrics_plots.sh" "$T1" | tee -a "$LOGS/run_all.log"
log "ALL DONE (Guam RTOFS: run_guam_rtofs.sh $T1)"
