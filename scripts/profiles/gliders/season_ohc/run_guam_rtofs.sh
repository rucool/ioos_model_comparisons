#!/usr/bin/env bash
# Guam RTOFS: pairs locally -> extract on baffin (via boardwalk) -> copy back -> metrics + plots.
# Guam RTOFS exists only as the binary-converted NetCDFs in /home/hurricaneadm/data/rtofs_archv on baffin.
# Needs ssh keys for boardwalk and baffin. Already-extracted pairs are skipped.
# usage: ./run_guam_rtofs.sh [end date for plot titles, default today]
set -euo pipefail
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n model_comps python"
ROOT=$($PY -c 'from season_config import ROOT; print(ROOT)')
T0=$($PY -c 'from season_config import T0; print(T0.date())')
T1=${1:-$(date +%Y-%m-%d)}
REMOTE=${GUAM_REMOTE:-michaesm@baffin.marine.rutgers.edu}
SSH="ssh -o BatchMode=yes -J ${GUAM_JUMP:-boardwalk}"
RDIR=guam_rtofs_extract                                   # under the remote user's home
RPY=/home/hurricaneadm/miniconda3/envs/model_comps/bin/python
LOGS=$ROOT/logs; mkdir -p "$LOGS"

$PY guam_rtofs_pairs.py
$SSH "$REMOTE" "mkdir -p $RDIR/out"
rsync -az -e "$SSH" extract_guam_rtofs.py "$ROOT/guam_rtofs_pairs.csv" "$REMOTE:$RDIR/"
$SSH "$REMOTE" "cd $RDIR && nice -n 19 $RPY extract_guam_rtofs.py guam_rtofs_pairs.csv out"
rsync -az -e "$SSH" "$REMOTE:$RDIR/out/" "$ROOT/guam/"
$PY step2_calculate_metrics.py guam rtofs > "$LOGS/p2_guam_rtofs.log" 2>&1
$PY step3_plot_hexbin.py Guam rtofs "$T0" "$T1" guam > "$LOGS/p3_guam_rtofs.log" 2>&1
grep -a -e 'pass qc' -e 'Saved plot' "$LOGS/p3_guam_rtofs.log"
