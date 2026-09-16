#!/usr/bin/env bash
# All cohorts in parallel; within a cohort, independent stages in parallel.
#
#   bash run_cloud.sh              # CRC GvHD IBD
#   bash run_cloud.sh CRC IBD
#
# Env: CONFIG, PY, RSCRIPT, RUN_ID, RUN_SHUFFLE (1), SKIP_GRAPH_DATA (1 = reuse graph_data.npz),
#      PIPELINE_DATA_ROOT / PIPELINE_SPLITS_ROOT / PIPELINE_RESULTS_DIR (see common.py).
# Logs: logs/<run_id>/<cohort>.<stage>.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$HERE/config.yaml}"
PY="${PY:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_SHUFFLE="${RUN_SHUFFLE:-1}"
SKIP_GRAPH_DATA="${SKIP_GRAPH_DATA:-1}"
COHORTS=("$@"); [[ ${#COHORTS[@]} -eq 0 ]] && COHORTS=(CRC GvHD IBD)
LOGDIR="$HERE/logs/$RUN_ID"; mkdir -p "$LOGDIR"

# One thread per worker, otherwise xgboost/torch processes oversubscribe the cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "run_id=$RUN_ID cohorts=${COHORTS[*]} config=$CONFIG logs=$LOGDIR"

stage () {   # stage <cohort> <name> <cmd...>
  local coh="$1" name="$2"; shift 2
  if ( cd "$HERE" && PIPELINE_COHORT="$coh" "$@" ) > "$LOGDIR/$coh.$name.log" 2>&1; then
    echo "  [ok]   $coh/$name"
  else
    echo "  [FAIL] $coh/$name -> $LOGDIR/$coh.$name.log"; return 1
  fi
}

run_cohort () {
  local C="$1" A=(--config "$CONFIG" --run-id "$RUN_ID")
  [[ "$SKIP_GRAPH_DATA" == "1" ]] || stage "$C" build "$PY" data.py build || return 1
  stage "$C" preprocess "$PY" data.py preprocess "${A[@]}" || return 1

  local pids=()
  stage "$C" experiment  "$PY" experiment.py "${A[@]}" & pids+=($!)
  stage "$C" w_threshold "$PY" analyses.py w_threshold "${A[@]}" & pids+=($!)
  stage "$C" stability   "$PY" analyses.py stability "${A[@]}" & pids+=($!)
  [[ "$RUN_SHUFFLE" == "1" ]] && { stage "$C" shuffle "$PY" analyses.py shuffle "${A[@]}" & pids+=($!); }
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [[ $rc -eq 0 ]] || { echo "[$C] a parallel stage failed"; return 1; }

  stage "$C" describe "$PY" data.py describe "${A[@]}"
  if command -v "$RSCRIPT" >/dev/null 2>&1; then
    for s in data.R results.R analyses.R; do
      stage "$C" "$s" "$RSCRIPT" "plots/$s" "$RUN_ID" "${C}_outputs"
    done
  else
    echo "  [warn] $C: Rscript not on PATH; plots skipped."
  fi
  echo "[$C] DONE"
}

START=$(date +%s); FAILED=0; cpids=()
for C in "${COHORTS[@]}"; do run_cohort "$C" & cpids+=($!); done
for p in "${cpids[@]}"; do wait "$p" || FAILED=1; done
echo "elapsed $(( ($(date +%s) - START) / 60 )) min | run_id $RUN_ID | $([[ $FAILED -eq 0 ]] && echo OK || echo FAILED)"
exit $FAILED
