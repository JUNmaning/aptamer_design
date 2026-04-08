#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/run_default.yaml}"

ts(){ date +"%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] [INFO] $*"; }
die(){ echo "[$(ts)] [ERROR] $*" >&2; exit 1; }

yaml_get() {
  local key="$1"
  python - "$CFG" "$key" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
key = sys.argv[2]
d = yaml.safe_load(open(cfg_path, 'r'))
cur = d
for part in key.split('.'):
    if cur is None or part not in cur:
        print("")
        sys.exit(0)
    cur = cur[part]
print(cur if cur is not None else "")
PY
}

RUN_DIR="$(yaml_get run.run_dir)"
[[ -n "${RUN_DIR}" ]] || die "run.run_dir not defined in YAML"
RUN_DIR="$(python - <<PY
import os
print(os.path.expanduser("$RUN_DIR"))
PY
)"

export RUN_PATH="$RUN_DIR"

MAX_ITERS="$(yaml_get ce.iters)"
BATCH="$(yaml_get ce.batch)"

TOP_FRAC="$(yaml_get rescore.top_frac)"
ELITE_FRAC="$(yaml_get rescore.elite_frac)"

[[ -n "$TOP_FRAC" ]] || die "rescore.top_frac must be defined"
[[ -n "$ELITE_FRAC" ]] || die "rescore.elite_frac must be defined"

TOPM=$(python - <<PY
import math
print(max(1, int(math.ceil(int("$BATCH") * float("$TOP_FRAC")))))
PY
)

ELITE_N=$(python - <<PY
import math
print(max(1, int(math.ceil(int("$TOPM") * float("$ELITE_FRAC")))))
PY
)

log "batch=$BATCH"
log "top_frac=$TOP_FRAC → topM=$TOPM"
log "elite_frac=$ELITE_FRAC → elite_n=$ELITE_N"

CKPT="$RUN_PATH/checkpoint.json"

iter_completed() {
  python - <<PY
import json, os
if not os.path.exists("$CKPT"):
    print(0)
else:
    print(json.load(open("$CKPT")).get("iter_completed",0))
PY
}

run_ce_to_iter() {
  local target="$1"
  python -m experiments.ce_design --config "$CFG" --iters "$target"
}

run_mmgbsa_for_iter() {
  local iter="$1"
  python -m experiments.rescore_mmgbsa \
    --run_dir "$RUN_PATH" \
    --iter "$iter" \
    --topM "$TOPM"
}

run_feedback() {
  local iter="$1"
  python -m experiments.gbsa_feedback_once \
    --run_dir "$RUN_PATH" \
    --iter "$iter" \
    --batch "$BATCH" \
    --elite_n "$ELITE_N"
}

log "==== START CE↔GBSA LOOP ===="

while true; do
  cur="$(iter_completed)"
  log "current_iter=$cur"

  if [[ "$cur" -ge "$MAX_ITERS" ]]; then
    log "Reached max iterations."
    break
  fi

  next=$((cur+1))

  run_ce_to_iter "$next"

  run_mmgbsa_for_iter "$next"

  run_feedback "$next"

done

log "==== LOOP DONE ===="
