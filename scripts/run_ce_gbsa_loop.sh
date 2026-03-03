#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/run_default.yaml}"

ts(){ date +"%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] [INFO] $*"; }
die(){ echo "[$(ts)] [ERROR] $*" >&2; exit 1; }

# --- tiny yaml getter ---
yaml_get() {
  local key="$1"
  python - "$CFG" "$key" <<'PY'
import sys
try:
    import yaml
except Exception:
    print("")
    sys.exit(0)

cfg_path=sys.argv[1]
key=sys.argv[2]
d=yaml.safe_load(open(cfg_path,'r'))
cur=d
for part in key.split('.'):
    if cur is None or part not in cur:
        print("")
        sys.exit(0)
    cur=cur[part]
print(cur if cur is not None else "")
PY
}

RUN_DIR="$(yaml_get run.run_dir)"
[[ -n "${RUN_DIR}" ]] || die "Cannot read run.run_dir from $CFG (is the YAML key correct?)"
RUN_DIR="$(python - <<PY
import os
print(os.path.expanduser("$RUN_DIR"))
PY
)"
export RUN_DIR
RUN_PATH="$RUN_DIR"

LOGFILE="/tmp/ce_gbsa_loop.$(date +%Y%m%d_%H%M%S).log"

# optional knobs (can be absent in YAML; we fallback)
MAX_ITERS="$(yaml_get ce.iters)"; MAX_ITERS="${MAX_ITERS:-30}"
BATCH="$(yaml_get ce.batch)"; BATCH="${BATCH:-24}"
ELITE_N="$(yaml_get rescore.elite_n)"; ELITE_N="${ELITE_N:-5}"
TOPM="$(yaml_get rescore.topM)"; TOPM="${TOPM:-5}"

log "config = $CFG"
log "run_dir = $RUN_PATH"
log "ce.iters=$MAX_ITERS, ce.batch=$BATCH, elite_n=$ELITE_N, topM=$TOPM"
log "log capture: $LOGFILE"

CKPT="$RUN_PATH/checkpoint.json"

iter_completed() {
  python - <<PY
import json
p="$CKPT"
try:
    j=json.load(open(p))
    print(int(j.get("iter_completed",0)))
except Exception:
    print(0)
PY
}

stop_reason() {
  python - <<PY
import json
p="$CKPT"
try:
    j=json.load(open(p))
    print((j.get("stop_reason") or "").strip())
except Exception:
    print("")
PY
}

need_manifest() {
  local iter="$1"
  local m="$RUN_PATH/rescore/iter_$(printf '%03d' "$iter")/manifest.jsonl"
  [[ -f "$m" ]] || die "missing manifest.jsonl for iter=$iter: $m"
}

run_ce_to_iter() {
  local target="$1"
  log "CE: run/resume to iter=$target"
  python -m experiments.ce_design --config "$CFG" --iters "$target" 2>&1 | tee -a "$LOGFILE"
}

run_mmgbsa_for_iter() {
  local iter="$1"
  log "MMGBSA: iter=$iter"
  python -m experiments.rescore_mmgbsa \
    --run_dir "$RUN_PATH" \
    --iter "$iter" \
    --topM "$TOPM" \
    --keep_tmp \
    --workers 1 \
    2>&1 | tee -a "$LOGFILE"
}

build_results_csv_for_feedback() {
  local iter="$1"
  local resdir="$RUN_PATH/rescore/iter_$(printf '%03d' "$iter")"
  local in_csv="$resdir/mmgbsa_results.csv"
  local out_csv="$resdir/results.csv"
  [[ -f "$in_csv" ]] || die "missing mmgbsa_results.csv: $in_csv"
  log "Build results.csv: $out_csv"

  python - "$iter" "$in_csv" "$out_csv" <<'PY'
import os, sys
import pandas as pd

iter_i = int(sys.argv[1])
in_csv = sys.argv[2]
out_csv = sys.argv[3]

df = pd.read_csv(in_csv)
if "seq" not in df.columns or "mmgbsa_dg_bind" not in df.columns:
    raise SystemExit("mmgbsa_results.csv missing required columns: seq/mmgbsa_dg_bind")

out = pd.DataFrame()
out["iter"] = iter_i
out["rank"] = df.get("rank", pd.Series(range(1, len(df)+1)))
out["seq"]  = df["seq"]

dock_col = None
for c in ["dock_score","score","score_vina","vina_score"]:
    if c in df.columns:
        dock_col = c
        break
out["dock_score"] = df[dock_col] if dock_col else float("nan")

# 给 feedback 用的统一列名（你的 gbsa_feedback_once 读 stub_mmgbsa）
out["stub_mmgbsa"] = pd.to_numeric(df["mmgbsa_dg_bind"], errors="coerce").fillna(1e9)

for c in ["receptor_pdb","pose_pdbqt","mmgbsa_workdir","mmgbsa_ok"]:
    out[c] = df[c] if c in df.columns else ""

out["receptor_exists"] = out["receptor_pdb"].apply(lambda p: bool(p) and os.path.exists(p))
out["pose_exists"] = out["pose_pdbqt"].apply(lambda p: bool(p) and os.path.exists(p))

out.to_csv(out_csv, index=False)
print("WROTE", out_csv, "rows=", len(out))
print(out.head(3).to_string(index=False))
PY
}

# --- feedback done marker / heuristics ---
feedback_flag() {
  local iter="$1"
  echo "$RUN_PATH/rescore/iter_$(printf '%03d' "$iter")/feedback.done"
}

feedback_done() {
  local iter="$1"
  local flag
  flag="$(feedback_flag "$iter")"
  if [[ -f "$flag" ]]; then
    return 0
  fi

  # 兼容你之前跑过但没写 flag 的情况：用时间戳粗略判断是否已经做过 feedback
  local resdir="$RUN_PATH/rescore/iter_$(printf '%03d' "$iter")"
  local results="$resdir/results.csv"
  local dist="$RUN_PATH/dist.json"
  local next="$RUN_PATH/next_batch.txt"
  if [[ -f "$results" && -f "$dist" && -f "$next" ]]; then
    # 如果 dist/next 比 results 更新，基本说明已做过 feedback（至少生成过下一批）
    if [[ "$dist" -nt "$results" && "$next" -nt "$results" ]]; then
      touch "$flag" || true
      return 0
    fi
  fi
  return 1
}

run_feedback() {
  local iter="$1"
  log "FEEDBACK: gbsa_feedback_once iter=$iter (batch=$BATCH elite_n=$ELITE_N)"
  python -m experiments.gbsa_feedback_once \
    --run_dir "$RUN_PATH" \
    --iter "$iter" \
    --batch "$BATCH" \
    --elite_n "$ELITE_N" \
    2>&1 | tee -a "$LOGFILE"

  [[ -f "$RUN_PATH/dist.json" ]] || die "dist.json not found after feedback"
  [[ -f "$RUN_PATH/next_batch.txt" ]] || die "next_batch.txt not found after feedback"

  touch "$(feedback_flag "$iter")" || true
}

# --- per-iter finalize: always do mmgbsa + results for the latest completed iter ---
finalize_iter_if_needed() {
  local iter="$1"
  [[ "$iter" -ge 1 ]] || return 0

  local resdir="$RUN_PATH/rescore/iter_$(printf '%03d' "$iter")"
  local mmcsv="$resdir/mmgbsa_results.csv"
  local results="$resdir/results.csv"

  need_manifest "$iter"

  if [[ ! -f "$mmcsv" ]]; then
    run_mmgbsa_for_iter "$iter"
  else
    log "MMGBSA: SKIP (exists): $mmcsv"
  fi

  if [[ ! -f "$results" ]]; then
    build_results_csv_for_feedback "$iter"
  else
    log "results.csv: SKIP (exists): $results"
  fi
}

log "==== START LOOP ===="

while true; do
  cur="$(iter_completed)"
  reason="$(stop_reason)"
  log "checkpoint: iter_completed=$cur stop_reason='${reason}'"

  # 1) 只要 CE 已经完成到 cur，就保证 cur 的 MMGBSA + results.csv 一定存在
  if [[ "$cur" -ge 1 ]]; then
    finalize_iter_if_needed "$cur"
  fi

  # 2) 如果还会继续跑下一轮，那么必须确保对 cur 做过 feedback（用于生成 next_batch + 更新 dist）
  if [[ -z "$reason" && "$cur" -ge 1 && "$cur" -lt "$MAX_ITERS" ]]; then
    if feedback_done "$cur"; then
      log "FEEDBACK: SKIP (already done): iter=$cur"
    else
      run_feedback "$cur"
    fi
  fi

  # 3) 终止条件：已经 stop 或已到 MAX_ITERS
  if [[ -n "$reason" ]]; then
    log "CE stopped (stop_reason=$reason). break."
    break
  fi
  if [[ "$cur" -ge "$MAX_ITERS" ]]; then
    log "Reached ce.iters=$MAX_ITERS. break."
    break
  fi

  # 4) 跑到下一轮（cur+1）
  target=$((cur+1))
  run_ce_to_iter "$target"

  new_cur="$(iter_completed)"
  reason="$(stop_reason)"
  log "after CE: iter_completed=$new_cur stop_reason='${reason}'"

  if [[ -n "$reason" || "$new_cur" -lt "$target" ]]; then
    log "CE stopped early. break."
    break
  fi
done
# after DONE LOOP
if [[ "$(yaml_get md.enabled)" == "True" || "$(yaml_get md.enabled)" == "true" ]]; then
  log "MD postprocess: run md_only"
  python -m experiments.ce_design --config "$CFG" --md_only 2>&1 | tee -a "$LOGFILE"
fi
log "==== DONE LOOP ===="
log "Run dir: $RUN_PATH"
log "Check: $RUN_PATH/checkpoint.json and $RUN_PATH/logs/ce_design.log"
echo
echo "[OK] Finished. Log: $LOGFILE"
