#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/run_default.yaml}"

ts(){ date +"%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] [INFO] $*"; }
warn(){ echo "[$(ts)] [WARN] $*" >&2; }
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
[[ -n "${RUN_DIR}" ]] || die "Cannot read run.run_dir from $CFG (is the YAML key correct?)"
RUN_DIR="$(python - <<PY
import os
print(os.path.expanduser("$RUN_DIR"))
PY
)"
export RUN_DIR
RUN_PATH="$RUN_DIR"

LOGFILE="/tmp/ce_gbsa_loop.$(date +%Y%m%d_%H%M%S).log"

# -----------------------------
# core knobs
# -----------------------------
MAX_ITERS="$(yaml_get ce.iters)"; MAX_ITERS="${MAX_ITERS:-30}"
BATCH="$(yaml_get ce.batch)"; BATCH="${BATCH:-24}"

# rescore candidate selection: ratio-first, fallback-to-fixed
TOPM_RAW="$(yaml_get rescore.topM)"; TOPM_RAW="${TOPM_RAW:-5}"
TOP_FRAC="$(yaml_get rescore.top_frac)"; TOP_FRAC="${TOP_FRAC:-}"

ELITE_N_RAW="$(yaml_get rescore.elite_n)"; ELITE_N_RAW="${ELITE_N_RAW:-5}"
ELITE_FRAC="$(yaml_get rescore.elite_frac)"; ELITE_FRAC="${ELITE_FRAC:-}"

# external dual-stop knobs (read from YAML monitor.external_stop.*)
GBSA_PATIENCE="$(yaml_get monitor.external_stop.gbsa_patience)"; GBSA_PATIENCE="${GBSA_PATIENCE:-5}"
GBSA_MIN_IMPROVE="$(yaml_get monitor.external_stop.gbsa_min_improve)"; GBSA_MIN_IMPROVE="${GBSA_MIN_IMPROVE:-1.0}"

DOCK_PATIENCE="$(yaml_get monitor.external_stop.dock_patience)"; DOCK_PATIENCE="${DOCK_PATIENCE:-6}"
DOCK_MIN_IMPROVE="$(yaml_get monitor.external_stop.dock_min_improve)"; DOCK_MIN_IMPROVE="${DOCK_MIN_IMPROVE:-0.10}"

calc_topm() {
  python - <<PY
import math
batch = int("$BATCH")
top_frac = "$TOP_FRAC".strip()
topm_raw = int("$TOPM_RAW")
if top_frac:
    try:
        v = max(1, int(math.ceil(batch * float(top_frac))))
        print(v)
    except Exception:
        print(topm_raw)
else:
    print(topm_raw)
PY
}

calc_elite_n() {
  python - <<PY
import math
topm = int("$TOPM")
elite_frac = "$ELITE_FRAC".strip()
elite_n_raw = int("$ELITE_N_RAW")
if elite_frac:
    try:
        v = max(1, int(math.ceil(topm * float(elite_frac))))
        print(v)
    except Exception:
        print(elite_n_raw)
else:
    print(elite_n_raw)
PY
}

TOPM="$(calc_topm)"
ELITE_N="$(calc_elite_n)"

log "config = $CFG"
log "run_dir = $RUN_PATH"
log "ce.iters=$MAX_ITERS, ce.batch=$BATCH"
log "rescore selection: topM=$TOPM (top_frac=${TOP_FRAC:-<none>}, topM_raw=$TOPM_RAW)"
log "feedback elite: elite_n=$ELITE_N (elite_frac=${ELITE_FRAC:-<none>}, elite_n_raw=$ELITE_N_RAW)"
log "GBSA stop: patience=$GBSA_PATIENCE min_improve=$GBSA_MIN_IMPROVE"
log "Dock stop: patience=$DOCK_PATIENCE min_improve=$DOCK_MIN_IMPROVE"
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
  log "MMGBSA: iter=$iter (topM=$TOPM)"
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

  local resdir="$RUN_PATH/rescore/iter_$(printf '%03d' "$iter")"
  local results="$resdir/results.csv"
  local dist="$RUN_PATH/dist.json"
  local next="$RUN_PATH/next_batch.txt"
  if [[ -f "$results" && -f "$dist" && -f "$next" ]]; then
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

# -----------------------------
# stop helpers
# -----------------------------
gbsa_stop_triggered() {
  python - "$RUN_PATH" "$GBSA_PATIENCE" "$GBSA_MIN_IMPROVE" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run_path = Path(sys.argv[1])
pat = int(sys.argv[2])
thr = float(sys.argv[3])

rescore_root = run_path / "rescore"
if not rescore_root.exists():
    print("0")
    raise SystemExit(0)

vals = []
for d in sorted(rescore_root.glob("iter_*")):
    p = d / "results.csv"
    if not p.exists():
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if "stub_mmgbsa" not in df.columns:
        continue
    s = pd.to_numeric(df["stub_mmgbsa"], errors="coerce").dropna()
    s = s[(s > -500) & (s < 100)]
    if s.empty:
        continue
    vals.append((int(d.name.split("_")[-1]), float(s.min())))

if len(vals) < pat:
    print("0")
    raise SystemExit(0)

tail = vals[-pat:]
best_old = tail[0][1]
best_new = min(v for _, v in tail)

# more negative is better
improve = best_old - best_new
if improve < thr:
    print("1")
else:
    print("0")
PY
}

dock_stop_triggered() {
  python - "$RUN_PATH" "$DOCK_PATIENCE" "$DOCK_MIN_IMPROVE" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run_path = Path(sys.argv[1])
pat = int(sys.argv[2])
thr = float(sys.argv[3])

hist = run_path / "history.csv"
if not hist.exists():
    print("0")
    raise SystemExit(0)

try:
    df = pd.read_csv(hist)
except Exception:
    print("0")
    raise SystemExit(0)

if "best_score" not in df.columns or "iter" not in df.columns:
    print("0")
    raise SystemExit(0)

df["best_score"] = pd.to_numeric(df["best_score"], errors="coerce")
df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
df = df.dropna(subset=["iter", "best_score"]).sort_values("iter")

if len(df) < pat:
    print("0")
    raise SystemExit(0)

tail = df.tail(pat)
best_old = float(tail.iloc[0]["best_score"])
best_new = float(tail["best_score"].min())

# lower is better
improve = best_old - best_new
if improve < thr:
    print("1")
else:
    print("0")
PY
}

set_external_stop_reason() {
  local reason="$1"
  python - "$CKPT" "$reason" <<'PY'
import json, sys, os
p = sys.argv[1]
reason = sys.argv[2]
obj = {}
if os.path.exists(p):
    try:
        obj = json.load(open(p))
    except Exception:
        obj = {}
obj["stop_reason"] = reason
with open(p, "w") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)
print(reason)
PY
}

log "==== START LOOP ===="

while true; do
  cur="$(iter_completed)"
  reason="$(stop_reason)"
  log "checkpoint: iter_completed=$cur stop_reason='${reason}'"

  # 1) ensure MMGBSA/results exist for current iter
  if [[ "$cur" -ge 1 ]]; then
    finalize_iter_if_needed "$cur"
  fi

  # 2) dual plateau stop check
  if [[ -z "$reason" && "$cur" -ge 1 ]]; then
    if [[ "$(gbsa_stop_triggered)" == "1" ]]; then
      reason="$(set_external_stop_reason "gbsa_plateau(patience=${GBSA_PATIENCE}, min_improve=${GBSA_MIN_IMPROVE})")"
      log "External stop triggered: $reason"
    elif [[ "$(dock_stop_triggered)" == "1" ]]; then
      reason="$(set_external_stop_reason "dock_plateau(patience=${DOCK_PATIENCE}, min_improve=${DOCK_MIN_IMPROVE})")"
      log "External stop triggered: $reason"
    fi
  fi

  # 3) if continuing, feedback current iter
  if [[ -z "$reason" && "$cur" -ge 1 && "$cur" -lt "$MAX_ITERS" ]]; then
    if feedback_done "$cur"; then
      log "FEEDBACK: SKIP (already done): iter=$cur"
    else
      run_feedback "$cur"
    fi
  fi

  # 4) stop conditions
  if [[ -n "$reason" ]]; then
    log "CE stopped (stop_reason=$reason). break."
    break
  fi
  if [[ "$cur" -ge "$MAX_ITERS" ]]; then
    log "Reached ce.iters=$MAX_ITERS. break."
    break
  fi

  # 5) run next iter
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
