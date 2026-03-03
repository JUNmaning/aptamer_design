#!/usr/bin/env bash
set -euo pipefail

CFG_IN="${1:-configs/run_default.yaml}"

if [[ ! -f "$CFG_IN" ]]; then
  echo "[FATAL] config not found: $CFG_IN" >&2
  exit 1
fi

ts() { date +"%F %T"; }

py_get_run_dir() {
python - <<'PY' "$1"
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    d = yaml.safe_load(f)
print(d["run"]["run_dir"])
PY
}

py_write_stage_cfg() {
# args: in_cfg out_cfg iters resume_bool
python - <<'PY' "$1" "$2" "$3" "$4"
import sys, yaml
in_cfg, out_cfg, iters_s, resume_s = sys.argv[1:5]
iters = int(iters_s)
resume = resume_s.lower() in ("1","true","yes","y")
with open(in_cfg, "r") as f:
    d = yaml.safe_load(f)

d.setdefault("run", {})
d["run"]["resume"] = resume

d.setdefault("ce", {})
d["ce"]["iters"] = iters

with open(out_cfg, "w") as f:
    yaml.safe_dump(d, f, sort_keys=False)

print(f"[OK] wrote stage config: {out_cfg} (iters={iters}, resume={resume})")
PY
}

echo "[$(ts)] [INFO] input config: $CFG_IN"

RUN_DIR="$(py_get_run_dir "$CFG_IN")"
export RUN_DIR
echo "[$(ts)] [INFO] run_dir from config: $RUN_DIR"

STAGE1_CFG="/tmp/ce_stage_iter1.yaml"
STAGE2_CFG="/tmp/ce_stage_iter2_resume.yaml"

py_write_stage_cfg "$CFG_IN" "$STAGE1_CFG" 1 false
py_write_stage_cfg "$CFG_IN" "$STAGE2_CFG" 2 true

LOG1="/tmp/ce_iter1_$(basename "$RUN_DIR").log"
LOG2="/tmp/ce_iter2_$(basename "$RUN_DIR").log"

ITER_DIR="$RUN_DIR/rescore/iter_001"
MANIFEST="$ITER_DIR/manifest.jsonl"

# ---- Stage 1: CE iters=1 (only if manifest missing) ----
if [[ -f "$MANIFEST" ]]; then
  echo "[$(ts)] [INFO] Stage1: SKIP (manifest already exists): $MANIFEST"
else
  echo "[$(ts)] [INFO] Stage1: ce_design iters=1"
  python -m experiments.ce_design --config "$STAGE1_CFG" 2>&1 | tee "$LOG1"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "[FATAL] manifest not found after Stage1: $MANIFEST" >&2
    echo "Hint: ensure rescore.enabled=true in config." >&2
    exit 1
  fi
fi

echo "[$(ts)] [OK] manifest exists: $MANIFEST"
echo "[$(ts)] [INFO] manifest head:"
head -n 2 "$MANIFEST" || true

# ---- Stage 2: MM/GBSA ----
echo "[$(ts)] [INFO] Stage2: rescore_mmgbsa (real)"
python -m experiments.rescore_mmgbsa \
  --run_dir "$RUN_DIR" \
  --iter 1 \
  --topM 5 \
  --workers 1 \
  --igb 5 \
  --saltcon 0.150 \
  --charge_method bcc

MMCSV="$ITER_DIR/mmgbsa_results.csv"
if [[ ! -f "$MMCSV" ]]; then
  echo "[FATAL] mmgbsa_results.csv not found: $MMCSV" >&2
  exit 1
fi
echo "[$(ts)] [OK] mmgbsa_results.csv exists: $MMCSV"
echo "[$(ts)] [INFO] mmgbsa_results.csv head:"
head -n 3 "$MMCSV" || true

# ---- Stage 3: build results.csv for feedback ----
echo "[$(ts)] [INFO] Stage3: build $ITER_DIR/results.csv (map mmgbsa_dg_bind -> stub_mmgbsa)"
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

run_dir = Path(os.environ["RUN_DIR"])
iter_dir = run_dir / "rescore" / "iter_001"
src = iter_dir / "mmgbsa_results.csv"
dst = iter_dir / "results.csv"

df = pd.read_csv(src)

out = pd.DataFrame()
out["iter"] = 1
out["rank"] = df["rank"] if "rank" in df.columns else range(1, len(df)+1)
out["seq"] = df["seq"]

dock_col = None
for c in ["dock_score", "score", "score_vina", "vina_score"]:
    if c in df.columns:
        dock_col = c
        break
out["dock_score"] = df[dock_col] if dock_col else float("nan")

if "mmgbsa_dg_bind" not in df.columns:
    raise RuntimeError("mmgbsa_dg_bind not found in mmgbsa_results.csv")
out["stub_mmgbsa"] = df["mmgbsa_dg_bind"]

for c in ["receptor_pdb", "pose_pdbqt"]:
    out[c] = df[c] if c in df.columns else ""

out["receptor_exists"] = out["receptor_pdb"].apply(lambda p: bool(p) and os.path.exists(p))
out["pose_exists"] = out["pose_pdbqt"].apply(lambda p: bool(p) and os.path.exists(p))

out.to_csv(dst, index=False)
print("[OK] wrote", dst, "rows=", len(out))
print(out.head(3).to_string(index=False))
PY

RESULTS="$ITER_DIR/results.csv"
if [[ ! -f "$RESULTS" ]]; then
  echo "[FATAL] results.csv not found: $RESULTS" >&2
  exit 1
fi

# ---- Stage 4: feedback ----
echo "[$(ts)] [INFO] Stage4: gbsa_feedback_once (update dist.json + next_batch.txt)"
python -m experiments.gbsa_feedback_once \
  --run_dir "$RUN_DIR" \
  --iter 1 \
  --batch 24 \
  --elite_n 5

if [[ ! -f "$RUN_DIR/dist.json" ]]; then
  echo "[FATAL] dist.json not found after feedback: $RUN_DIR/dist.json" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/next_batch.txt" ]]; then
  echo "[FATAL] next_batch.txt not found after feedback: $RUN_DIR/next_batch.txt" >&2
  exit 1
fi

echo "[$(ts)] [OK] dist.json updated: $RUN_DIR/dist.json"
echo "[$(ts)] [OK] next_batch.txt written: $RUN_DIR/next_batch.txt"
echo "[$(ts)] [INFO] next_batch head:"
head -n 5 "$RUN_DIR/next_batch.txt" || true

# ---- Stage 5: resume CE to iters=2 ----
echo "[$(ts)] [INFO] Stage5: resume ce_design to iters=2"
python -m experiments.ce_design --config "$STAGE2_CFG" 2>&1 | tee "$LOG2"

echo "[$(ts)] [DONE] One round complete (iter1 -> mmgbsa -> dist update -> iter2 resume)."
echo "Logs:"
echo "  $LOG1"
echo "  $LOG2"
