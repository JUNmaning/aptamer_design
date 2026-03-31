#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/preflight.sh [configs/run_default.yaml] [ITER] [TOPM] [MMGBSA_WORKERS]
#
# Examples:
#   bash scripts/preflight.sh configs/run_default.yaml
#   bash scripts/preflight.sh configs/run_default.yaml 1 5 4
#
# Behavior:
# - If RUN_DIR/rescore/iter_XXX/manifest.jsonl exists for ITER, this script runs MMGBSA once.
# - If MMGBSA produces NO successful decoys, it auto-cleans that iter's mmgbsa workdir/results and retries once.
# - Always writes two global summaries to RUN_DIR:
#     summary_by_vina.csv
#     summary_by_mmgbsa.csv

CFG="${1:-configs/run_default.yaml}"
ITER="${2:-1}"
TOPM_IN="${3:-}"
MMGBSA_WORKERS_IN="${4:-}"

ts(){ date +"%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] [INFO] $*"; }
warn(){ echo "[$(ts)] [WARN] $*" >&2; }
die(){ echo "[$(ts)] [ERROR] $*" >&2; exit 1; }

# --- tiny yaml getter (requires PyYAML) ---
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
with open(cfg_path, "r") as f:
    d=yaml.safe_load(f)

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
[[ -n "${RUN_DIR}" ]] || die "Cannot read run.run_dir from $CFG (PyYAML installed? YAML key correct?)"
RUN_DIR="$(python - <<PY
import os
print(os.path.expanduser("$RUN_DIR"))
PY
)"
export RUN_DIR
SIMRNA_CONFIG="$(yaml_get eval.simrna_config)"
export SIMRNA_CONFIG
# Defaults from YAML (fallback if missing)
TOPM="$(yaml_get rescore.topM)"; TOPM="${TOPM:-5}"
if [[ -n "${TOPM_IN}" ]]; then TOPM="${TOPM_IN}"; fi

# Default MMGBSA workers: min(4, cpu_count)
CPU_N="$(python - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
MMGBSA_WORKERS="${MMGBSA_WORKERS_IN:-}"
if [[ -z "${MMGBSA_WORKERS}" ]]; then
  if [[ "${CPU_N}" -le 1 ]]; then
    MMGBSA_WORKERS=1
  elif [[ "${CPU_N}" -le 4 ]]; then
    MMGBSA_WORKERS="${CPU_N}"
  else
    MMGBSA_WORKERS=4
  fi
fi

log "CFG=$CFG"
log "RUN_DIR=$RUN_DIR"
log "ITER=$ITER TOPM=$TOPM MMGBSA_WORKERS=$MMGBSA_WORKERS"

echo "[1/8] Python env"
python -V
python -c "import rdkit, numpy; print('rdkit/numpy OK')"
python -c "import yaml; print('pyyaml OK')" >/dev/null

echo "[2/8] ViennaRNA"
command -v RNAfold >/dev/null && echo "RNAfold OK" || (echo "RNAfold missing" && exit 1)

echo "[3/8] SimRNA"
command -v SimRNA >/dev/null && echo "SimRNA OK" || (echo "SimRNA missing" && exit 1)
command -v SimRNA_trafl2pdbs >/dev/null && echo "SimRNA_trafl2pdbs OK" || (echo "SimRNA_trafl2pdbs missing" && exit 1)
test -f "${SIMRNA_CONFIG:-}" && echo "SIMRNA_CONFIG OK: $SIMRNA_CONFIG" || (echo "SIMRNA_CONFIG invalid" && exit 1)
test -d "${SIMRNA_HOME:-}/data" && echo "SimRNA data OK" || (echo "SimRNA data dir missing" && exit 1)

echo "[4/8] OpenBabel"
command -v obabel >/dev/null && echo "obabel OK" || (echo "obabel missing" && exit 1)

echo "[5/8] Vina"
command -v vina >/dev/null && echo "vina OK" || (echo "vina missing" && exit 1)

echo "[6/8] AmberTools (MMGBSA)"
command -v tleap >/dev/null && echo "tleap OK" || (echo "tleap missing" && exit 1)
command -v MMPBSA.py >/dev/null && echo "MMPBSA.py OK" || (echo "MMPBSA.py missing" && exit 1)

echo "[7/8] Quick fold+dock smoke test (no CE)"
python -m experiments.dock_rank_batch >/dev/null
echo "SMOKE TEST PASSED"

# -----------------------------
# MMGBSA parallel + retry once
# -----------------------------
run_mmgbsa_once() {
  local iter="$1"
  log "MMGBSA run: iter=$iter (workers=$MMGBSA_WORKERS topM=$TOPM)"
  python -m experiments.rescore_mmgbsa \
    --run_dir "$RUN_DIR" \
    --iter "$iter" \
    --topM "$TOPM" \
    --workers "$MMGBSA_WORKERS" \
    --keep_tmp \
    2>&1 | tee "/tmp/preflight_mmgbsa_iter_${iter}.log"
}

mmgbsa_ok_any() {
  local iter="$1"
  python - "$RUN_DIR" "$iter" <<'PY'
import sys
from pathlib import Path
import pandas as pd

run_dir = Path(sys.argv[1])
iter_i = int(sys.argv[2])
resdir = run_dir / "rescore" / f"iter_{iter_i:03d}"
p = resdir / "mmgbsa_results.csv"
if not p.exists():
    print("0")
    raise SystemExit(0)
try:
    df = pd.read_csv(p)
except Exception:
    print("0")
    raise SystemExit(0)

if "mmgbsa_ok" not in df.columns:
    ok = (pd.to_numeric(df.get("mmgbsa_dg_bind", 1e9), errors="coerce") < 1e8).any()
    print("1" if ok else "0")
else:
    ok = df["mmgbsa_ok"].astype(str).str.lower().isin(["true","1","yes"]).any()
    print("1" if ok else "0")
PY
}

cleanup_mmgbsa_iter() {
  local iter="$1"
  local resdir="$RUN_DIR/rescore/iter_$(printf '%03d' "$iter")"
  log "Cleanup MMGBSA artifacts for retry: $resdir/mmgbsa + results files"
  rm -rf "$resdir/mmgbsa" || true
  rm -f  "$resdir/mmgbsa_results.csv" "$resdir/mmgbsa_results.jsonl" || true
}

manifest_path="$RUN_DIR/rescore/iter_$(printf '%03d' "$ITER")/manifest.jsonl"
if [[ -f "$manifest_path" ]]; then
  run_mmgbsa_once "$ITER" || true
  if [[ "$(mmgbsa_ok_any "$ITER")" != "1" ]]; then
    warn "MMGBSA had no successful decoys on iter=$ITER -> clean & retry once"
    cleanup_mmgbsa_iter "$ITER"
    run_mmgbsa_once "$ITER" || true
    if [[ "$(mmgbsa_ok_any "$ITER")" != "1" ]]; then
      warn "MMGBSA retry still produced no successful decoys for iter=$ITER. Check: /tmp/preflight_mmgbsa_iter_${ITER}.log"
    else
      log "MMGBSA retry succeeded for iter=$ITER"
    fi
  else
    log "MMGBSA succeeded for iter=$ITER"
  fi
else
  warn "Skip MMGBSA: manifest not found for iter=$ITER: $manifest_path"
fi

# -----------------------------
# Global summaries
# -----------------------------
echo "[8/8] Build global summaries in RUN_DIR"
python - "$RUN_DIR" <<'PY'
import csv, json, math, sys
from pathlib import Path

run_dir = Path(sys.argv[1]).expanduser().resolve()
rescore_root = run_dir / "rescore"
iters = []
if rescore_root.exists():
    for d in sorted(rescore_root.glob("iter_*")):
        try:
            it = int(d.name.split("_")[-1])
            iters.append((it, d))
        except Exception:
            continue

def best_by_vina(iter_dir: Path):
    m = iter_dir / "manifest.jsonl"
    if not m.exists():
        return None
    best = None
    with m.open("r", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sc = rec.get("score_vina", rec.get("dock_score", rec.get("score", None)))
            try:
                sc = float(sc)
            except Exception:
                continue
            if best is None or sc < best["vina_score"]:
                best = {
                    "iter": int(rec.get("iter", int(iter_dir.name.split("_")[-1]))),
                    "rank": int(rec.get("rank", -1)),
                    "seq": rec.get("seq", ""),
                    "vina_score": sc,
                    "best_frame": rec.get("best_frame", ""),
                    "receptor_pdb": rec.get("receptor_pdb", ""),
                    "pose_pdbqt": rec.get("pose_pdbqt", ""),
                }
    return best

def best_by_mmgbsa(iter_dir: Path):
    p = iter_dir / "mmgbsa_results.csv"
    if not p.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty or "seq" not in df.columns:
        return None
    if "mmgbsa_ok" in df.columns:
        ok = df["mmgbsa_ok"].astype(str).str.lower().isin(["true","1","yes"])
        df_ok = df[ok].copy()
        if not df_ok.empty:
            df = df_ok
    if "mmgbsa_dg_bind" not in df.columns:
        return None
    df["mmgbsa_dg_bind"] = pd.to_numeric(df["mmgbsa_dg_bind"], errors="coerce")
    df = df[df["mmgbsa_dg_bind"].notna()].copy()
    if df.empty:
        return None
    row = df.sort_values("mmgbsa_dg_bind", ascending=True).iloc[0].to_dict()
    it = int(iter_dir.name.split("_")[-1])
    return {
        "iter": it,
        "rank": int(row.get("rank", -1)) if not (row.get("rank", None) is None or (isinstance(row.get("rank", None), float) and math.isnan(row["rank"]))) else -1,
        "seq": row.get("seq", ""),
        "mmgbsa_dg_bind": float(row.get("mmgbsa_dg_bind", 1e9)),
        "score_vina": float(row.get("score_vina", row.get("dock_score", float("nan")))) if row.get("score_vina", row.get("dock_score", None)) is not None else float("nan"),
        "best_frame": row.get("best_frame", ""),
        "mmgbsa_workdir": row.get("mmgbsa_workdir", ""),
        "mmgbsa_log": row.get("mmgbsa_log", ""),
        "mmgbsa_error": row.get("mmgbsa_error", ""),
    }

vina_rows = []
mmgbsa_rows = []
for it, d in iters:
    bv = best_by_vina(d)
    if bv:
        vina_rows.append(bv)
    bm = best_by_mmgbsa(d)
    if bm:
        mmgbsa_rows.append(bm)

out_vina = run_dir / "summary_by_vina.csv"
out_mmgbsa = run_dir / "summary_by_mmgbsa.csv"

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

vina_rows.sort(key=lambda r: int(r["iter"]))
mmgbsa_rows.sort(key=lambda r: int(r["iter"]))

write_csv(out_vina, vina_rows, ["iter","rank","seq","vina_score","best_frame","receptor_pdb","pose_pdbqt"])
write_csv(out_mmgbsa, mmgbsa_rows, ["iter","rank","seq","mmgbsa_dg_bind","score_vina","best_frame","mmgbsa_workdir","mmgbsa_log","mmgbsa_error"])

print("WROTE", out_vina)
print("WROTE", out_mmgbsa)
print("vina iters:", len(vina_rows), "mmgbsa iters:", len(mmgbsa_rows))
PY

log "Done."
