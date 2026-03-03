"""
Generate rescore manifest.jsonl from topN.csv (ce_design output).

Why:
- rescore_mmgbsa_stub expects manifest fields: dock_score or score; if missing => 1e9.
- topN.csv uses columns like score / best_score and paths best_receptor_pdb / best_dock_pose_pdbqt.
- This script normalizes schema and writes:
    <run_dir>/rescore/iter_XXX/manifest.jsonl
    <run_dir>/rescore/iter_XXX/best.json

Example:
python -m experiments.rescore_manifest_from_topN \
  --run_dir runs/ce_run1 \
  --topN_csv topN.csv \
  --iter_out 1 \
  --topM 5 \
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _resolve_topn_path(run_dir: Path, topn_csv: str) -> Path:
    """
    Robust path resolution:
    1) if absolute -> use it
    2) if exists relative to CWD -> use it
    3) else try run_dir / topn_csv
    4) else try run_dir / "topN.csv"
    """
    p = Path(topn_csv)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    cand = run_dir / p
    if cand.exists():
        return cand
    fallback = run_dir / "topN.csv"
    return fallback


def _read_topn_rows(topn_path: Path) -> List[Dict[str, str]]:
    if not topn_path.exists():
        raise FileNotFoundError(f"topN.csv not found: {topn_path}")
    with topn_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _pick_dock_score(row: Dict[str, str]) -> Tuple[float, str]:
    """
    Prefer these columns, in order:
    - best_score (most explicit "best pose score")
    - score      (ce_design main score, often equal best_score)
    - score_min / score_mean / score_p10 / score_boltz (fallback)
    """
    candidates = [
        ("best_score", row.get("best_score")),
        ("score", row.get("score")),
        ("score_min", row.get("score_min")),
        ("score_mean", row.get("score_mean")),
        ("score_p10", row.get("score_p10")),
        ("score_boltz", row.get("score_boltz")),
    ]
    for name, raw in candidates:
        v = _to_float(raw)
        if v is not None:
            return float(v), name
    # last resort
    return 1e9, "missing->1e9"


def _norm_seq(seq: str) -> str:
    s = (seq or "").strip().upper()
    return s.replace("T", "U")


def _row_to_manifest_item(row: Dict[str, str], iter_out: int, rank: int) -> Dict[str, object]:
    seq = _norm_seq(row.get("seq", "") or row.get("sequence", ""))
    dock_score, src = _pick_dock_score(row)

    receptor_pdb = row.get("best_receptor_pdb") or row.get("receptor_pdb") or ""
    pose_pdbqt = row.get("best_dock_pose_pdbqt") or row.get("pose_pdbqt") or ""
    dot_bracket = row.get("dot_bracket") or None
    if dot_bracket is not None:
        dot_bracket = dot_bracket.strip() or None

    item: Dict[str, object] = {
        "iter": int(iter_out),
        "rank": int(rank),
        "seq": seq,
        "dock_score": float(dock_score),
        "receptor_pdb": receptor_pdb,
        "pose_pdbqt": pose_pdbqt,
        "dot_bracket": dot_bracket,
        # keep helpful traceability fields if present
        "eval_hash": row.get("eval_hash") or row.get("hash") or None,
        "dock_score_source": src,
        "topn_row": rank,
    }
    return item


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/ce_run1")
    ap.add_argument("--topN_csv", default="topN.csv", help="path to topN.csv (relative allowed)")
    ap.add_argument("--iter_out", type=int, required=True, help="write into rescore/iter_XXX")
    ap.add_argument("--topM", type=int, default=5, help="how many rows from topN to export")
    ap.add_argument("--overwrite", action="store_true", help="overwrite manifest.jsonl if exists")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    topn_path = _resolve_topn_path(run_dir, args.topN_csv)
    rows = _read_topn_rows(topn_path)

    if not rows:
        raise RuntimeError(f"topN.csv is empty: {topn_path}")

    # topN.csv is usually already sorted by score ascending (more negative is better).
    # Still, keep as-is to match user's expectation.
    topM = max(1, int(args.topM))
    picked = rows[:topM]

    out_dir = run_dir / "rescore" / f"iter_{int(args.iter_out):03d}"
    manifest_path = out_dir / "manifest.jsonl"
    best_path = out_dir / "best.json"

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"manifest.jsonl already exists: {manifest_path}. Use --overwrite to overwrite."
        )

    manifest_items: List[Dict[str, object]] = []
    for i, r in enumerate(picked, start=1):
        manifest_items.append(_row_to_manifest_item(r, iter_out=int(args.iter_out), rank=i))

    _write_jsonl(manifest_path, manifest_items)

    # best.json = best by dock_score
    best = min(manifest_items, key=lambda x: float(x.get("dock_score", 1e9)))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] manifest written: {manifest_path} (n={len(manifest_items)})")
    print(f"[OK] best dock_score={best['dock_score']:.3f} seq={best.get('seq','')}")
    print(f"[OK] source topN: {topn_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
