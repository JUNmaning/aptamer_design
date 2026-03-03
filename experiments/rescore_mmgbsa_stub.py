"""
Rescore stub for Top-M exports.

Reads:  <run_dir>/rescore/iter_XXX/manifest.jsonl
Writes: <run_dir>/rescore/iter_XXX/results.jsonl + results.csv + best.json

The "stub_score" is a deterministic transform of the docking score so that the plumbing
can be validated end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class Item:
    iter: int
    rank: int
    seq: str
    dock_score: float
    receptor_pdb: str
    pose_pdbqt: str
    dot_bracket: Optional[str] = None


def read_manifest(path: Path) -> List[Item]:
    items: List[Item] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                Item(
                    iter=int(d.get("iter", 0)),
                    rank=int(d.get("rank", 0)),
                    seq=str(d.get("seq", "")),
                    dock_score=float(d.get("dock_score", d.get("score", 1e9))),
                    receptor_pdb=str(d.get("receptor_pdb", "")),
                    pose_pdbqt=str(d.get("pose_pdbqt", "")),
                    dot_bracket=d.get("dot_bracket"),
                )
            )
    return items


def stub_mmgbsa_score(dock_score: float, offset: float, scale: float) -> float:
    # Deterministic placeholder; lower is better (consistent with docking)
    return (dock_score * scale) + offset


def score_one(item: Item, offset: float, scale: float) -> Dict[str, Any]:
    rec_ok = Path(item.receptor_pdb).exists()
    pose_ok = Path(item.pose_pdbqt).exists()
    s = stub_mmgbsa_score(item.dock_score, offset=offset, scale=scale)
    return {
        "iter": item.iter,
        "rank": item.rank,
        "seq": item.seq,
        "dock_score": item.dock_score,
        "stub_mmgbsa": s,
        "receptor_pdb": item.receptor_pdb,
        "pose_pdbqt": item.pose_pdbqt,
        "receptor_exists": rec_ok,
        "pose_exists": pose_ok,
        "dot_bracket": item.dot_bracket,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--iter", type=int, default=None, help="If omitted, rescore latest iter_*")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rescore_root = run_dir / "rescore"
    if not rescore_root.exists():
        print(f"[stub] rescore directory not found: {rescore_root}")
        return 2

    if args.iter is None:
        iters = sorted([p for p in rescore_root.glob("iter_*") if p.is_dir()])
        if not iters:
            print(f"[stub] no iter_* directories under {rescore_root}")
            return 2
        out_dir = iters[-1]
    else:
        out_dir = rescore_root / f"iter_{args.iter:03d}"
        if not out_dir.exists():
            print(f"[stub] iteration directory not found: {out_dir}")
            return 2

    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"[stub] manifest not found: {manifest}")
        return 2

    items = read_manifest(manifest)
    if not items:
        print(f"[stub] manifest empty: {manifest}")
        return 0

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, it, args.offset, args.scale) for it in items]
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda r: float(r.get("stub_mmgbsa", 1e9)))

    results_jsonl = out_dir / "results.jsonl"
    results_csv = out_dir / "results.csv"
    best_json = out_dir / "best.json"

    write_jsonl(results_jsonl, rows)
    write_csv(
        results_csv,
        rows,
        fieldnames=[
            "iter",
            "rank",
            "seq",
            "dock_score",
            "stub_mmgbsa",
            "receptor_exists",
            "pose_exists",
            "receptor_pdb",
            "pose_pdbqt",
        ],
    )
    best = rows[0]
    best_json.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[stub] wrote {len(rows)} results -> {results_jsonl}")
    print(f"[stub] best stub_mmgbsa={best['stub_mmgbsa']:.4f} seq={best.get('seq','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
