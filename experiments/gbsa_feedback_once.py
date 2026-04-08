#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GBSA feedback for CE optimization.

This module reads MM/GBSA results and updates the CE distribution.
Key change: By default, only uses entries with mmgbsa_ok=True for elite selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

# CE implementation
from optimizers.ce import CEConfig, SequenceDistribution


@dataclass
class GbsaRow:
    """Single result row with score and validation status."""
    seq: str
    score: float
    ok: bool
    source: str


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    """Iterate over CSV rows as dictionaries."""
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            yield r


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, object]]:
    """Iterate over JSONL rows as dictionaries."""
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _as_bool(v: object, default: bool = True) -> bool:
    """Convert value to boolean with fallback."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "ok"):
        return True
    if s in ("false", "0", "no", "n", "bad", "fail", "failed"):
        return False
    return default


def _as_float(v: object) -> Optional[float]:
    """Convert value to float, return None if invalid or penalty value."""
    try:
        x = float(v)
        if not np.isfinite(x):
            return None
        # Treat very large values (penalty) as invalid
        if x >= 1e8:
            return None
        return float(x)
    except Exception:
        return None


def _pick_score(row: Dict[str, object]) -> Tuple[Optional[str], Optional[float], bool]:
    """
    Extract score and validation status from row.
    
    Compatible with multiple score columns:
      1) mmgbsa_dg_bind (+ mmgbsa_ok) - real MM/GBSA results
      2) stub_mmgbsa - placeholder scores
      3) dock_score / score - fallback to docking
    
    For mmgbsa_dg_bind, default ok=False (must be explicitly True).
    This ensures failed MM/GBSA calculations are not used.
    
    Returns:
        (column_name, score, is_ok)
    """
    # Real MM/GBSA results - default ok=False to filter failed calculations
    if "mmgbsa_dg_bind" in row:
        sc = _as_float(row.get("mmgbsa_dg_bind"))
        # IMPORTANT: Default to False for MM/GBSA to exclude failed calculations
        ok = _as_bool(row.get("mmgbsa_ok"), default=False)
        return "mmgbsa_dg_bind", sc, ok

    # Stub scores (from build_results_csv before real MM/GBSA)
    if "stub_mmgbsa" in row:
        sc = _as_float(row.get("stub_mmgbsa"))
        # For stub, check mmgbsa_ok if present, otherwise assume ok
        ok = _as_bool(row.get("mmgbsa_ok"), default=True)
        return "stub_mmgbsa", sc, ok

    # Fallback to docking score
    if "dock_score" in row:
        sc = _as_float(row.get("dock_score"))
        ok = _as_bool(row.get("ok"), default=True)
        return "dock_score", sc, ok

    if "score" in row:
        sc = _as_float(row.get("score"))
        ok = _as_bool(row.get("ok"), default=True)
        return "score", sc, ok

    return None, None, True


def load_results(results_path: Path) -> List[GbsaRow]:
    """
    Load results from CSV or JSONL file.
    
    Args:
        results_path: Path to results file (CSV or JSONL).
    
    Returns:
        List of GbsaRow with parsed scores and validation status.
    """
    if not results_path.exists():
        return []

    rows: List[GbsaRow] = []

    if results_path.suffix.lower() == ".csv":
        it = _iter_csv_rows(results_path)
    else:
        it = _iter_jsonl_rows(results_path)

    for r in it:
        rr: Dict[str, object] = dict(r)

        # Extract sequence (try multiple column names)
        seq = ""
        for key in ["seq", "sequence", "SEQ", "Sequence"]:
            if key in rr and rr[key]:
                seq = str(rr[key]).strip().upper()
                break
        
        # Normalize: T -> U for RNA
        seq = seq.replace("T", "U")
        if not seq:
            continue

        col, sc, ok = _pick_score(rr)
        if col is None or sc is None:
            continue

        rows.append(GbsaRow(seq=seq, score=float(sc), ok=bool(ok), source=col))

    return rows


def choose_elites(
    rows: List[GbsaRow],
    elite_n: int,
    require_ok: bool,
    minimize: bool,
) -> List[GbsaRow]:
    """
    Select elite sequences from results.
    
    Args:
        rows: List of parsed result rows.
        elite_n: Number of elites to select.
        require_ok: If True, only consider rows with ok=True.
        minimize: If True, lower scores are better (default for binding energy).
    
    Returns:
        List of selected elite rows.
    """
    # Filter by ok status
    if require_ok:
        cand = [x for x in rows if x.ok]
    else:
        cand = list(rows)
    
    if not cand:
        return []
    
    # Sort by score
    cand.sort(key=lambda x: x.score, reverse=(not minimize))
    
    # Select top elite_n
    elite_n = max(1, min(elite_n, len(cand)))
    return cand[:elite_n]


def _load_or_init_dist(
    dist_path: Path,
    seq_len: int,
    batch: int,
    seed: int,
    smoothing_alpha: float,
    min_prob: float,
    temperature: float,
) -> SequenceDistribution:
    """Load existing distribution or initialize a new one."""
    if dist_path.exists():
        return SequenceDistribution.from_file(dist_path)

    cfg = CEConfig(
        length=seq_len,
        alphabet="RNA",
        batch_size=batch,
        elite_frac=0.1,
        smoothing_alpha=smoothing_alpha,
        temperature=temperature,
        min_prob=min_prob,
        seed=seed,
    )
    return SequenceDistribution(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Update CE distribution from MM/GBSA results"
    )
    ap.add_argument("--run_dir", required=True, help="Run directory, e.g. runs/ce_run1")
    ap.add_argument("--iter", type=int, required=True, help="Iteration index, e.g. 1 -> iter_001")
    ap.add_argument("--elite_n", type=int, default=3, help="Number of elites to select")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for next iteration")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    # CHANGED: Default to True - only use successful MM/GBSA results
    ap.add_argument("--require_ok", action="store_true", default=True,
                    help="Only use rows with mmgbsa_ok=True (default: True)")
    ap.add_argument("--no_require_ok", action="store_true",
                    help="Disable require_ok filter (use all results)")
    ap.add_argument("--minimize", action="store_true", default=True,
                    help="Smaller score is better (default: True)")
    ap.add_argument("--alpha", type=float, default=0.4, help="CE smoothing_alpha")
    ap.add_argument("--min_prob", type=float, default=0.02, help="Minimum probability floor")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--results_path", default="", help="Override results file path (csv/jsonl)")
    args = ap.parse_args()

    # Handle require_ok flag
    require_ok = args.require_ok and not args.no_require_ok

    run_dir = Path(args.run_dir)
    iter_dir = run_dir / "rescore" / f"iter_{args.iter:03d}"
    dist_path = run_dir / "dist.json"

    # Determine results path (priority: arg > results.csv > results.jsonl > mmgbsa_results.csv)
    if args.results_path:
        results_path = Path(args.results_path)
    else:
        # Try multiple filenames in order
        candidates = [
            iter_dir / "results.csv",
            iter_dir / "results.jsonl",
            iter_dir / "mmgbsa_results.csv",
            iter_dir / "mmgbsa_results.jsonl",
        ]
        results_path = None
        for cand in candidates:
            if cand.exists():
                results_path = cand
                break
        
        if results_path is None:
            raise RuntimeError(
                f"[ERR] No results file found in {iter_dir}.\n"
                f"      Searched: {[c.name for c in candidates]}"
            )

    # Load and parse results
    rows = load_results(results_path)
    if not rows:
        raise RuntimeError(
            f"[ERR] No usable rows parsed from {results_path}.\n"
            f"      Check file exists and has score columns: mmgbsa_dg_bind / stub_mmgbsa / dock_score.\n"
            f"      You can run: head -n 3 {results_path}"
        )

    # Report statistics before filtering
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.ok)
    n_failed = n_total - n_ok
    
    print(f"[INFO] Loaded {n_total} rows from {results_path}")
    print(f"[INFO] Status: {n_ok} ok, {n_failed} failed")
    
    if require_ok and n_ok == 0:
        raise RuntimeError(
            f"[ERR] All {n_total} rows have ok=False (MM/GBSA failed).\n"
            f"      Check MM/GBSA logs in {iter_dir / 'mmgbsa'}/*/mmpbsa.log\n"
            f"      Common causes: wrong ligand charge, missing parameters.\n"
            f"      To proceed anyway, use --no_require_ok"
        )

    # Select elites
    elites = choose_elites(
        rows,
        elite_n=args.elite_n,
        require_ok=require_ok,
        minimize=args.minimize,
    )
    
    if not elites:
        cols_hint = sorted({r.source for r in rows})
        raise RuntimeError(
            f"[ERR] No elites selected after filtering (require_ok={require_ok}).\n"
            f"      Parsed rows={n_total}, ok rows={n_ok}, score sources={cols_hint}.\n"
            f"      Suggest: check MM/GBSA logs or use --no_require_ok"
        )

    # Write elites file
    elites_path = iter_dir / "elites_for_ce.txt"
    elites_path.parent.mkdir(parents=True, exist_ok=True)
    elites_path.write_text("".join([e.seq + "\n" for e in elites]))

    # Update the distribution and sample next batch
    seq_len = len(elites[0].seq)
    dist = _load_or_init_dist(
        dist_path=dist_path,
        seq_len=seq_len,
        batch=args.batch,
        seed=args.seed,
        smoothing_alpha=args.alpha,
        min_prob=args.min_prob,
        temperature=args.temperature,
    )

    dist.update_from_elites([e.seq for e in elites])
    dist.save(dist_path)

    rng = np.random.default_rng(args.seed)
    next_seqs = dist.sample(args.batch, rng)

    next_batch_path = run_dir / "next_batch.txt"
    next_batch_path.write_text("".join([s + "\n" for s in next_seqs]))

    # Print summary
    best = elites[0]
    print(f"[OK] Results:      {results_path}")
    print(f"[OK] Filter:       require_ok={require_ok}, selected {len(elites)}/{n_total} as elites")
    print(f"[OK] Best elite:   score={best.score:.4f} source={best.source} seq={best.seq[:20]}...")
    print(f"[OK] Elites file:  {elites_path}")
    print(f"[OK] Dist updated: {dist_path}")
    print(f"[OK] Next batch:   {next_batch_path} (n={len(next_seqs)})")


if __name__ == "__main__":
    main()
