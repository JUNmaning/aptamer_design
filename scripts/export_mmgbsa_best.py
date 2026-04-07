#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def normalize_ok_value(x) -> Optional[bool]:
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def collect_all_mmgbsa_records(run_dir: Path, ok_only: bool = False) -> pd.DataFrame:
    pattern = str(run_dir / "rescore" / "iter_*" / "mmgbsa_results.csv")
    files = sorted(glob.glob(pattern))

    rows: List[Dict] = []

    for fp in files:
        p = Path(fp)

        try:
            iter_idx = int(p.parent.name.split("_")[-1])
        except Exception:
            iter_idx = np.nan

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        required_cols = {"seq", "rank", "score_vina", "mmgbsa_dg_bind"}
        if df.empty or not required_cols.issubset(df.columns):
            continue

        df = df.copy()
        df["seq"] = df["seq"].astype(str)
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["score_vina"] = pd.to_numeric(df["score_vina"], errors="coerce")
        df["mmgbsa_dg_bind"] = pd.to_numeric(df["mmgbsa_dg_bind"], errors="coerce")

        df = df.dropna(subset=["seq", "mmgbsa_dg_bind"])
        df = df[(df["mmgbsa_dg_bind"] > -500) & (df["mmgbsa_dg_bind"] < 100)]
        if df.empty:
            continue

        if "mmgbsa_ok" in df.columns:
            df["mmgbsa_ok_bool"] = df["mmgbsa_ok"].map(normalize_ok_value)
        else:
            df["mmgbsa_ok_bool"] = None
            df["mmgbsa_ok"] = ""

        if ok_only:
            df = df[df["mmgbsa_ok_bool"] == True]
            if df.empty:
                continue

        for _, r in df.iterrows():
            rows.append({
                "seq": r["seq"],
                "best_mmgbsa_dg_bind": float(r["mmgbsa_dg_bind"]),
                "dock_score": float(r["score_vina"]) if pd.notna(r["score_vina"]) else np.nan,
                "mmgbsa_delta": (
                    float(r["score_vina"]) - float(r["mmgbsa_dg_bind"])
                    if pd.notna(r["score_vina"]) and pd.notna(r["mmgbsa_dg_bind"])
                    else np.nan
                ),
                "mmgbsa_iter": iter_idx,
                "mmgbsa_rank": int(r["rank"]) if pd.notna(r["rank"]) else np.nan,
                "best_frame": r.get("best_frame", np.nan),
                "mmgbsa_ok": r.get("mmgbsa_ok", ""),
                "mmgbsa_charge": r.get("mmgbsa_charge", ""),
                "mmgbsa_charge_method": r.get("mmgbsa_charge_method", ""),
                "mmgbsa_igb": r.get("mmgbsa_igb", ""),
                "mmgbsa_saltcon": r.get("mmgbsa_saltcon", ""),
                "mmgbsa_ph": r.get("mmgbsa_ph", ""),
                "receptor_pdb": r.get("receptor_pdb", ""),
                "pose_pdbqt": r.get("pose_pdbqt", ""),
                "mmgbsa_workdir": r.get("mmgbsa_workdir", ""),
                "mmgbsa_final": r.get("mmgbsa_final", ""),
                "mmgbsa_log": r.get("mmgbsa_log", ""),
                "mmgbsa_error": r.get("mmgbsa_error", ""),
                "mmgbsa_file": str(p),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "seq",
            "best_mmgbsa_dg_bind",
            "dock_score",
            "mmgbsa_delta",
            "mmgbsa_iter",
            "mmgbsa_rank",
            "best_frame",
            "mmgbsa_ok",
            "mmgbsa_charge",
            "mmgbsa_charge_method",
            "mmgbsa_igb",
            "mmgbsa_saltcon",
            "mmgbsa_ph",
            "receptor_pdb",
            "pose_pdbqt",
            "mmgbsa_workdir",
            "mmgbsa_final",
            "mmgbsa_log",
            "mmgbsa_error",
            "mmgbsa_file",
        ])

    return pd.DataFrame(rows)


def build_top_mmgbsa_table(all_df: pd.DataFrame, topk: int = 50) -> pd.DataFrame:
    if all_df.empty:
        return all_df.copy()

    best_df = (
        all_df.sort_values(
            by=["seq", "best_mmgbsa_dg_bind", "mmgbsa_iter", "mmgbsa_rank"],
            ascending=[True, True, True, True]
        )
        .groupby("seq", as_index=False)
        .first()
    )

    best_df = best_df.sort_values(
        by=["best_mmgbsa_dg_bind", "dock_score"],
        ascending=[True, True]
    ).reset_index(drop=True)

    best_df.insert(0, "global_mmgbsa_rank", np.arange(1, len(best_df) + 1))

    return best_df.head(int(topk))


def export_top50_mmgbsa_best(
    run_dir: Path,
    out_csv: Optional[Path] = None,
    topk: int = 50,
    ok_only: bool = False,
) -> Path:
    all_df = collect_all_mmgbsa_records(run_dir, ok_only=ok_only)
    top_df = build_top_mmgbsa_table(all_df, topk=topk)

    if out_csv is None:
        out_csv = run_dir / f"top{int(topk)}_global_mmgbsa_best.csv"

    top_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export top-K sequences ranked by best MMGBSA across all rescore iterations"
    )
    ap.add_argument("--run_dir", required=True, type=str, help="Path to run directory")
    ap.add_argument("--topk", default=50, type=int, help="Number of top sequences to export")
    ap.add_argument("--out_csv", default=None, type=str, help="Output CSV path")
    ap.add_argument("--ok_only", action="store_true", help="Use only MMGBSA rows with mmgbsa_ok=True/1/yes")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else None

    out = export_top50_mmgbsa_best(
        run_dir=run_dir,
        out_csv=out_csv,
        topk=args.topk,
        ok_only=args.ok_only,
    )

    print(f"[OK] Wrote top MMGBSA table: {out}")


if __name__ == "__main__":
    main()
