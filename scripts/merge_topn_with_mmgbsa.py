#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd


MMGBSA_CANDIDATE_COLS = [
    "mmgbsa_dg_bind",
    "dg_bind",
    "stub_mmgbsa",
    "mmgbsa_score",
]


def find_mmgbsa_col(df: pd.DataFrame) -> Optional[str]:
    for c in MMGBSA_CANDIDATE_COLS:
        if c in df.columns:
            return c
    return None


def read_topn(topn_csv: Path) -> pd.DataFrame:
    if not topn_csv.exists():
        raise FileNotFoundError(f"topN.csv not found: {topn_csv}")
    df = pd.read_csv(topn_csv)
    if "seq" not in df.columns:
        raise ValueError(f"topN.csv is missing required column: seq ({topn_csv})")
    return df


def _normalize_ok_value(x) -> Optional[bool]:
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def collect_mmgbsa_all(run_dir: Path, ok_only: bool = False) -> pd.DataFrame:
    pattern = str(run_dir / "rescore" / "iter_*" / "mmgbsa_results.csv")
    files = sorted(glob.glob(pattern))

    rows: List[Dict] = []

    for fp in files:
        p = Path(fp)
        try:
            iter_idx = int(p.parent.name.split("_")[-1])
        except Exception:
            iter_idx = -1

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        if df.empty or "seq" not in df.columns:
            continue

        dg_col = find_mmgbsa_col(df)
        if dg_col is None:
            continue

        df = df.copy()
        df["_mmgbsa"] = pd.to_numeric(df[dg_col], errors="coerce")
        df = df.dropna(subset=["_mmgbsa"])
        df = df[(df["_mmgbsa"] > -500) & (df["_mmgbsa"] < 100)]
        if df.empty:
            continue

        if "rank" in df.columns:
            df["_rank"] = pd.to_numeric(df["rank"], errors="coerce")
        else:
            df["_rank"] = np.arange(1, len(df) + 1)

        if "mmgbsa_ok" in df.columns:
            df["_ok_raw"] = df["mmgbsa_ok"]
            df["_ok"] = df["mmgbsa_ok"].map(_normalize_ok_value)
        else:
            df["_ok_raw"] = ""
            df["_ok"] = None

        if ok_only:
            df = df[df["_ok"] == True]
            if df.empty:
                continue

        dock_col = None
        for c in ["dock_score", "score", "score_vina", "vina_score"]:
            if c in df.columns:
                dock_col = c
                break

        receptor_col = "receptor_pdb" if "receptor_pdb" in df.columns else None
        pose_col = "pose_pdbqt" if "pose_pdbqt" in df.columns else None

        for _, r in df.iterrows():
            rows.append({
                "seq": r["seq"],
                "mmgbsa_dg_bind": float(r["_mmgbsa"]),
                "mmgbsa_iter": iter_idx,
                "mmgbsa_rank": int(r["_rank"]) if pd.notna(r["_rank"]) else np.nan,
                "mmgbsa_ok": r["_ok_raw"],
                "mmgbsa_ok_bool": r["_ok"],
                "dock_score_from_mmgbsa": r[dock_col] if dock_col else np.nan,
                "receptor_pdb_from_mmgbsa": r[receptor_col] if receptor_col else "",
                "pose_pdbqt_from_mmgbsa": r[pose_col] if pose_col else "",
                "mmgbsa_source_file": str(p),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "seq",
            "mmgbsa_dg_bind",
            "mmgbsa_iter",
            "mmgbsa_rank",
            "mmgbsa_ok",
            "mmgbsa_ok_bool",
            "dock_score_from_mmgbsa",
            "receptor_pdb_from_mmgbsa",
            "pose_pdbqt_from_mmgbsa",
            "mmgbsa_source_file",
        ])

    return pd.DataFrame(rows)


def build_mmgbsa_summary(mm_all: pd.DataFrame) -> pd.DataFrame:
    if mm_all.empty:
        return pd.DataFrame(columns=[
            "seq",
            "best_mmgbsa_dg_bind",
            "best_mmgbsa_iter",
            "best_mmgbsa_rank",
            "best_mmgbsa_ok",
            "best_mmgbsa_source_file",
            "best_mmgbsa_dock_score",
            "best_mmgbsa_receptor_pdb",
            "best_mmgbsa_pose_pdbqt",
            "latest_mmgbsa_dg_bind",
            "latest_mmgbsa_iter",
            "latest_mmgbsa_rank",
            "latest_mmgbsa_ok",
            "latest_mmgbsa_source_file",
            "latest_mmgbsa_dock_score",
            "latest_mmgbsa_receptor_pdb",
            "latest_mmgbsa_pose_pdbqt",
            "mmgbsa_n_records",
        ])

    best_df = (
        mm_all.sort_values(
            by=["seq", "mmgbsa_dg_bind", "mmgbsa_iter", "mmgbsa_rank"],
            ascending=[True, True, True, True]
        )
        .groupby("seq", as_index=False)
        .first()
        .rename(columns={
            "mmgbsa_dg_bind": "best_mmgbsa_dg_bind",
            "mmgbsa_iter": "best_mmgbsa_iter",
            "mmgbsa_rank": "best_mmgbsa_rank",
            "mmgbsa_ok": "best_mmgbsa_ok",
            "mmgbsa_source_file": "best_mmgbsa_source_file",
            "dock_score_from_mmgbsa": "best_mmgbsa_dock_score",
            "receptor_pdb_from_mmgbsa": "best_mmgbsa_receptor_pdb",
            "pose_pdbqt_from_mmgbsa": "best_mmgbsa_pose_pdbqt",
        })
    )

    latest_df = (
        mm_all.sort_values(
            by=["seq", "mmgbsa_iter", "mmgbsa_rank"],
            ascending=[True, False, True]
        )
        .groupby("seq", as_index=False)
        .first()
        .rename(columns={
            "mmgbsa_dg_bind": "latest_mmgbsa_dg_bind",
            "mmgbsa_iter": "latest_mmgbsa_iter",
            "mmgbsa_rank": "latest_mmgbsa_rank",
            "mmgbsa_ok": "latest_mmgbsa_ok",
            "mmgbsa_source_file": "latest_mmgbsa_source_file",
            "dock_score_from_mmgbsa": "latest_mmgbsa_dock_score",
            "receptor_pdb_from_mmgbsa": "latest_mmgbsa_receptor_pdb",
            "pose_pdbqt_from_mmgbsa": "latest_mmgbsa_pose_pdbqt",
        })
    )

    count_df = (
        mm_all.groupby("seq", as_index=False)
        .size()
        .rename(columns={"size": "mmgbsa_n_records"})
    )

    out = best_df.merge(latest_df, on="seq", how="outer")
    out = out.merge(count_df, on="seq", how="outer")
    return out


def merge_topn_with_mmgbsa(
    run_dir: Path,
    out_csv: Optional[Path] = None,
    out_sorted_csv: Optional[Path] = None,
    ok_only: bool = False,
) -> Tuple[Path, Path]:
    topn_csv = run_dir / "topN.csv"
    topn = read_topn(topn_csv)

    mm_all = collect_mmgbsa_all(run_dir, ok_only=ok_only)
    mm_sum = build_mmgbsa_summary(mm_all)

    merged = topn.merge(mm_sum, on="seq", how="left")

    numeric_cols = [
        "score",
        "best_mmgbsa_dg_bind",
        "latest_mmgbsa_dg_bind",
        "best_mmgbsa_dock_score",
        "latest_mmgbsa_dock_score",
    ]
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if "best_mmgbsa_dg_bind" in merged.columns:
        merged["has_mmgbsa"] = merged["best_mmgbsa_dg_bind"].notna()

    if out_csv is None:
        out_csv = run_dir / "topN_with_mmgbsa.csv"
    if out_sorted_csv is None:
        out_sorted_csv = run_dir / "topN_with_mmgbsa_sorted_by_gbsa.csv"

    merged.to_csv(out_csv, index=False)

    sorted_df = merged.copy()
    if "best_mmgbsa_dg_bind" in sorted_df.columns:
        sorted_df["_has_mmgbsa_sort"] = sorted_df["best_mmgbsa_dg_bind"].notna().astype(int)
        if "score" in sorted_df.columns:
            sorted_df = sorted_df.sort_values(
                by=["_has_mmgbsa_sort", "best_mmgbsa_dg_bind", "score"],
                ascending=[False, True, True]
            )
        else:
            sorted_df = sorted_df.sort_values(
                by=["_has_mmgbsa_sort", "best_mmgbsa_dg_bind"],
                ascending=[False, True]
            )
        sorted_df = sorted_df.drop(columns=["_has_mmgbsa_sort"])
    elif "score" in sorted_df.columns:
        sorted_df = sorted_df.sort_values(by="score", ascending=True)

    sorted_df.to_csv(out_sorted_csv, index=False)

    return out_csv, out_sorted_csv


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge topN.csv with MMGBSA results collected from rescore/iter_*/mmgbsa_results.csv"
    )
    ap.add_argument("--run_dir", required=True, type=str, help="Path to run directory")
    ap.add_argument("--out_csv", default=None, type=str, help="Output merged CSV path")
    ap.add_argument("--out_sorted_csv", default=None, type=str, help="Output GBSA-sorted CSV path")
    ap.add_argument("--ok_only", action="store_true", help="Use only rows with mmgbsa_ok=True/1/yes")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else None
    out_sorted_csv = Path(args.out_sorted_csv).expanduser().resolve() if args.out_sorted_csv else None

    out1, out2 = merge_topn_with_mmgbsa(
        run_dir=run_dir,
        out_csv=out_csv,
        out_sorted_csv=out_sorted_csv,
        ok_only=args.ok_only,
    )

    print(f"[OK] Wrote merged file: {out1}")
    print(f"[OK] Wrote GBSA-sorted file: {out2}")


if __name__ == "__main__":
    main()
