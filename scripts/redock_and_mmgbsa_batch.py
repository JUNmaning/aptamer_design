#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redock multiple ligands against the BEST GBSA aptamer
(from final_result/summary_top50.csv)
+ run MMGBSA automatically
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd

from dock.vina import dock_one


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError("Command failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ligands", nargs="+", required=True)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--igb", type=int, default=5)
    parser.add_argument("--saltcon", type=float, default=0.150)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    summary_file = run_dir / "final_result" / "summary_top50.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Not found: {summary_file}")

    df = pd.read_csv(summary_file)

    if "mmgbsa_dg_bind" not in df.columns:
        raise RuntimeError("summary_top50.csv missing mmgbsa_dg_bind column")

    best = df.sort_values("mmgbsa_dg_bind").iloc[0]

    receptor_pdb = Path(best["best_receptor_pdb"]).resolve()
    best_seq = best["seq"]

    print("Using GBSA-best receptor:", receptor_pdb)
    print("Best sequence:", best_seq)
    print("Best GBSA:", best["mmgbsa_dg_bind"])

    out_root = run_dir / "redock_mmgbsa_batch"
    out_root.mkdir(exist_ok=True)

    docking_rows = []

    # =========================
    #  Dock
    # =========================
    for idx, lig in enumerate(args.ligands, start=1):
        lig = Path(lig).resolve()
        lig_name = lig.stem

        print(f"\n=== Docking {lig_name} ===")

        workdir = out_root / lig_name

        res = dock_one(
            name=lig_name,
            receptor_pdb=receptor_pdb,
            ligand_sdf=lig,
            workdir=workdir,
            padding_angstrom=args.padding,
            exhaustiveness=args.exhaustiveness,
            cpu=args.cpu,
        )

        docking_rows.append({
            "seq": best_seq,
            "rank": idx,
            "best_frame": 1,
            "score_vina": res.affinity_kcal,
            "receptor_pdb": str(receptor_pdb),
            "pose_pdbqt": str(res.out_pdbqt),
            "ligand_sdf": str(lig),
        })

        print("Affinity:", res.affinity_kcal)

    dock_csv = out_root / "redock_topN.csv"
    pd.DataFrame(docking_rows).to_csv(dock_csv, index=False)

    print("\nDocking summary written to:", dock_csv)

    # =========================
    #  Run MMGBSA
    # =========================
    print("\n=== Running MMGBSA ===")

    cmd = [
        "python",
        "-m",
        "experiments.rescore_mmgbsa",
        "--run_dir", str(run_dir),
        "--topN_csv", str(dock_csv),
        "--iter_out", "999",
        "--workers", str(args.workers),
        "--igb", str(args.igb),
        "--saltcon", str(args.saltcon),
    ]

    if args.minimize:
        cmd += ["--minimize", "--min_restraint"]

    run_cmd(cmd)

    mmgbsa_csv = run_dir / "rescore" / "iter_999" / "mmgbsa_results.csv"
    if not mmgbsa_csv.exists():
        print("MMGBSA results not found.")
        return

    df_mm = pd.read_csv(mmgbsa_csv)
    df_mm = df_mm.sort_values("mmgbsa_dg_bind")

    final_csv = out_root / "final_ranking.csv"
    df_mm.to_csv(final_csv, index=False)

    print("\n===== FINAL RANKING =====")
    print(df_mm[["score_vina", "mmgbsa_dg_bind"]])
    print("\nSaved to:", final_csv)


if __name__ == "__main__":
    main()
