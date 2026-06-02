#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Redock multiple ligands against TOP-N GBSA aptamers
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
    parser.add_argument("--aptamer_topN", type=int, default=1,
                        help="Use top-N GBSA aptamers (default=1)")
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

    df = pd.read_csv(summary_file)
    df_sorted = df.sort_values("mmgbsa_dg_bind").reset_index(drop=True)

    topN = min(args.aptamer_topN, len(df_sorted))
    selected_aptamers = df_sorted.iloc[:topN]

    print(f"\nUsing top {topN} GBSA aptamers\n")

    all_results = []

    for apt_idx, apt in selected_aptamers.iterrows():
        receptor_pdb = Path(apt["best_receptor_pdb"]).resolve()
        apt_seq = apt["seq"]
        apt_rank = apt_idx + 1

        print(f"\n==============================")
        print(f"Aptamer rank {apt_rank}")
        print(f"GBSA = {apt['mmgbsa_dg_bind']}")
        print("==============================")

        docking_rows = []

        for lig in args.ligands:
            lig = Path(lig).resolve()
            lig_name = lig.stem

            print(f"\nDocking {lig_name} vs aptamer {apt_rank}")

            workdir = run_dir / "redock_mmgbsa_batch" / f"aptamer_{apt_rank}" / lig_name

            res = dock_one(
                name=f"{lig_name}_apt{apt_rank}",
                receptor_pdb=receptor_pdb,
                ligand_sdf=lig,
                workdir=workdir,
                padding_angstrom=args.padding,
                exhaustiveness=args.exhaustiveness,
                cpu=args.cpu,
            )

            docking_rows.append({
                "seq": apt_seq,
                "rank": apt_rank,
                "best_frame": 1,
                "score_vina": res.affinity_kcal,
                "receptor_pdb": str(receptor_pdb),
                "pose_pdbqt": str(res.out_pdbqt),
                "ligand_sdf": str(lig),
            })

        dock_csv = run_dir / "redock_mmgbsa_batch" / f"aptamer_{apt_rank}_dock.csv"
        pd.DataFrame(docking_rows).to_csv(dock_csv, index=False)

        cmd = [
            "python",
            "-m",
            "experiments.rescore_mmgbsa",
            "--run_dir", str(run_dir),
            "--topN_csv", str(dock_csv),
            "--iter_out", f"99{apt_rank}",
            "--workers", str(args.workers),
            "--igb", str(args.igb),
            "--saltcon", str(args.saltcon),
        ]

        if args.minimize:
            cmd += ["--minimize", "--min_restraint"]

        run_cmd(cmd)

        mmgbsa_csv = run_dir / "rescore" / f"iter_99{apt_rank}" / "mmgbsa_results.csv"
        df_mm = pd.read_csv(mmgbsa_csv)

        df_mm["aptamer_rank"] = apt_rank
        all_results.append(df_mm)

    final_df = pd.concat(all_results)
    final_df = final_df.sort_values("mmgbsa_dg_bind")

    out_file = run_dir / "redock_mmgbsa_batch" / "final_matrix.csv"
    final_df.to_csv(out_file, index=False)

    print("\n===== FINAL MATRIX (sorted by ΔG) =====")
    print(final_df[["aptamer_rank", "ligand_sdf", "mmgbsa_dg_bind"]])
    print("\nSaved to:", out_file)


if __name__ == "__main__":
    main()
