#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError("Command failed")


def compute_center(pdb):
    coords = []
    with open(pdb) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    coords = np.array(coords)
    return coords.mean(axis=0)


def prepare_receptor(receptor_pdb, out_dir):
    receptor_pdbqt = out_dir / "receptor.pdbqt"

    run_cmd([
        "prepare_receptor4.py",
        "-r", str(receptor_pdb),
        "-o", str(receptor_pdbqt),
        "-A", "checkhydrogens"
    ])

    return receptor_pdbqt


def prepare_ligand(ligand_sdf, out_dir):
    ligand_pdbqt = out_dir / "ligand.pdbqt"

    run_cmd([
        "obabel",
        str(ligand_sdf),
        "-O", str(ligand_pdbqt)
    ])

    return ligand_pdbqt


def vina_dock(receptor_pdb, ligand_sdf, out_dir, exhaustiveness):
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = prepare_receptor(receptor_pdb, out_dir)
    ligand_pdbqt = prepare_ligand(ligand_sdf, out_dir)

    center = compute_center(receptor_pdb)
    size = 30

    out_pdbqt = out_dir / "out.pdbqt"
    log_file = out_dir / "vina.log"

    run_cmd([
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size),
        "--size_y", str(size),
        "--size_z", str(size),
        "--exhaustiveness", str(exhaustiveness),
        "--out", str(out_pdbqt),
        "--log", str(log_file)
    ])

    affinity = None
    with open(log_file) as f:
        for line in f:
            if line.strip().startswith("1 "):
                affinity = float(line.split()[1])
                break

    return affinity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ligands", nargs="+", required=True)
    parser.add_argument("--exhaustiveness", type=int, default=8)

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    df = pd.read_csv(run_dir / "topN.csv")
    best = df.iloc[0]

    receptor_pdb = Path(best["best_receptor_pdb"])
    print("Using receptor:", receptor_pdb)

    out_root = run_dir / "redock_multi"
    out_root.mkdir(exist_ok=True)

    results = []

    for lig in args.ligands:
        lig = Path(lig).resolve()
        lig_name = lig.stem

        print("\n=== Docking", lig_name, "===")

        out_dir = out_root / lig_name

        affinity = vina_dock(
            receptor_pdb,
            lig,
            out_dir,
            args.exhaustiveness
        )

        print("Affinity:", affinity)

        results.append({
            "ligand": lig_name,
            "vina_affinity": affinity
        })

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values("vina_affinity")
    df_out.to_csv(out_root / "summary.csv", index=False)

    print("\n===== FINAL RANKING (by MMGBSA) =====")
    cols = [c for c in ["ligand_sdf", "pose_pdbqt", "score_vina", "mmgbsa_dg_bind"] if c in df_mm.columns]
    print(df_mm[cols])


if __name__ == "__main__":
    main()
