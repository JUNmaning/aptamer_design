from __future__ import annotations

from pathlib import Path
import csv

from structure.simrna import fold_sequence_to_pdb_with_vienna
from dock.vina import dock_one


def _fold_compat(seq: str, workdir: Path, simrna_iter: int, frames: str, omp: int):
    """
    Compatible wrapper: supports frames_to_export, frame_to_export, or simrna_frames.
    """
    try:
        return fold_sequence_to_pdb_with_vienna(
            seq=seq,
            workdir=workdir,
            out_prefix="aptamer",
            omp_threads=omp,
            n_iterations=simrna_iter,
            frames_to_export=frames,
        )
    except TypeError:
        try:
            return fold_sequence_to_pdb_with_vienna(
                seq=seq,
                workdir=workdir,
                out_prefix="aptamer",
                omp_threads=omp,
                n_iterations=simrna_iter,
                frame_to_export=frames,
            )
        except TypeError:
            return fold_sequence_to_pdb_with_vienna(
                seq=seq,
                workdir=workdir,
                out_prefix="aptamer",
                omp_threads=omp,
                n_iterations=simrna_iter,
                simrna_frames=frames,
            )


def main():
    run_dir = Path("runs/dock_rank_batch").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    ligand = Path("data/ligand.sdf")
    if not ligand.exists():
        ligand = Path("data/ligand.pdb")
    if not ligand.exists():
        raise FileNotFoundError("Put ligand at data/ligand.sdf or data/ligand.pdb")

    seqs = [
        "GAAAAUUCUUGAUAGGUAUGAACGAACGUG",
        "GGCAAUCUCAGAUAGGUAUCCACCAACGUG",
    ]

    out_csv = run_dir / "dock_scores.csv"
    rows = []

    for i, seq in enumerate(seqs, start=1):
        tag = f"seq{i:03d}"
        fold_dir = run_dir / "fold" / tag
        dock_dir = run_dir / "dock" / tag

        res2d, res3d = _fold_compat(seq, fold_dir, simrna_iter=2000, frames="1", omp=1)
        receptor_pdb = Path(res3d.all_atom_pdb)

        dres = dock_one(
            name=tag,
            receptor_pdb=receptor_pdb,
            ligand_sdf=ligand,
            workdir=dock_dir,
            padding_angstrom=8.0,
            exhaustiveness=2,
            cpu=1,
            seed=123,
            num_modes=5,
        )

        rows.append(
            {"seq": seq, "score": dres.affinity_kcal, "dock_dir": str(dock_dir), "pose": str(dres.out_pdbqt)}
        )
        print(tag, seq[:12], "score=", dres.affinity_kcal)

    rows.sort(key=lambda r: float(r["score"]))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "score", "dock_dir", "pose"])
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
