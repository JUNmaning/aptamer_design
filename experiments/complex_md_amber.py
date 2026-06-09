from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import os
import time
from pathlib import Path

def run_cmd(cmd, cwd, input_text=None, env=None):
    print(f"[CMD] {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        res = subprocess.run(
            cmd, cwd=cwd, input=input_text, 
            text=True, capture_output=True, check=True,
            env=run_env
        )
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise e

def sanitize_rna_pdb(in_pdb: Path, out_pdb: Path):
    phosphate_atoms = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}
    lines = in_pdb.read_text().splitlines(True)
    residues = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")): continue
        key = (line[21], line[22:26])
        if not residues or residues[-1] != key: residues.append(key)
    first_res = residues[0] if residues else None
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            atom = line[12:16].strip()
            key = (line[21], line[22:26])
            if first_res and key == first_res and atom in phosphate_atoms: continue
        out.append(line)
    out_pdb.write_text("".join(out))

def build_ligand_files(workdir: Path, pose_pdbqt: Path):
    run_cmd(["obabel", str(pose_pdbqt), "-O", "docked_ligand.sdf", "-f", "1", "-l", "1"], cwd=workdir)
    run_cmd(["antechamber", "-i", "docked_ligand.sdf", "-fi", "sdf", "-o", "ligand.mol2", "-fo", "mol2", "-c", "bcc", "-s", "2", "-rn", "LIG"], cwd=workdir)
    run_cmd(["parmchk2", "-i", "ligand.mol2", "-f", "mol2", "-o", "ligand.frcmod"], cwd=workdir)
    run_cmd(["antechamber", "-i", "ligand.mol2", "-fi", "mol2", "-o", "ligand_fixed.pdb", "-fo", "pdb"], cwd=workdir)

def prepare_complex(run_dir: Path, rec: dict, workdir: Path):
    receptor_path = Path(rec["best_receptor_pdb"])
    if not receptor_path.is_absolute(): receptor_path = (run_dir / receptor_path).resolve()
    
    ligand_fixed_pdb = workdir / "ligand_fixed.pdb"
    complex_raw = workdir / "complex_raw.pdb"
    
    with complex_raw.open("w") as out:
        for line in receptor_path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM", "TER")):
                out.write(line + "\n")
        
        out.write("TER\n")

        for line in ligand_fixed_pdb.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                out.write(line + "\n")

        out.write("END\n")
    
    complex_sanitized = workdir / "complex_sanitized.pdb"
    sanitize_rna_pdb(complex_raw, complex_sanitized)
    return complex_sanitized

def build_amber_system(workdir: Path):

    tleap_script = """
source leaprc.RNA.OL3
source leaprc.gaff2
LIG = loadmol2 ligand.mol2
loadamberparams ligand.frcmod
complex = loadpdb complex_sanitized.pdb
saveamberparm complex complex.prmtop complex.inpcrd
quit
"""
    (workdir / "tleap.in").write_text(tleap_script)
    run_cmd(["tleap", "-f", "tleap.in"], cwd=workdir)

def convert_to_gromacs(workdir: Path):

    import parmed as pmd
    amber = pmd.load_file(str(workdir / "complex.prmtop"), str(workdir / "complex.inpcrd"))
    top_path = workdir / "complex.top"
    amber.save(str(top_path), overwrite=True)
    amber.save(str(workdir / "complex.gro"), overwrite=True)

    lines = top_path.read_text().splitlines()
    existing_types = set()
    in_atomtypes = False
    for line in lines:
        if line.startswith("[ atomtypes ]"): in_atomtypes = True
        elif line.startswith("["): in_atomtypes = False
        if in_atomtypes and len(line.split()) > 1 and not line.startswith(";"):
            existing_types.add(line.split()[0])

    needed_atomtypes = {
        "OW": "OW      8      15.99940     0.0000  A   0.315061    0.636386",
        "HW": "HW      1       1.00800     0.0000  A   0.000000    0.000000",
        "NA": "NA      11     22.98977     0.0000  A   0.332840    0.011590",
        "CL": "CL      17     35.45300     0.0000  A   0.440104    0.418400"
    }

    mol_definitions = """
[ moleculetype ]
; Name            nrexcl
SOL               3
[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass
     1         OW      1    SOL     OW      1    -0.8340   15.99940
     2         HW      1    SOL    HW1      2     0.4170    1.00800
     3         HW      1    SOL    HW2      3     0.4170    1.00800
[ settles ]
; OW    funct   doh     dhh
1       1       0.09572 0.15139
[ exclusions ]
1       2       3
2       1       3
3       1       2
[ moleculetype ]
; Name            nrexcl
NA               1
[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass
     1         NA      1     NA     NA      1     1.0000   22.98977
[ moleculetype ]
; Name            nrexcl
CL               1
[ atoms ]
;   nr       type  resnr residue  atom   cgnr     charge       mass
     1         CL      1     CL     CL      1    -1.0000   35.45300
"""
    new_lines = []
    for line in lines:
        if line.startswith("[ system ]"): new_lines.append(mol_definitions)
        new_lines.append(line)
        if line.startswith("[ atomtypes ]"):
            for t_name, t_def in needed_atomtypes.items():
                if t_name not in existing_types:
                    new_lines.append(t_def)
                    existing_types.add(t_name)
    top_path.write_text("\n".join(new_lines) + "\n")

def run_md_steps(workdir: Path, gmx: str, mdp_dir: Path, threads: int, gpu: bool):
    """GROMACS"""
    md_env = {"OMP_NUM_THREADS": str(threads)}
    mdrun_base = [gmx, "mdrun", "-ntomp", str(threads), "-ntmpi", "1"]

    run_cmd([gmx, "editconf", "-f", "complex.gro", "-o", "boxed.gro", "-c", "-d", "1.0", "-bt", "cubic"], cwd=workdir)
    run_cmd([gmx, "solvate", "-cp", "boxed.gro", "-cs", "spc216.gro", "-o", "solv.gro", "-p", "complex.top"], cwd=workdir)
    shutil.copy2(mdp_dir / "ions.mdp", workdir / "ions.mdp")
    run_cmd([gmx, "grompp", "-f", "ions.mdp", "-c", "solv.gro", "-p", "complex.top", "-o", "ions.tpr", "-maxwarn", "2"], cwd=workdir)
    run_cmd([gmx, "genion", "-s", "ions.tpr", "-o", "solv_ions.gro", "-p", "complex.top", "-pname", "NA", "-nname", "CL", "-neutral"], 
            cwd=workdir, input_text="SOL\n")

    shutil.copy2(mdp_dir / "em.mdp", workdir / "em.mdp")
    run_cmd([gmx, "grompp", "-f", "em.mdp", "-c", "solv_ions.gro", "-p", "complex.top", "-o", "em.tpr", "-maxwarn", "2"], cwd=workdir)
    em_cmd = list(mdrun_base)
    if gpu: em_cmd += ["-nb", "gpu"]
    run_cmd(em_cmd + ["-deffnm", "em"], cwd=workdir, env=md_env)

    dyn_cmd = list(mdrun_base)
    if gpu: dyn_cmd += ["-nb", "gpu", "-pme", "gpu"]

    for step in ["nvt", "npt", "md"]:
        shutil.copy2(mdp_dir / f"{step}.mdp", workdir / f"{step}.mdp")
        prev = "em" if step == "nvt" else ("nvt" if step == "npt" else "npt")
        args = [gmx, "grompp", "-f", f"{step}.mdp", "-c", f"{prev}.gro", "-p", "complex.top", "-o", f"{step}.tpr", "-maxwarn", "2"]
        if step != "nvt": args += ["-t", f"{prev}.cpt"]
        if step in ["nvt", "npt"]: args += ["-r", f"{prev}.gro"]
        run_cmd(args, cwd=workdir)
        run_cmd(dyn_cmd + ["-deffnm", step], cwd=workdir, env=md_env)

def run_analysis(workdir: Path, gmx: str):
    print("[INFO] Running Analysis (RMSD/RMSF)...")
    run_cmd([gmx, "make_ndx", "-f", "md.tpr", "-o", "index.ndx"], cwd=workdir, input_text="q\n")
    run_cmd([gmx, "rms", "-s", "md.tpr", "-f", "md.xtc", "-o", "rmsd.xvg", "-tu", "ns"], 
            cwd=workdir, input_text="1\n1\n")
    run_cmd([gmx, "rmsf", "-s", "md.tpr", "-f", "md.xtc", "-o", "rmsf.xvg", "-res"], 
            cwd=workdir, input_text="1\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--mdp_dir", default="mdp")
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    mdp_dir = Path(args.mdp_dir).resolve()
    cache_path = run_dir / "cache.jsonl"

    if not cache_path.exists(): return
    records = []
    with cache_path.open() as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))

    records.sort(key=lambda r: float(r.get("score", 1e9)))
    selected = records[:args.top]
    md_root = run_dir / "complex_md_amber"
    md_root.mkdir(exist_ok=True)

    for rank, rec in enumerate(selected, 1):
        print(f"\n=== Processing Rank {rank} ===\n")
        workdir = md_root / f"rank_{rank:03d}"
        if workdir.exists(): shutil.rmtree(workdir)
        workdir.mkdir(parents=True)

        try:
            pose_pdbqt = Path(rec["best_dock_pose_pdbqt"])
            if not pose_pdbqt.is_absolute(): pose_pdbqt = (run_dir / pose_pdbqt).resolve()

            build_ligand_files(workdir, pose_pdbqt)
            prepare_complex(run_dir, rec, workdir)
            build_amber_system(workdir)
            convert_to_gromacs(workdir)
            run_md_steps(workdir, args.gmx, mdp_dir, args.threads, args.gpu)
            run_analysis(workdir, args.gmx)
            print(f"\n✅ Rank {rank} Success!")
        except Exception as e:
            print(f"\n❌ Rank {rank} failed: {e}")
            continue

if __name__ == "__main__":
    main()
