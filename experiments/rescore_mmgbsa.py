#!/usr/bin/env python3
#rescore_mmgbsa.py
# -*- coding: utf-8 -*-
"""
MM/GBSA rescoring for aptamer-ligand complexes.

Key improvements over original:
  - Auto-detect ligand charge from ligand_ensemble/manifest.json
  - Always prefer original SDF for antechamber (not PDBQT-derived PDB)
  - Robust fallback chain for charge detection
  - Better error diagnostics
  - Fix MOL2 residue name mismatch (MOL -> LIG)
  - (NEW) Optional short GB minimization (sander) before MMPBSA to reduce clashes
"""

from __future__ import annotations

import argparse
import pandas as pd
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


# -----------------------
# utilities
# -----------------------
def _require_exe(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(
            f"Required executable not found: {name}\n"
            f"Install AmberTools and ensure it's on PATH.\n"
            f"Try: conda install -c conda-forge ambertools -y"
        )
    return exe


def _run_logged(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    allow_fail: bool = False,
) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    log_path.write_text(out, errors="ignore")
    if (not allow_fail) and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"Workdir: {cwd}\n"
            f"See log: {log_path}"
        )
    return p.returncode, out


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    _safe_mkdir(path.parent)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    _safe_mkdir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _jobs_from_topN_csv(topn_csv: Path, topM: Optional[int] = None) -> List[Dict]:
    if not topn_csv.exists():
        raise FileNotFoundError(f"topN.csv not found: {topn_csv}")

    jobs: List[Dict] = []
    with topn_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, start=1):
            if topM is not None and len(jobs) >= int(topM):
                break

            seq = (row.get("seq") or "").strip()
            if not seq:
                continue

            score_vina = row.get("score")
            try:
                score_vina_f = float(score_vina) if score_vina is not None and score_vina != "" else None
            except Exception:
                score_vina_f = None

            bf = row.get("best_frame")
            try:
                best_frame = int(float(bf)) if bf is not None and bf != "" else 1
            except Exception:
                best_frame = 1

            job = {
                "seq": seq,
                "rank": i,
                "best_frame": best_frame,
                "score_vina": score_vina_f,
                "receptor_pdb": (row.get("best_receptor_pdb") or row.get("receptor_pdb") or "").strip(),
                "pose_pdbqt": (row.get("best_dock_pose_pdbqt") or row.get("pose_pdbqt") or "").strip(),
                "ligand_sdf": (row.get("best_ligand_sdf") or row.get("ligand_sdf") or "").strip(),
                "eval_hash": (row.get("eval_hash") or "").strip(),
            }
            jobs.append(job)

    return jobs


# -----------------------
# PDB fixing helpers
# -----------------------
_RNA_MAP = {
    "ADE": "A", "A": "A",
    "URA": "U", "U": "U",
    "GUA": "G", "G": "G",
    "CYT": "C", "C": "C",
    "RA": "A", "RC": "C", "RG": "G", "RU": "U",
}


def _normalize_rna_resname(res: str) -> str:
    r = (res or "").strip()
    if not r:
        return r
    if r in _RNA_MAP:
        return _RNA_MAP[r]
    letters = re.sub(r"[^A-Za-z]", "", r).upper()
    if letters in _RNA_MAP:
        return _RNA_MAP[letters]
    if len(letters) >= 2 and letters.startswith("R") and letters[1] in "ACGU":
        return letters[1]
    if len(letters) >= 1 and letters[0] in "ACGU":
        return letters[0]
    return r


def _fix_rna_pdb_resnames(in_pdb: Path, out_pdb: Path) -> None:
    txt = in_pdb.read_text(errors="ignore").splitlines()
    out_lines: List[str] = []
    for ln in txt:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            out_lines.append(ln)
            continue
        if len(ln) < 80:
            ln = ln.rstrip("\n")
            ln = ln + (" " * (80 - len(ln)))
        resname = ln[17:20]
        new_res = _normalize_rna_resname(resname)
        ln = ln[:17] + f"{new_res:>3s}" + ln[20:]
        if ln[21].strip() == "":
            ln = ln[:21] + "A" + ln[22:]
        out_lines.append(ln.rstrip())
    out_pdb.write_text("\n".join(out_lines) + "\n", errors="ignore")


def _strip_5prime_phosphate(in_pdb: Path, out_pdb: Path) -> int:
    removed = 0
    first_key = None

    def _res_key(line: str):
        chain = line[21:22]
        resseq = line[22:26]
        icode = line[26:27]
        return (chain, resseq, icode)

    with in_pdb.open("r", errors="ignore") as f, out_pdb.open("w") as g:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")) and len(ln) >= 54:
                key = _res_key(ln)
                if first_key is None:
                    first_key = key
                if key == first_key:
                    atom = ln[12:16].strip()
                    if atom in {"P", "OP1", "OP2", "O1P", "O2P"}:
                        removed += 1
                        continue
            g.write(ln)
    return removed


def _sanitize_ligand_pdb(in_pdb: Path, out_pdb: Path, resname: str = "LIG", chain: str = "Z") -> None:
    txt = in_pdb.read_text(errors="ignore").splitlines()
    out_lines: List[str] = []
    for ln in txt:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if len(ln) < 80:
            ln = ln + (" " * (80 - len(ln)))
        ln = ln[:17] + f"{resname:>3s}" + ln[20:]
        ln = ln[:21] + chain + ln[22:]
        ln = ln[:22] + f"{1:>4d}" + ln[26:]
        out_lines.append(ln.rstrip())
    out_pdb.write_text("\n".join(out_lines) + "\n", errors="ignore")


def _make_complex_pdb(receptor_pdb: Path, ligand_pdb: Path, out_complex: Path) -> None:
    rec_txt = receptor_pdb.read_text(errors="ignore").splitlines()
    lig_txt = ligand_pdb.read_text(errors="ignore").splitlines()
    rec_lines: List[str] = []
    for ln in rec_txt:
        if ln.startswith(("END", "MODEL", "ENDMDL")):
            continue
        if ln.startswith(("ATOM", "HETATM", "TER")):
            rec_lines.append(ln.rstrip())
    if rec_lines and not rec_lines[-1].startswith("TER"):
        rec_lines.append("TER")
    lig_lines: List[str] = []
    for ln in lig_txt:
        if ln.startswith(("END", "MODEL", "ENDMDL", "TER")):
            continue
        if ln.startswith(("ATOM", "HETATM")):
            lig_lines.append(ln.rstrip())
    lig_lines.append("TER")
    lig_lines.append("END")
    out_complex.write_text("\n".join(rec_lines + lig_lines) + "\n", errors="ignore")


# -----------------------
# ligand coordinate injection
# -----------------------
def _pdb_atom_lines(path: Path, only_resname: Optional[str] = None) -> List[str]:
    lines = []
    for ln in path.read_text(errors="ignore").splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        if only_resname is not None:
            if len(ln) < 20:
                continue
            if ln[17:20].strip() != only_resname:
                continue
        if len(ln) < 80:
            ln = ln + (" " * (80 - len(ln)))
        lines.append(ln)
    return lines


def _pdb_element(ln: str) -> str:
    if len(ln) >= 78:
        el = ln[76:78].strip()
        if el:
            return el.upper()
    name = ln[12:16].strip()
    name = re.sub(r"\d+", "", name)
    if not name:
        return ""
    if len(name) >= 2 and name[0].isalpha() and name[1].islower():
        return (name[0] + name[1]).upper()
    return name[0].upper()


def _is_hydrogen_line(ln: str) -> bool:
    el = _pdb_element(ln)
    if el == "H":
        return True
    an = ln[12:16].strip().upper()
    return an.startswith("H")


def _extract_xyz(ln: str) -> Tuple[float, float, float]:
    return float(ln[30:38]), float(ln[38:46]), float(ln[46:54])


def _set_xyz(ln: str, x: float, y: float, z: float) -> str:
    if len(ln) < 80:
        ln = ln + (" " * (80 - len(ln)))
    return ln[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + ln[54:]


def _write_ligand_tpl_pdb_from_mol2(work: Path) -> Path:
    tpl = work / "ligand_tpl.pdb"
    _run_logged(
        ["antechamber", "-i", "ligand.mol2", "-fi", "mol2",
         "-o", str(tpl), "-fo", "pdb", "-s", "2"],
        cwd=work, log_path=work / "antechamber_mol2_to_pdb.log",
    )
    return tpl


def _inject_pose_coords_to_tpl(tpl_pdb: Path, pose_pdb: Path, out_pdb: Path) -> None:
    tpl_lines = _pdb_atom_lines(tpl_pdb)
    pose_lines = _pdb_atom_lines(pose_pdb)

    if len(tpl_lines) == len(pose_lines):
        out = []
        for tln, pln in zip(tpl_lines, pose_lines):
            x, y, z = _extract_xyz(pln)
            out.append(_set_xyz(tln, x, y, z).rstrip())
        out_pdb.write_text("\n".join(out) + "\n", errors="ignore")
        return

    tpl_heavy = [i for i, ln in enumerate(tpl_lines) if not _is_hydrogen_line(ln)]
    pose_heavy = [i for i, ln in enumerate(pose_lines) if not _is_hydrogen_line(ln)]

    if len(tpl_heavy) != len(pose_heavy):
        raise RuntimeError(
            f"Atom count mismatch: tpl={len(tpl_lines)}(heavy={len(tpl_heavy)}) "
            f"pose={len(pose_lines)}(heavy={len(pose_heavy)})"
        )

    out = tpl_lines[:]
    for ti, pi in zip(tpl_heavy, pose_heavy):
        x, y, z = _extract_xyz(pose_lines[pi])
        out[ti] = _set_xyz(tpl_lines[ti], x, y, z)
    out_pdb.write_text("\n".join([ln.rstrip() for ln in out]) + "\n", errors="ignore")


def _fix_mol2_resname(mol2_path: Path, new_resname: str = "LIG") -> None:
    """Fix residue name in MOL2 for tleap compatibility."""
    if not mol2_path.exists():
        return

    lines = mol2_path.read_text(errors="ignore").splitlines()
    out_lines: List[str] = []
    in_atom_section = False

    for line in lines:
        if line.strip().startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            out_lines.append(line)
            continue
        elif line.strip().startswith("@<TRIPOS>"):
            in_atom_section = False
            out_lines.append(line)
            continue

        if in_atom_section:
            parts = line.split()
            if len(parts) >= 8:
                parts[7] = new_resname
                try:
                    line = "{:>7s} {:<8s} {:>13s} {:>9s} {:>9s} {:<5s} {:>5s} {:<8s}".format(
                        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
                    )
                    if len(parts) > 8:
                        line += " " + " ".join(parts[8:])
                except (ValueError, IndexError):
                    parts[7] = new_resname
                    line = " ".join(parts)

        out_lines.append(line)

    mol2_path.write_text("\n".join(out_lines) + "\n", errors="ignore")


# -----------------------
# charge detection
# -----------------------
def _detect_net_charge_from_pdbqt(pdbqt: Path) -> Optional[int]:
    """Detect charge from PDBQT Gasteiger charges (last resort)."""
    try:
        s = 0.0
        n = 0
        for ln in pdbqt.read_text(errors="ignore").splitlines():
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                q = float(parts[-1])
            except Exception:
                continue
            s += q
            n += 1
        if n == 0:
            return None
        return int(round(s))
    except Exception:
        return None


def _detect_net_charge_from_sdf_rdkit(sdf_path: Path) -> Optional[int]:
    """Detect formal charge using RDKit (reliable)."""
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        if suppl and len(suppl) > 0:
            mol = suppl[0]
            if mol:
                return Chem.GetFormalCharge(mol)
    except Exception:
        pass
    return None


def _detect_net_charge_from_sdf_property(sdf_path: Path) -> Optional[int]:
    """Read NET_CHARGE property from SDF (written by our ligand prep)."""
    try:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        if suppl and len(suppl) > 0:
            mol = suppl[0]
            if mol and mol.HasProp("NET_CHARGE"):
                return mol.GetIntProp("NET_CHARGE")
    except Exception:
        pass
    return None


def determine_ligand_charge(
    job: Dict,
    run_dir: Path,
    net_charge_override: Optional[int],
    work: Path,
) -> Tuple[int, str]:
    """
    Determine ligand charge with priority:
      1. --net_charge CLI override
      2. ligand_ensemble/manifest.json (charge_info.formal_charge)
      3. ligand_ensemble/charge.json
      4. ligand_ensemble/charge.txt
      5. SDF NET_CHARGE property from job ligand_sdf
      6. RDKit formal charge from job ligand_sdf
      7. RDKit formal charge from original data/ligand.sdf
      8. PDBQT Gasteiger sum
      9. Fallback 0
    """
    if net_charge_override is not None:
        return int(net_charge_override), "cli_override"

    manifest_path = run_dir / "ligand_ensemble" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            ci = manifest.get("charge_info", {})
            if "formal_charge" in ci:
                charge = int(ci["formal_charge"])
                if not ci.get("validation_passed", True):
                    (work / "charge_warning.txt").write_text(
                        f"Charge validation warning: {ci.get('validation_notes', '')}\n"
                    )
                return charge, "manifest"
        except Exception:
            pass

    cj = run_dir / "ligand_ensemble" / "charge.json"
    if cj.exists():
        try:
            info = json.loads(cj.read_text())
            if "formal_charge" in info:
                return int(info["formal_charge"]), "charge_json"
        except Exception:
            pass

    ct = run_dir / "ligand_ensemble" / "charge.txt"
    if ct.exists():
        try:
            return int(ct.read_text().strip()), "charge_txt"
        except Exception:
            pass

    ligand_sdf = job.get("best_ligand_sdf") or job.get("ligand_sdf") or ""
    if ligand_sdf:
        sdf_path = Path(ligand_sdf)
        if sdf_path.exists():
            c = _detect_net_charge_from_sdf_property(sdf_path)
            if c is not None:
                return c, "sdf_property"
            c = _detect_net_charge_from_sdf_rdkit(sdf_path)
            if c is not None:
                return c, "rdkit_formal"

    for orig in [
        run_dir / "data" / "ligand.sdf",
        Path("data/ligand.sdf").resolve(),
        Path("data/ligand.pdb").resolve(),
    ]:
        if orig.exists():
            c = _detect_net_charge_from_sdf_rdkit(orig)
            if c is not None:
                return c, f"rdkit_original({orig.name})"

    pose_pdbqt = job.get("best_dock_pose_pdbqt") or job.get("pose_pdbqt") or ""
    if pose_pdbqt:
        pp = Path(pose_pdbqt)
        if pp.exists():
            c = _detect_net_charge_from_pdbqt(pp)
            if c is not None:
                return c, "pdbqt_gasteiger"

    (work / "charge_warning.txt").write_text(
        "WARNING: Could not auto-detect ligand charge from any source.\n"
        "Defaulting to 0. If ligand is charged, set --net_charge explicitly.\n"
    )
    return 0, "fallback"


def _find_original_ligand_sdf(job: Dict, run_dir: Path) -> Optional[Path]:
    """Find the ORIGINAL ligand SDF file (with correct bond orders)."""
    for key in ["best_ligand_sdf", "ligand_sdf"]:
        val = (job.get(key) or "").strip()
        if val:
            p = Path(val)
            if p.exists() and p.stat().st_size > 0:
                return p

    ensemble_dir = run_dir / "ligand_ensemble"
    if ensemble_dir.exists():
        manifest_path = ensemble_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                items = manifest.get("items", [])
                if items:
                    p = Path(items[0].get("path", ""))
                    if p.exists() and p.stat().st_size > 0:
                        return p
            except Exception:
                pass
        conf1 = ensemble_dir / "conf_001.sdf"
        if conf1.exists():
            return conf1

    for candidate in [
        run_dir / "data" / "ligand.sdf",
        Path("data/ligand.sdf").resolve(),
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate

    resolved_cfg = run_dir / "resolved_config.yaml"
    if resolved_cfg.exists():
        try:
            import yaml
            cfg = yaml.safe_load(resolved_cfg.read_text())
            lig = cfg.get("inputs", {}).get("ligand", "")
            if lig:
                p = Path(lig).expanduser().resolve()
                if p.exists():
                    return p
        except Exception:
            pass

    return None


# -----------------------
# MMPBSA output parsing
# -----------------------
def _parse_mmpbsa_final_results(final_dat: Path) -> Optional[float]:
    if not final_dat.exists():
        return None
    txt = final_dat.read_text(errors="ignore")
    num = r"(-?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
    m = re.search(r"DELTA\s+TOTAL\s+" + num, txt)
    if m:
        return float(m.group(1))
    m2 = re.search(r"DELTA\s+G\s+binding.*?" + num, txt, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        return float(m2.group(1))
    return None


# -----------------------
# MM/GBSA config & core
# -----------------------
@dataclass
class MMGBSAConfig:
    leaprc_nucleic: str = "leaprc.RNA.OL3"
    leaprc_smallmol: str = "leaprc.gaff2"
    at_forcefield: str = "gaff2"
    charge_method: str = "bcc"
    no_gas_fallback: bool = False
    igb: int = 5
    saltcon: float = 0.150
    ph: float = 7.4
    keep_tmp: bool = True

    # (NEW) minimization options
    minimize: bool = False
    min_maxcyc: int = 1500
    min_ncyc: int = 500
    min_restraint: bool = False
    min_restraint_wt: float = 2.0
    min_restraint_mask: str = "!:LIG & !@H="


def _write_tleap_in(work: Path, cfg: MMGBSAConfig) -> Path:
    leap_in = work / "tleap.in"

    lines = [
        f"source {cfg.leaprc_nucleic}",
        f"source {cfg.leaprc_smallmol}",
        "",
    ]

    # If using GBn/GBn2, set radii to bondi to avoid sander radius errors
    if int(cfg.igb) in (7, 8):
        lines += [
            "set default PBRadii bondi",
            "",
        ]

    lines += [
        "LIG = loadMol2 ligand.mol2",
        "loadamberparams ligand.frcmod",
        "",
        "REC = loadpdb receptor_amber.pdb",
        "COM = loadpdb complex.pdb",
        "",
        "check REC",
        "check LIG",
        "check COM",
        "",
        "saveamberparm REC receptor.prmtop receptor.inpcrd",
        "saveamberparm LIG ligand.prmtop ligand.inpcrd",
        "saveamberparm COM complex.prmtop complex.inpcrd",
        "quit",
        "",
    ]

    leap_in.write_text("\n".join(lines), errors="ignore")
    return leap_in


def _write_cpptraj_in(work: Path) -> Path:
    inp = work / "cpptraj.in"
    inp.write_text(
        "\n".join([
            "parm complex.prmtop",
            "trajin complex.inpcrd 1 1",
            "trajout complex.mdcrd",
            "go",
            "quit",
            "",
        ]),
        errors="ignore",
    )
    return inp


def _write_cpptraj_min_in(work: Path, inp_rst7: str = "complex_min.rst7", out_mdcrd: str = "complex_min.mdcrd") -> Path:
    inp = work / "cpptraj_min.in"
    inp.write_text(
        "\n".join([
            "parm complex.prmtop",
            f"trajin {inp_rst7} 1 1",
            f"trajout {out_mdcrd} mdcrd",
            "go",
            "quit",
            "",
        ]),
        errors="ignore",
    )
    return inp


def _write_mmpbsa_in(work: Path, cfg: MMGBSAConfig) -> Path:
    inp = work / "mmpbsa.in"
    inp.write_text(
        "\n".join([
            "&general",
            "  startframe=1, endframe=1, interval=1,",
            "  verbose=1,",
            "/",
            "&gb",
            f"  igb={int(cfg.igb)}, saltcon={float(cfg.saltcon):.3f},",
            "/",
            "",
        ]),
        errors="ignore",
    )
    return inp


def _write_min_in(work: Path, cfg: MMGBSAConfig) -> Path:
    """
    Write sander minimization input. Uses the SAME igb/saltcon as MMPBSA to be consistent.
    """
    lines = [
        "Minimize complex (GB)",
        "&cntrl",
        "  imin=1,",
        f"  maxcyc={int(cfg.min_maxcyc)},",
        f"  ncyc={int(cfg.min_ncyc)},",
        "  ntb=0,",
        "  cut=999.0,",
        f"  igb={int(cfg.igb)},",
        f"  saltcon={float(cfg.saltcon):.3f},",
        "  ntpr=100,",
    ]
    if cfg.min_restraint:
        lines += [
            "  ntr=1,",
            f"  restraint_wt={float(cfg.min_restraint_wt):.3f},",
            f"  restraintmask='{cfg.min_restraint_mask}',",
        ]
    else:
        lines += ["  ntr=0,"]

    lines += ["/", ""]
    p = work / "min.in"
    p.write_text("\n".join(lines), errors="ignore")
    return p


def _pick_job_paths(job: Dict) -> Tuple[Path, Path]:
    rec = job.get("best_receptor_pdb") or job.get("receptor_pdb") or ""
    pose = job.get("best_dock_pose_pdbqt") or job.get("pose_pdbqt") or ""
    if not rec or not pose:
        raise KeyError("Job missing receptor/pose path.")
    return Path(rec).expanduser().resolve(), Path(pose).expanduser().resolve()


def run_one_mmgbsa(
    job: Dict,
    out_dir: Path,
    cfg: MMGBSAConfig,
    net_charge_override: Optional[int],
    run_dir: Path,
) -> Dict:
    _require_exe("obabel")
    _require_exe("pdb4amber")
    _require_exe("antechamber")
    _require_exe("parmchk2")
    _require_exe("tleap")
    _require_exe("cpptraj")
    _require_exe("MMPBSA.py")
    if cfg.minimize:
        _require_exe("sander")

    receptor_pdb, pose_pdbqt = _pick_job_paths(job)

    rank = int(job.get("rank", 0) or 0)
    best_frame = int(job.get("best_frame", 1) or 1)
    work = out_dir / f"rank_{rank:03d}_f{best_frame:03d}"
    _safe_mkdir(work)

    # -----------------------
    # 1) receptor prep
    # -----------------------
    receptor_raw = work / "receptor_raw.pdb"
    receptor_fix = work / "receptor_fix_resname.pdb"
    receptor_amber_tmp = work / "receptor_amber_tmp.pdb"
    receptor_amber_final = work / "receptor_amber.pdb"
    receptor_amber_stripped = work / "receptor_amber_stripped.pdb"

    shutil.copyfile(receptor_pdb, receptor_raw)
    _fix_rna_pdb_resnames(receptor_raw, receptor_fix)

    _run_logged(
        ["pdb4amber", "-i", str(receptor_fix), "-o", str(receptor_amber_tmp), "--nohyd"],
        cwd=work, log_path=work / "pdb4amber.log",
    )

    n_removed = _strip_5prime_phosphate(receptor_amber_tmp, receptor_amber_stripped)
    if n_removed > 0:
        shutil.copyfile(receptor_amber_stripped, receptor_amber_final)
        print(f"[mmgbsa] stripped 5' phosphate: removed={n_removed} -> {receptor_amber_final.name}")
    else:
        shutil.copyfile(receptor_amber_tmp, receptor_amber_final)

    # -----------------------
    # 2) ligand pose prep
    # -----------------------
    lig_pose_raw = work / "ligand_pose_raw.pdb"
    lig_pose_sane = work / "ligand_pose.pdb"
    lig_pose_h = work / "ligand_pose_H.pdb"

    _run_logged(
        ["obabel", "-ipdbqt", str(pose_pdbqt), "-opdb", "-O",
         str(lig_pose_raw), "-f", "1", "-l", "1"],
        cwd=work, log_path=work / "obabel_pose_to_pdb.log",
    )
    _sanitize_ligand_pdb(lig_pose_raw, lig_pose_sane, resname="LIG", chain="Z")

    _run_logged(
        ["obabel", "-ipdb", str(lig_pose_sane), "-opdb", "-O",
         str(lig_pose_h), "-p", f"{cfg.ph:.2f}"],
        cwd=work, log_path=work / "obabel_addh.log", allow_fail=True,
    )
    if (not lig_pose_h.exists()) or lig_pose_h.stat().st_size == 0:
        shutil.copyfile(lig_pose_sane, lig_pose_h)

    # -----------------------
    # 3) Determine ligand charge
    # -----------------------
    net_charge, charge_source = determine_ligand_charge(
        job=job, run_dir=run_dir,
        net_charge_override=net_charge_override, work=work,
    )

    charge_log = work / "charge_detection.log"
    charge_log.write_text(
        f"Net charge: {net_charge}\n"
        f"Source: {charge_source}\n"
        f"Override: {net_charge_override}\n"
    )
    print(f"[mmgbsa] rank={rank} charge={net_charge} (source={charge_source})")

    # -----------------------
    # 4) ligand parameterization
    # -----------------------
    lig_mol2 = work / "ligand.mol2"
    lig_frcmod = work / "ligand.frcmod"

    original_sdf = _find_original_ligand_sdf(job, run_dir)

    antechamber_input: Optional[Path] = None
    antechamber_format = "sdf"

    if original_sdf is not None:
        lig_input = work / "ligand_input.sdf"
        shutil.copyfile(original_sdf, lig_input)
        antechamber_input = lig_input
        antechamber_format = "sdf"
        (work / "ligand_source.txt").write_text(
            f"Using original SDF: {original_sdf}\n"
            f"This preserves bond order information for antechamber.\n"
        )
    else:
        lig_input = work / "ligand_input.sdf"
        rc, _ = _run_logged(
            ["obabel", "-ipdbqt", str(pose_pdbqt), "-osdf", "-O", str(lig_input),
             "-f", "1", "-l", "1", "-h"],
            cwd=work, log_path=work / "obabel_pdbqt_to_sdf.log",
            allow_fail=True,
        )
        if rc == 0 and lig_input.exists() and lig_input.stat().st_size > 0:
            antechamber_input = lig_input
            antechamber_format = "sdf"
        else:
            antechamber_input = lig_pose_h
            antechamber_format = "pdb"

        (work / "ligand_source.txt").write_text(
            f"WARNING: No original SDF found for ligand.\n"
            f"Using: {antechamber_input} (format={antechamber_format})\n"
            f"This may fail for ligands with complex bonding.\n"
        )

    def _do_antechamber(method: str) -> None:
        _run_logged(
            [
                "antechamber",
                "-i", str(antechamber_input),
                "-fi", antechamber_format,
                "-o", str(lig_mol2),
                "-fo", "mol2",
                "-at", cfg.at_forcefield,
                "-c", method,
                "-nc", str(int(net_charge)),
                "-s", "2",
            ],
            cwd=work, log_path=work / "antechamber.log",
        )

    try:
        _do_antechamber(cfg.charge_method)
    except Exception as e1:
        if (cfg.charge_method == "bcc") and (not cfg.no_gas_fallback):
            (work / "antechamber_fallback_reason.txt").write_text(
                f"bcc failed: {repr(e1)}\nTrying gas fallback.\n"
            )
            _do_antechamber("gas")
        else:
            raise

    _run_logged(
        ["parmchk2", "-i", str(lig_mol2), "-f", "mol2", "-o", str(lig_frcmod)],
        cwd=work, log_path=work / "parmchk2.log",
    )

    # Fix residue name in mol2 to match complex.pdb
    _fix_mol2_resname(lig_mol2, new_resname="LIG")

    # -----------------------
    # 5) Build ligand pose with correct atom names
    # -----------------------
    lig_tpl = _write_ligand_tpl_pdb_from_mol2(work)
    lig_pose_named = work / "ligand_pose_named.pdb"
    _inject_pose_coords_to_tpl(lig_tpl, lig_pose_h, lig_pose_named)

    # -----------------------
    # 6) complex.pdb
    # -----------------------
    complex_pdb = work / "complex.pdb"
    _make_complex_pdb(receptor_amber_final, lig_pose_named, complex_pdb)

    # -----------------------
    # 7) tleap
    # -----------------------
    leap_in = _write_tleap_in(work, cfg)
    _run_logged(["tleap", "-f", str(leap_in)], cwd=work, log_path=work / "tleap.log")

    for fn in ["receptor.prmtop", "ligand.prmtop", "complex.prmtop"]:
        ptop = work / fn
        if (not ptop.exists()) or ptop.stat().st_size < 1000:
            raise RuntimeError(f"tleap failed to generate {fn}. See {work / 'tleap.log'}")

    # -----------------------
    # 8) cpptraj (original single-frame mdcrd)
    # -----------------------
    cpp_in = _write_cpptraj_in(work)
    _run_logged(["cpptraj", "-i", str(cpp_in)], cwd=work, log_path=work / "cpptraj.log")

    # -----------------------
    # 9) Optional minimization + MMPBSA
    # -----------------------
    traj_for_mmpbsa = "complex.mdcrd"
    final_name = "FINAL_RESULTS_MMPBSA.dat"

    if cfg.minimize:
        # sander minimization
        min_in = _write_min_in(work, cfg)
        _run_logged(
            [
                "sander", "-O",
                "-i", str(min_in),
                "-p", "complex.prmtop",
                "-c", "complex.inpcrd",
                "-ref", "complex.inpcrd",
                "-r", "complex_min.rst7",
                "-o", "complex_min.out",
            ],
            cwd=work, log_path=work / "sander_min.log",
        )

        # convert minimized restart to mdcrd
        cpp_min = _write_cpptraj_min_in(work, inp_rst7="complex_min.rst7", out_mdcrd="complex_min.mdcrd")
        _run_logged(
            ["cpptraj", "-p", "complex.prmtop", "-i", str(cpp_min)],
            cwd=work, log_path=work / "cpptraj_min.log"
        )

        if not (work / "complex_min.mdcrd").exists():
            raise RuntimeError("Minimization enabled but complex_min.mdcrd not found. Check cpptraj_min.log")

        traj_for_mmpbsa = "complex_min.mdcrd"
        final_name = "FINAL_RESULTS_MMPBSA_MIN.dat"

    # run MMPBSA
    mmp_in = _write_mmpbsa_in(work, cfg)
    _run_logged(
        [
            "MMPBSA.py", "-O",
            "-i", str(mmp_in),
            "-cp", "complex.prmtop",
            "-rp", "receptor.prmtop",
            "-lp", "ligand.prmtop",
            "-y", traj_for_mmpbsa,
            "-o", final_name,
        ],
        cwd=work, log_path=work / "mmpbsa.log",
    )

    final_dat = work / final_name
    dg = _parse_mmpbsa_final_results(final_dat)

    rec = dict(job)
    rec.update({
        "mmgbsa_ok": bool(dg is not None),
        "mmgbsa_dg_bind": dg if dg is not None else 1e9,
        "mmgbsa_workdir": str(work),
        "mmgbsa_final": str(final_dat) if final_dat.exists() else "",
        "mmgbsa_log": str(work / "mmpbsa.log") if (work / "mmpbsa.log").exists() else "",
        "mmgbsa_charge": int(net_charge),
        "mmgbsa_charge_source": charge_source,
        "mmgbsa_charge_method": cfg.charge_method,
        "mmgbsa_igb": int(cfg.igb),
        "mmgbsa_saltcon": float(cfg.saltcon),
        "mmgbsa_ph": float(cfg.ph),

        # minimization metadata
        "mmgbsa_minimized": bool(cfg.minimize),
        "mmgbsa_min_maxcyc": int(cfg.min_maxcyc) if cfg.minimize else 0,
        "mmgbsa_min_ncyc": int(cfg.min_ncyc) if cfg.minimize else 0,
        "mmgbsa_min_restraint": bool(cfg.min_restraint) if cfg.minimize else False,
        "mmgbsa_min_restraint_wt": float(cfg.min_restraint_wt) if cfg.minimize else 0.0,
        "mmgbsa_min_restraint_mask": str(cfg.min_restraint_mask) if cfg.minimize else "",
    })
    if dg is None:
        rec["mmgbsa_ok"] = False
        rec["mmgbsa_error"] = (
            f"parse_failed: {final_dat}" if final_dat.exists()
            else "missing_final_results"
        )
    return rec
def write_mmgbsa_back_to_cache(run_dir: Path, rows: List[Dict]) -> None:
    """
    Merge MMGBSA results back into cache.jsonl
    Priority: keep best (lowest dg_bind) if multiple entries
    """
    cache_path = run_dir / "cache.jsonl"
    if not cache_path.exists():
        print("[mmgbsa] cache.jsonl not found, skip write-back")
        return

    cache_rows = []
    with cache_path.open("r", errors="ignore") as f:
        for line in f:
            if line.strip():
                cache_rows.append(json.loads(line))

    gbsa_map: Dict[str, Dict] = {}
    for r in rows:
        seq = r.get("seq")
        if not seq:
            continue
        dg = float(r.get("mmgbsa_dg_bind", 1e9))
        ok = bool(r.get("mmgbsa_ok", False))

        if seq not in gbsa_map:
            gbsa_map[seq] = r
        else:
            if dg < float(gbsa_map[seq].get("mmgbsa_dg_bind", 1e9)):
                gbsa_map[seq] = r

    for rec in cache_rows:
        seq = rec.get("seq")
        if seq in gbsa_map:
            gb = gbsa_map[seq]
            rec.update({
                "mmgbsa_ok": bool(gb.get("mmgbsa_ok", False)),
                "mmgbsa_dg_bind": float(gb.get("mmgbsa_dg_bind", 1e9)),
                "mmgbsa_charge": gb.get("mmgbsa_charge"),
                "mmgbsa_charge_source": gb.get("mmgbsa_charge_source"),
                "mmgbsa_igb": gb.get("mmgbsa_igb"),
                "mmgbsa_saltcon": gb.get("mmgbsa_saltcon"),
                "mmgbsa_minimized": gb.get("mmgbsa_minimized", False),
            })

    with cache_path.open("w") as f:
        for rec in cache_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[mmgbsa] ✅ results written back to {cache_path}")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MM/GBSA rescoring with automatic charge detection (+ optional minimization)"
    )
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--topN_csv", type=str, default=None)
    ap.add_argument("--iter_out", type=int, default=1)
    ap.add_argument("--overwrite_manifest", action="store_true")
    ap.add_argument("--create_only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--topM", type=int, default=None)
    ap.add_argument("--igb", type=int, default=5)
    ap.add_argument("--saltcon", type=float, default=0.150)
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--charge_method", type=str, default="bcc", choices=["bcc", "gas"])
    ap.add_argument("--no_gas_fallback", action="store_true")
    ap.add_argument("--net_charge", type=int, default=None,
                    help="Override ligand net charge (auto-detect if not set)")
    ap.add_argument("--keep_tmp", action="store_true")

    # (NEW) minimization flags
    ap.add_argument("--minimize", action="store_true",
                    help="Run a short GB minimization (sander) on the complex before MMPBSA")
    ap.add_argument("--min_maxcyc", type=int, default=1500, help="sander maxcyc for minimization")
    ap.add_argument("--min_ncyc", type=int, default=500, help="sander ncyc for minimization")
    ap.add_argument("--min_restraint", action="store_true",
                    help="Restrain receptor heavy atoms during minimization (recommended)")
    ap.add_argument("--min_restraint_wt", type=float, default=2.0, help="Restraint weight (kcal/mol/A^2)")
    ap.add_argument("--min_restraint_mask", type=str, default="!:LIG & !@H=",
                    help="Amber mask for restrained atoms (default restrains non-ligand heavy atoms)")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    rescore_root = run_dir / "rescore"
    _safe_mkdir(rescore_root)

    manifest_path = run_dir / "ligand_ensemble" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            ci = manifest.get("charge_info", {})
            print(f"[mmgbsa] Ligand charge from ensemble: {ci.get('formal_charge', 'N/A')} "
                  f"(method={ci.get('detection_method', 'N/A')}, "
                  f"validation={'PASSED' if ci.get('validation_passed', True) else 'FAILED'})")
        except Exception as e:
            print(f"[mmgbsa] Warning: Could not read manifest: {e}")
    else:
        charge_txt = run_dir / "ligand_ensemble" / "charge.txt"
        if charge_txt.exists():
            print(f"[mmgbsa] Ligand charge from charge.txt: {charge_txt.read_text().strip()}")
        else:
            print("[mmgbsa] Note: No ligand_ensemble found. Will auto-detect charge per job.")

    if args.net_charge is not None:
        print(f"[mmgbsa] CLI override: --net_charge={args.net_charge}")

    if args.minimize:
        print(f"[mmgbsa] Minimization enabled: maxcyc={args.min_maxcyc} ncyc={args.min_ncyc} "
              f"restraint={args.min_restraint} wt={args.min_restraint_wt} mask='{args.min_restraint_mask}'")

    if args.topN_csv is not None:
        topn_csv = Path(args.topN_csv).expanduser()
        if not topn_csv.is_absolute():
            topn_csv = (run_dir / topn_csv).resolve()
        iter_dir = rescore_root / f"iter_{int(args.iter_out):03d}"
        _safe_mkdir(iter_dir)
        manifest = iter_dir / "manifest.jsonl"
        if manifest.exists() and not args.overwrite_manifest:
            raise FileExistsError(f"manifest exists: {manifest} (use --overwrite_manifest)")
        jobs = _jobs_from_topN_csv(topn_csv, topM=args.topM)
        _write_jsonl(manifest, jobs)
        if args.create_only:
            print(f"[OK] manifest created: {manifest}")
            return
    else:
        iter_dirs = sorted([p for p in rescore_root.glob("iter_*") if p.is_dir()])
        if not iter_dirs:
            raise FileNotFoundError(f"No rescore/iter_*/ under {rescore_root}")
        if args.iter is None:
            iter_dir = iter_dirs[-1]
        else:
            iter_dir = rescore_root / f"iter_{int(args.iter):03d}"
            if not iter_dir.exists():
                raise FileNotFoundError(f"Iteration dir not found: {iter_dir}")
        manifest = iter_dir / "manifest.jsonl"
        jobs = _read_jsonl(manifest)

    if not jobs:
        print("[mmgbsa] empty manifest")
        return

    if args.topM is not None:
        jobs = jobs[: int(args.topM)]

    out_dir = iter_dir / "mmgbsa"
    _safe_mkdir(out_dir)

    cfg = MMGBSAConfig(
        igb=int(args.igb),
        saltcon=float(args.saltcon),
        ph=float(args.ph),
        charge_method=str(args.charge_method),
        no_gas_fallback=bool(args.no_gas_fallback),
        keep_tmp=bool(args.keep_tmp),

        minimize=bool(args.minimize),
        min_maxcyc=int(args.min_maxcyc),
        min_ncyc=int(args.min_ncyc),
        min_restraint=bool(args.min_restraint),
        min_restraint_wt=float(args.min_restraint_wt),
        min_restraint_mask=str(args.min_restraint_mask),
    )

    results: List[Dict] = []
    errors: List[Dict] = []

    def _handle_err(j: Dict, e: Exception) -> Dict:
        rank = int(j.get("rank", 0) or 0)
        best_frame = int(j.get("best_frame", 1) or 1)
        workdir = str((out_dir / f"rank_{rank:03d}_f{best_frame:03d}").resolve())
        er = dict(j)
        er["mmgbsa_ok"] = False
        er["mmgbsa_dg_bind"] = 1e9
        er["mmgbsa_error"] = repr(e)
        er["mmgbsa_workdir"] = workdir
        return er

    print(f"[mmgbsa] Running {len(jobs)} jobs (workers={args.workers})")

    if int(args.workers) <= 1:
        for idx, j in enumerate(jobs, 1):
            print(f"[mmgbsa] --- job {idx}/{len(jobs)} seq={j.get('seq','')[:15]}... ---")
            try:
                results.append(run_one_mmgbsa(j, out_dir, cfg, args.net_charge, run_dir))
            except Exception as e:
                print(f"[mmgbsa] FAILED: {repr(e)[:120]}")
                errors.append(_handle_err(j, e))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {
                ex.submit(run_one_mmgbsa, j, out_dir, cfg, args.net_charge, run_dir): j
                for j in jobs
            }
            for fu in as_completed(futs):
                j = futs[fu]
                try:
                    results.append(fu.result())
                except Exception as e:
                    errors.append(_handle_err(j, e))

    merged = results + errors
    merged.sort(key=lambda x: float(x.get("mmgbsa_dg_bind", 1e9)))

    out_jsonl = iter_dir / "mmgbsa_results.jsonl"
    out_csv = iter_dir / "mmgbsa_results.csv"

    fields = [
        "seq", "rank", "best_frame", "score_vina",
        "mmgbsa_dg_bind", "mmgbsa_ok",
        "mmgbsa_charge", "mmgbsa_charge_source", "mmgbsa_charge_method",
        "mmgbsa_igb", "mmgbsa_saltcon", "mmgbsa_ph",

        # minimization info
        "mmgbsa_minimized", "mmgbsa_min_maxcyc", "mmgbsa_min_ncyc",
        "mmgbsa_min_restraint", "mmgbsa_min_restraint_wt", "mmgbsa_min_restraint_mask",

        "receptor_pdb", "pose_pdbqt",
        "mmgbsa_workdir", "mmgbsa_final", "mmgbsa_log", "mmgbsa_error",
    ]

    _write_jsonl(out_jsonl, merged)
    _write_csv(out_csv, merged, fields)
    write_mmgbsa_back_to_cache(run_dir, merged)

    n_ok = sum(1 for r in merged if r.get("mmgbsa_ok"))
    n_fail = len(merged) - n_ok

    print(f"[mmgbsa] wrote: {out_jsonl}")
    print(f"[mmgbsa] wrote: {out_csv}")
    print(f"[mmgbsa] summary: {n_ok} succeeded, {n_fail} failed, total={len(merged)}")

    if merged:
        best = merged[0]
        if best.get("mmgbsa_ok"):
            print(f"[mmgbsa] best dg_bind={best.get('mmgbsa_dg_bind'):.2f} "
                  f"seq={best.get('seq','')} charge_source={best.get('mmgbsa_charge_source','')}")
        else:
            print("[mmgbsa] WARNING: All jobs failed. Check antechamber.log and tleap.log")
            first_err = next((r for r in merged if r.get("mmgbsa_error")), None)
            if first_err:
                print(f"[mmgbsa] First error: {str(first_err.get('mmgbsa_error',''))[:200]}")
        print(f"[mmgbsa] best workdir={best.get('mmgbsa_workdir','')}")


if __name__ == "__main__":
    main()
