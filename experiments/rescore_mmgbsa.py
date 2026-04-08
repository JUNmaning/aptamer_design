#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
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
    """
    Optional mode: create a manifest directly from topN.csv.
    In the CE pipeline, the preferred mode is manifest.jsonl generated under rescore/iter_xxx/.
    """
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
    first_key = None  # (chain, resseq, icode)

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
# ligand name alignment: mol2 -> template pdb, then inject dock coords
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
    # Common cases: "Cl", "Br"
    if len(name) >= 2 and name[0].isalpha() and name[1].islower():
        return (name[0] + name[1]).upper()
    return name[0].upper()


def _is_hydrogen_line(ln: str) -> bool:
    el = _pdb_element(ln)
    if el == "H":
        return True
    an = ln[12:16].strip().upper()
    return an.startswith("H")


def _extract_xyz_from_pdb_atom_line(ln: str) -> Tuple[float, float, float]:
    x = float(ln[30:38])
    y = float(ln[38:46])
    z = float(ln[46:54])
    return x, y, z


def _format_xyz_into_pdb_atom_line(ln: str, x: float, y: float, z: float) -> str:
    if len(ln) < 80:
        ln = ln + (" " * (80 - len(ln)))
    return ln[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + ln[54:]


def _write_ligand_tpl_pdb_from_mol2(work: Path) -> Path:
    tpl = work / "ligand_tpl.pdb"
    _run_logged(
        ["antechamber", "-i", "ligand.mol2", "-fi", "mol2", "-o", str(tpl), "-fo", "pdb", "-s", "2"],
        cwd=work,
        log_path=work / "antechamber_mol2_to_pdb.log",
    )
    return tpl


def _inject_pose_coords_to_tpl(
    tpl_pdb: Path,
    pose_pdb: Path,
    out_pdb: Path,
) -> None:
    """
    Inject dock pose coordinates into the ligand template PDB (names/types from template are preserved).

    If atom counts mismatch (commonly due to missing H in pose), we inject only heavy-atom coordinates
    and keep hydrogen coordinates from the template.
    """
    tpl_lines = _pdb_atom_lines(tpl_pdb)
    pose_lines = _pdb_atom_lines(pose_pdb)

    if len(tpl_lines) == len(pose_lines):
        out_lines: List[str] = []
        for tln, pln in zip(tpl_lines, pose_lines):
            x, y, z = _extract_xyz_from_pdb_atom_line(pln)
            out_lines.append(_format_xyz_into_pdb_atom_line(tln, x, y, z).rstrip())
        out_pdb.write_text("\n".join(out_lines) + "\n", errors="ignore")
        return

    # Heavy-atom-only injection fallback
    tpl_heavy_idx = [i for i, ln in enumerate(tpl_lines) if not _is_hydrogen_line(ln)]
    pose_heavy_idx = [i for i, ln in enumerate(pose_lines) if not _is_hydrogen_line(ln)]

    if len(tpl_heavy_idx) != len(pose_heavy_idx):
        raise RuntimeError(
            f"Atom count mismatch when aligning ligand coordinates:\n"
            f"  tpl atoms={len(tpl_lines)} heavy={len(tpl_heavy_idx)} ({tpl_pdb})\n"
            f"  pose atoms={len(pose_lines)} heavy={len(pose_heavy_idx)} ({pose_pdb})\n"
            f"Cannot align coordinates reliably."
        )

    out_lines = tpl_lines[:]
    for ti, pi in zip(tpl_heavy_idx, pose_heavy_idx):
        x, y, z = _extract_xyz_from_pdb_atom_line(pose_lines[pi])
        out_lines[ti] = _format_xyz_into_pdb_atom_line(tpl_lines[ti], x, y, z)

    out_pdb.write_text("\n".join([ln.rstrip() for ln in out_lines]) + "\n", errors="ignore")


# -----------------------
# charge detection
# -----------------------
def _detect_net_charge_from_pdbqt(pdbqt: Path) -> Optional[int]:
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


# -----------------------
# parsing MMPBSA output
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
# MM/GBSA job
# -----------------------
@dataclass
class MMGBSAConfig:
    leaprc_nucleic: str = "leaprc.RNA.OL3"
    leaprc_smallmol: str = "leaprc.gaff2"
    at_forcefield: str = "gaff2"
    charge_method: str = "bcc"         # bcc or gas
    no_gas_fallback: bool = False
    igb: int = 5
    saltcon: float = 0.150
    ph: float = 7.4
    keep_tmp: bool = True


def _write_tleap_in(work: Path, cfg: MMGBSAConfig) -> Path:
    leap_in = work / "tleap.in"
    leap_in.write_text(
        "\n".join(
            [
                f"source {cfg.leaprc_nucleic}",
                f"source {cfg.leaprc_smallmol}",
                "",
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
        ),
        errors="ignore",
    )
    return leap_in


def _write_cpptraj_in(work: Path) -> Path:
    inp = work / "cpptraj.in"
    inp.write_text(
        "\n".join(
            [
                "parm complex.prmtop",
                "trajin complex.inpcrd 1 1",
                "trajout complex.mdcrd",
                "go",
                "quit",
                "",
            ]
        ),
        errors="ignore",
    )
    return inp


def _write_mmpbsa_in(work: Path, cfg: MMGBSAConfig) -> Path:
    inp = work / "mmpbsa.in"
    inp.write_text(
        "\n".join(
            [
                "&general",
                "  startframe=1, endframe=1, interval=1,",
                "  verbose=1,",
                "/",
                "&gb",
                f"  igb={int(cfg.igb)}, saltcon={float(cfg.saltcon):.3f},",
                "/",
                "",
            ]
        ),
        errors="ignore",
    )
    return inp


def _pick_job_paths(job: Dict) -> Tuple[Path, Path]:
    rec = job.get("best_receptor_pdb") or job.get("receptor_pdb") or ""
    pose = job.get("best_dock_pose_pdbqt") or job.get("pose_pdbqt") or ""
    if not rec or not pose:
        raise KeyError("Job missing receptor/pose path. Need receptor_pdb and pose_pdbqt (or best_* equivalents).")
    return Path(rec).expanduser().resolve(), Path(pose).expanduser().resolve()


def _pick_ligand_source(job: Dict) -> Optional[Path]:
    lig = (job.get("best_ligand_sdf") or job.get("ligand_sdf") or "").strip()
    if not lig:
        return None
    p = Path(lig).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    return p if p.exists() else None


def run_one_mmgbsa(job: Dict, out_dir: Path, cfg: MMGBSAConfig, net_charge_override: Optional[int]) -> Dict:
    _require_exe("obabel")
    _require_exe("pdb4amber")
    _require_exe("antechamber")
    _require_exe("parmchk2")
    _require_exe("tleap")
    _require_exe("cpptraj")
    _require_exe("MMPBSA.py")

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
        cwd=work,
        log_path=work / "pdb4amber.log",
    )

    n_removed = _strip_5prime_phosphate(receptor_amber_tmp, receptor_amber_stripped)
    if n_removed > 0:
        shutil.copyfile(receptor_amber_stripped, receptor_amber_final)
        print(f"[mmgbsa] stripped 5' phosphate atoms: removed={n_removed} -> {receptor_amber_final.name}")
    else:
        shutil.copyfile(receptor_amber_tmp, receptor_amber_final)

    # -----------------------
    # 2) ligand pose prep (pdbqt -> pdb, first pose)
    # -----------------------
    lig_pose_raw = work / "ligand_pose_raw.pdb"
    lig_pose_sane = work / "ligand_pose.pdb"
    lig_pose_h = work / "ligand_pose_H.pdb"

    _run_logged(
        ["obabel", "-ipdbqt", str(pose_pdbqt), "-opdb", "-O", str(lig_pose_raw), "-f", "1", "-l", "1"],
        cwd=work,
        log_path=work / "obabel_pose_to_pdb.log",
    )
    _sanitize_ligand_pdb(lig_pose_raw, lig_pose_sane, resname="LIG", chain="Z")

    # Try to add H to the pose (may fail; heavy-atom-only injection is supported later)
    _run_logged(
        ["obabel", "-ipdb", str(lig_pose_sane), "-opdb", "-O", str(lig_pose_h), "-p", f"{cfg.ph:.2f}"],
        cwd=work,
        log_path=work / "obabel_addh.log",
        allow_fail=True,
    )
    if (not lig_pose_h.exists()) or lig_pose_h.stat().st_size == 0:
        shutil.copyfile(lig_pose_sane, lig_pose_h)

    # -----------------------
    # 3) ligand parameterization (IMPORTANT: use original ligand file if available)
    # -----------------------
    if net_charge_override is not None:
        net_charge = int(net_charge_override)
    else:
        net_charge = _detect_net_charge_from_pdbqt(pose_pdbqt)
        if net_charge is None:
            net_charge = 0

    lig_src = _pick_ligand_source(job)
    lig_input = work / "ligand_input.sdf"

    if lig_src is not None:
        # Convert to SDF with explicit hydrogens when possible
        # This preserves bond order better than PDB derived from docking.
        _run_logged(
            ["obabel", str(lig_src), "-O", str(lig_input), "-h"],
            cwd=work,
            log_path=work / "obabel_ligand_to_sdf.log",
            allow_fail=True,
        )
        if (not lig_input.exists()) or lig_input.stat().st_size == 0:
            # Fallback: just copy the source
            shutil.copyfile(lig_src, lig_input)
    else:
        # Fallback: use pose PDB (less reliable)
        shutil.copyfile(lig_pose_h, lig_input)

    lig_mol2 = work / "ligand.mol2"
    lig_frcmod = work / "ligand.frcmod"

    def _do_antechamber_from_sdf(method: str) -> None:
        _run_logged(
            [
                "antechamber",
                "-i", str(lig_input),
                "-fi", "sdf",
                "-o", str(lig_mol2),
                "-fo", "mol2",
                "-at", cfg.at_forcefield,
                "-c", method,
                "-nc", str(int(net_charge)),
                "-s", "2",
            ],
            cwd=work,
            log_path=work / "antechamber.log",
        )

    def _do_antechamber_from_pose_pdb(method: str) -> None:
        # Last-resort fallback; may fail for ligands requiring correct bond orders.
        _run_logged(
            [
                "antechamber",
                "-i", str(lig_pose_h),
                "-fi", "pdb",
                "-o", str(lig_mol2),
                "-fo", "mol2",
                "-at", cfg.at_forcefield,
                "-c", method,
                "-nc", str(int(net_charge)),
                "-s", "2",
            ],
            cwd=work,
            log_path=work / "antechamber.log",
        )

    try:
        if lig_src is not None:
            _do_antechamber_from_sdf(cfg.charge_method)
        else:
            _do_antechamber_from_pose_pdb(cfg.charge_method)
    except Exception as e1:
        if (cfg.charge_method == "bcc") and (not cfg.no_gas_fallback):
            (work / "antechamber_fallback_reason.txt").write_text(
                f"Primary antechamber failed with bcc. Error: {repr(e1)}\nTrying gas fallback.\n",
                errors="ignore",
            )
            if lig_src is not None:
                _do_antechamber_from_sdf("gas")
            else:
                _do_antechamber_from_pose_pdb("gas")
        else:
            raise

    _run_logged(
        ["parmchk2", "-i", str(lig_mol2), "-f", "mol2", "-o", str(lig_frcmod)],
        cwd=work,
        log_path=work / "parmchk2.log",
    )

    # -----------------------
    # 4) Build ligand pose with correct names (template from ligand.mol2 + inject dock coords)
    # -----------------------
    lig_tpl = _write_ligand_tpl_pdb_from_mol2(work)
    lig_pose_named = work / "ligand_pose_named.pdb"
    _inject_pose_coords_to_tpl(lig_tpl, lig_pose_h, lig_pose_named)

    # -----------------------
    # 5) complex.pdb uses receptor_amber_final + ligand_pose_named
    # -----------------------
    complex_pdb = work / "complex.pdb"
    _make_complex_pdb(receptor_amber_final, lig_pose_named, complex_pdb)

    # -----------------------
    # 6) tleap build prmtop/inpcrd
    # -----------------------
    leap_in = _write_tleap_in(work, cfg)
    _run_logged(["tleap", "-f", str(leap_in)], cwd=work, log_path=work / "tleap.log")

    for fn in ["receptor.prmtop", "ligand.prmtop", "complex.prmtop"]:
        ptop = work / fn
        if (not ptop.exists()) or ptop.stat().st_size < 1000:
            raise RuntimeError(f"tleap did not generate a valid {fn}. See {work/'tleap.log'}")

    # -----------------------
    # 7) cpptraj -> 1-frame mdcrd
    # -----------------------
    cpp_in = _write_cpptraj_in(work)
    _run_logged(["cpptraj", "-i", str(cpp_in)], cwd=work, log_path=work / "cpptraj.log")

    # -----------------------
    # 8) MMPBSA single-frame GBSA
    # -----------------------
    mmp_in = _write_mmpbsa_in(work, cfg)
    _run_logged(
        [
            "MMPBSA.py",
            "-O",
            "-i", str(mmp_in),
            "-cp", "complex.prmtop",
            "-rp", "receptor.prmtop",
            "-lp", "ligand.prmtop",
            "-y", "complex.mdcrd",
        ],
        cwd=work,
        log_path=work / "mmpbsa.log",
    )

    final_dat = work / "FINAL_RESULTS_MMPBSA.dat"
    dg = _parse_mmpbsa_final_results(final_dat)

    rec = dict(job)
    rec.update(
        {
            "mmgbsa_ok": bool(dg is not None),
            "mmgbsa_dg_bind": dg if dg is not None else 1e9,
            "mmgbsa_workdir": str(work),
            "mmgbsa_final": str(final_dat) if final_dat.exists() else "",
            "mmgbsa_log": str(work / "mmpbsa.log") if (work / "mmpbsa.log").exists() else "",
            "mmgbsa_charge": int(net_charge),
            "mmgbsa_charge_method": cfg.charge_method,
            "mmgbsa_igb": int(cfg.igb),
            "mmgbsa_saltcon": float(cfg.saltcon),
            "mmgbsa_ph": float(cfg.ph),
        }
    )
    if dg is None:
        rec["mmgbsa_ok"] = False
        rec["mmgbsa_error"] = f"parse_failed: {final_dat}" if final_dat.exists() else "missing_final_results"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--net_charge", type=int, default=None)
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    rescore_root = run_dir / "rescore"
    _safe_mkdir(rescore_root)

    # topN.csv -> manifest mode
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

    if int(args.workers) <= 1:
        for j in jobs:
            try:
                results.append(run_one_mmgbsa(j, out_dir, cfg, args.net_charge))
            except Exception as e:
                errors.append(_handle_err(j, e))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(run_one_mmgbsa, j, out_dir, cfg, args.net_charge): j for j in jobs}
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
        "mmgbsa_charge", "mmgbsa_charge_method",
        "mmgbsa_igb", "mmgbsa_saltcon", "mmgbsa_ph",
        "receptor_pdb", "pose_pdbqt",
        "mmgbsa_workdir", "mmgbsa_final", "mmgbsa_log", "mmgbsa_error",
    ]

    _write_jsonl(out_jsonl, merged)
    _write_csv(out_csv, merged, fields)

    print(f"[mmgbsa] wrote: {out_jsonl}")
    print(f"[mmgbsa] wrote: {out_csv}")
    if merged:
        best = merged[0]
        print(f"[mmgbsa] best dg_bind={best.get('mmgbsa_dg_bind')} seq={best.get('seq','')}")
        print(f"[mmgbsa] best workdir={best.get('mmgbsa_workdir','')}")


if __name__ == "__main__":
    main()
