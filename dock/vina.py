# dock/vina.py
from __future__ import annotations

import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class DockResult:
    affinity_kcal: float
    out_pdbqt: Path
    log_file: Path
    receptor_pdbqt: Path
    ligand_pdbqt: Path
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]


def _which(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Executable not found in PATH: {name}")
    return p


def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{out}")
    return out


def _parse_pdb_coords(pdb: Path) -> List[Tuple[float, float, float]]:
    coords = []
    with pdb.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDB fixed columns: x[30:38], y[38:46], z[46:54]
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append((x, y, z))
                except Exception:
                    continue
    return coords


def _compute_box_from_receptor_pdb(receptor_pdb: Path, padding: float, min_size: float = 20.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    coords = _parse_pdb_coords(receptor_pdb)
    if not coords:
        # fallback: origin box
        center = (0.0, 0.0, 0.0)
        size = (min_size, min_size, min_size)
        return center, size

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    sx = max(min_size, (xmax - xmin) + 2.0 * padding)
    sy = max(min_size, (ymax - ymin) + 2.0 * padding)
    sz = max(min_size, (zmax - zmin) + 2.0 * padding)

    return (cx, cy, cz), (sx, sy, sz)


def _obabel_to_pdbqt(
    inp: Path,
    out: Path,
    inp_fmt: Optional[str] = None,
    add_h: bool = True,
    gasteiger: bool = True,
    cwd: Optional[Path] = None,
) -> str:
    _which("obabel")
    out.parent.mkdir(parents=True, exist_ok=True)

    # infer input format
    ext = inp.suffix.lower().lstrip(".")
    if inp_fmt is None:
        inp_fmt = ext if ext else "pdb"

    cmd = ["obabel", f"-i{inp_fmt}", str(inp), "-opdbqt", "-O", str(out)]
    if add_h:
        cmd.append("-h")
    if gasteiger:
        # OpenBabel widely supports this option
        cmd += ["--partialcharge", "gasteiger"]

    log = _run(cmd, cwd=cwd)

    # strict check: must exist and not empty
    if (not out.exists()) or out.stat().st_size < 10:
        raise RuntimeError(
            f"OpenBabel conversion produced no pdbqt: {out}\n"
            f"Input: {inp}\n"
            f"Output log:\n{log}"
        )
    return log


def _parse_vina_affinity(vina_log_text: str) -> float:
    """
    Parse best affinity from vina log.
    Looks for the table:
      1    -7.3 ...
    """
    lines = vina_log_text.splitlines()
    # find first pose line beginning with "1"
    for ln in lines:
        ln_strip = ln.strip()
        m = re.match(r"^1\s+(-?\d+(\.\d+)?)\s+", ln_strip)
        if m:
            return float(m.group(1))
    raise RuntimeError("Failed to parse affinity from vina log.")


def _sanitize_receptor_pdbqt(path: Path) -> None:
    """
    Receptors in Vina should be rigid PDBQT without ROOT/BRANCH/TORSDOF lines.
    """
    kept = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            tag = line[:6].strip()
            if tag in {"ATOM", "HETATM", "TER", "END"}:
                kept.append(line.rstrip("\n"))
    if not kept:
        raise RuntimeError(f"Sanitized receptor PDBQT is empty: {path}")
    path.write_text("\n".join(kept) + "\n")


def dock_one(
    name: str,
    receptor_pdb: Path,
    ligand_sdf: Path,
    workdir: Path,
    padding_angstrom: float = 10.0,
    exhaustiveness: int = 8,
    num_modes: int = 5,
    seed: int = 123,
    cpu: int = 1,
) -> DockResult:
    """
    Robust docking:
      receptor_pdb -> receptor.pdbqt (obabel)
      ligand_sdf (or pdb/sdf/mol2/...) -> ligand.pdbqt (obabel)
      run vina
    """
    _which("vina")

    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    ligand_sdf = Path(ligand_sdf).expanduser().resolve()

    if not receptor_pdb.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb}")
    if not ligand_sdf.exists():
        raise FileNotFoundError(f"Ligand file not found: {ligand_sdf}")

    receptor_pdbqt = workdir / f"{name}.receptor.pdbqt"
    ligand_pdbqt = workdir / "ligand.pdbqt"
    out_pdbqt = workdir / f"{name}.dock.pdbqt"
    log_file = workdir / f"{name}.vina.log"

    # 1) compute docking box from receptor PDB
    center, size = _compute_box_from_receptor_pdb(receptor_pdb, padding=padding_angstrom)

    # 2) convert receptor/ligand to pdbqt (strict)
    # write convert logs for debugging
    try:
        rec_log = _obabel_to_pdbqt(receptor_pdb, receptor_pdbqt, inp_fmt="pdb", cwd=workdir)
        _sanitize_receptor_pdbqt(receptor_pdbqt)
        (workdir / f"{name}.receptor_convert.log").write_text(rec_log)
    except Exception as e:
        raise RuntimeError(f"Receptor PDB->PDBQT failed: {receptor_pdb}\n{repr(e)}")
    # ligand input format inferred from suffix
    lig_ext = ligand_sdf.suffix.lower().lstrip(".")
    lig_fmt = lig_ext if lig_ext else "sdf"
    try:
        lig_log = _obabel_to_pdbqt(ligand_sdf, ligand_pdbqt, inp_fmt=lig_fmt, cwd=workdir)
        (workdir / f"{name}.ligand_convert.log").write_text(lig_log)
    except Exception as e:
        raise RuntimeError(f"Ligand -> PDBQT failed: {ligand_sdf}\n{repr(e)}")

    # 3) run vina
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{size[0]:.3f}",
        "--size_y", f"{size[1]:.3f}",
        "--size_z", f"{size[2]:.3f}",
        "--exhaustiveness", str(int(exhaustiveness)),
        "--num_modes", str(int(num_modes)),
        "--out", str(out_pdbqt),
        "--log", str(log_file),
        "--seed", str(int(seed)),
        "--cpu", str(int(cpu)),
    ]
    vina_out = _run(cmd, cwd=workdir)

    # vina writes the real score into log_file, read that for parsing
    if not log_file.exists():
        raise RuntimeError(f"Vina finished but no log file: {log_file}")

    log_txt = log_file.read_text(errors="ignore")
    affinity = _parse_vina_affinity(log_txt)

    return DockResult(
        affinity_kcal=float(affinity),
        out_pdbqt=out_pdbqt,
        log_file=log_file,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        size=size,
    )
