from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from structure.vienna import fold_sequence_to_ss, ViennaSSResult


@dataclass
class Fold2DResult:
    dot_bracket: str
    mfe_kcal_mol: float
    ss_path: str


@dataclass
class Fold3DResult:
    all_atom_pdbs: List[str]          # e.g. [.../aptamer-000001_AA.pdb, .../aptamer-000002_AA.pdb]
    all_atom_pdb: Optional[str]       # first frame AA pdb (for legacy callers)
    trafl_path: str                   # .../aptamer.trafl
    pdb0_path: str                    # .../aptamer-000001.pdb (reduced rep)
    n_frames_trafl: int
    exported_frames: List[int]
    workdir: str


def _which_or_raise(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FileNotFoundError(f"Cannot find executable in PATH: {name}")
    return p


def _guess_simrna_data_dir() -> Optional[Path]:
    # Priority: explicit env
    for k in ("SIMRNA_DATA", "SIMRNA_HOME"):
        v = os.environ.get(k, "").strip()
        if v:
            p = Path(v)
            if k == "SIMRNA_HOME":
                cand = p / "data"
                if cand.exists():
                    return cand
            if p.exists() and p.is_dir() and (p / "dist_PC.data.new_hist").exists():
                return p

    # Try infer from tool location
    tool = shutil.which("SimRNA_trafl2pdbs") or shutil.which("SimRNA")
    if tool:
        base = Path(tool).resolve().parent
        for cand in (base / "data", base.parent / "data"):
            if cand.exists() and (cand / "dist_PC.data.new_hist").exists():
                return cand
    return None


def _ensure_data_symlink(workdir: Path) -> None:
    data_link = workdir / "data"
    if data_link.exists():
        return
    data_dir = _guess_simrna_data_dir()
    if not data_dir:
        # 不硬编码你的 /home/maning/tools/...，让错误信息明确
        raise FileNotFoundError(
            "Cannot locate SimRNA data directory. Set SIMRNA_HOME or SIMRNA_DATA, "
            "or ensure SimRNA_trafl2pdbs is installed alongside a 'data/' directory."
        )
    data_link.symlink_to(data_dir, target_is_directory=True)


def _parse_frames(frames_to_export: str) -> List[int]:
    """
    Accept:
      "1-2", "1:2", "1,2,5-7", "1 2 3"
    Return sorted unique frames (1-based).
    """
    s = (frames_to_export or "").strip()
    if not s:
        return [1]

    s = s.replace(":", "-")
    parts = re.split(r"[,\s]+", s)
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a); b = int(b)
            if a <= 0 or b <= 0:
                continue
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            v = int(p)
            if v > 0:
                out.append(v)
    out = sorted(set(out))
    return out or [1]


def _count_frames_in_trafl(trafl_path: Path) -> int:
    """
    Heuristic: SimRNA trafl frame header lines often have 5 columns.
    Your previous quick check used exactly this and it matched '1 frame(s) were read'.
    """
    n = 0
    try:
        for line in trafl_path.read_text(errors="ignore").splitlines():
            if len(line.split()) == 5:
                n += 1
    except FileNotFoundError:
        return 0
    return n


def _write_simrna_config(cfg_path: Path, *, n_iterations: int, tra_write_every: int) -> None:
    """
    Minimal config ensuring multi-frame trajectory:
      NUMBER_OF_ITERATIONS
      TRA_WRITE_IN_EVERY_N_ITERATIONS

    The manual documents TRA_WRITE controlling how often frames are written and
    approx frame count ~ iters/TRA_WRITE + 1. :contentReference[oaicite:1]{index=1}
    """
    txt = "\n".join(
        [
            f"NUMBER_OF_ITERATIONS {int(n_iterations)}",
            f"TRA_WRITE_IN_EVERY_N_ITERATIONS {int(tra_write_every)}",
            # Keep temperature constant by default (fast smoke runs)
            "INIT_TEMP 1.35",
            "FINAL_TEMP 1.35",
            "",
        ]
    )
    cfg_path.write_text(txt)


def run_simrna(
    *,
    workdir: str,
    out_prefix: str,
    seq_path: str,
    ss_path: Optional[str],
    n_iterations: int,
    tra_write_every: int,
    seed: int,
    omp_threads: int,
) -> Tuple[Path, Path]:
    """
    Run SimRNA to produce:
      <out_prefix>.trafl  (usually aptamer.trafl)
      <out_prefix>-000001.pdb  (usually aptamer-000001.pdb)
    """
    simrna_bin = _which_or_raise("SimRNA")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    _ensure_data_symlink(work)

    cfg_path = work / "simrna_config.dat"
    _write_simrna_config(cfg_path, n_iterations=n_iterations, tra_write_every=tra_write_every)

    cmd = [simrna_bin, "-s", str(seq_path), "-c", str(cfg_path), "-o", str(work / out_prefix), "-R", str(int(seed))]
    if ss_path:
        cmd += ["-S", str(ss_path)]

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(max(1, omp_threads)))

    p = subprocess.run(
        cmd,
        cwd=str(work),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    out = (p.stdout or b"").decode("utf-8", errors="ignore")
    err = (p.stderr or b"").decode("utf-8", errors="ignore")
    if p.returncode != 0:
        raise RuntimeError(f"SimRNA failed (code={p.returncode}).\nSTDERR:\n{err}\nSTDOUT:\n{out}")

    # Expected outputs:
    trafl = work / f"{out_prefix}.trafl"
    pdb0 = work / f"{out_prefix}-000001.pdb"
    if not trafl.exists():
        raise FileNotFoundError(f"SimRNA did not produce trafl: {trafl}")
    if not pdb0.exists():
        raise FileNotFoundError(f"SimRNA did not produce initial pdb: {pdb0}")

    return trafl, pdb0


def export_frames_all_atom(
    *,
    workdir: str,
    pdb0_path: str,
    trafl_path: str,
    frames: List[int],
    aa_tag: str = "AA",
) -> List[Path]:
    """
    Use SimRNA_trafl2pdbs to export selected frames to PDB (all-atom).
    Manual shows usage with frames list and optional AA. 
    """
    tool = _which_or_raise("SimRNA_trafl2pdbs")
    work = Path(workdir)

    # SimRNA_trafl2pdbs syntax wants frame numbers/ranges as separate args:
    #   SimRNA_trafl2pdbs input.pdb traj.trafl 1 2 AA
    # (Do NOT put "..." placeholders; and "1:2" is not necessary.)
    frame_args = [str(x) for x in frames]

    cmd = [tool, str(pdb0_path), str(trafl_path)] + frame_args + [aa_tag]
    p = subprocess.run(
        cmd,
        cwd=str(work),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    out = (p.stdout or b"").decode("utf-8", errors="ignore")
    err = (p.stderr or b"").decode("utf-8", errors="ignore")
    if p.returncode != 0:
        raise RuntimeError(f"SimRNA_trafl2pdbs failed (code={p.returncode}).\nSTDERR:\n{err}\nSTDOUT:\n{out}")

    # Collect outputs: <prefix>-000001_AA.pdb, <prefix>-000002_AA.pdb, ...
    # Prefix is inferred from input pdb name (e.g. aptamer-000001.pdb -> aptamer-000001_AA.pdb etc.)
    aa_files = sorted(work.glob(f"*-*_AA.pdb")) + sorted(work.glob(f"*_AA.pdb"))
    # Filter strictly to the requested frames when possible.
    wanted = set(frames)
    picked: List[Path] = []
    for f in aa_files:
        m = re.search(r"-(\d{6})_AA\.pdb$", f.name)
        if m:
            fr = int(m.group(1))
            if fr in wanted:
                picked.append(f)
    # If parsing fails, fallback to whatever exists.
    if not picked:
        picked = aa_files
    return sorted(set(picked))


def fold_sequence_to_pdb_with_vienna(
    *,
    seq: Optional[str] = None,
    sequence: Optional[str] = None,
    workdir: str,
    out_prefix: str = "aptamer",
    n_iterations: int = 2000,
    frames_to_export: str = "1",
    seed: int = 1,
    omp_threads: int = 1,
    **_kwargs,
) -> Tuple[Fold2DResult, Fold3DResult]:
    """
    This is the ONE function ce_design.py calls.

    It must accept keywords:
      seq, workdir, out_prefix, omp_threads, n_iterations, frames_to_export, seed
    and return (res2d, res3d) with attributes used downstream.
    """
    s = (seq or sequence or "").strip().upper().replace("T", "U")
    if not s:
        raise ValueError("fold_sequence_to_pdb_with_vienna: empty sequence")

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    # 2D fold via Vienna
    ss: ViennaSSResult = fold_sequence_to_ss(s)
    ss_path = work / f"{out_prefix}.ss"
    ss_path.write_text(ss.dot_bracket + "\n")

    seq_path = work / f"{out_prefix}.seq"
    seq_path.write_text(s + "\n")

    # Decide how many frames we need
    frames = _parse_frames(frames_to_export)
    max_frame = max(frames) if frames else 1

    # Ensure trafl contains at least max_frame frames:
    # frames ~ iters/tra_write + 1  => choose tra_write so frames >= max_frame.
    if max_frame <= 1:
        tra_write = max(1, int(n_iterations))
    else:
        tra_write = max(1, int(n_iterations // (max_frame - 1)))
        # If rounding caused too few frames, reduce tra_write until enough
        while (n_iterations // tra_write + 1) < max_frame and tra_write > 1:
            tra_write -= 1

    trafl_path, pdb0_path = run_simrna(
        workdir=str(work),
        out_prefix=out_prefix,
        seq_path=str(seq_path),
        ss_path=str(ss_path),
        n_iterations=int(n_iterations),
        tra_write_every=int(tra_write),
        seed=int(seed),
        omp_threads=int(omp_threads),
    )

    n_frames = _count_frames_in_trafl(trafl_path)

    aa_pdbs = export_frames_all_atom(
        workdir=str(work),
        pdb0_path=str(pdb0_path),
        trafl_path=str(trafl_path),
        frames=frames,
        aa_tag="AA",
    )

    res2d = Fold2DResult(dot_bracket=ss.dot_bracket, mfe_kcal_mol=ss.mfe_kcal_mol, ss_path=str(ss_path))
    res3d = Fold3DResult(
        all_atom_pdbs=[str(p) for p in aa_pdbs],
        all_atom_pdb=str(aa_pdbs[0]) if aa_pdbs else None,
        trafl_path=str(trafl_path),
        pdb0_path=str(pdb0_path),
        n_frames_trafl=int(n_frames),
        exported_frames=frames,
        workdir=str(work),
    )
    return res2d, res3d
