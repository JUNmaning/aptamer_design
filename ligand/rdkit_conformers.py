# ligand/rdkit_conformers.py
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class LigandConformerConfig:
    n_confs: int = 20
    seed: int = 123
    prune_rmsd: float = 0.5
    forcefield: str = "MMFF"   # "MMFF" or "UFF"
    optimize: bool = True
    max_opt_iters: int = 200
    max_embed_attempts: int = 50


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


def _load_ligand_any(path: Path, workdir: Path) -> Chem.Mol:
    """
    Try RDKit native read first. If PDB parse fails, fallback:
      obabel ligand.pdb -> tmp.sdf -> RDKit SDMolSupplier
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ligand not found: {path}")

    ext = path.suffix.lower()

    mol = None
    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
    elif ext in [".mol", ".mol2"]:
        mol = Chem.MolFromMolFile(str(path), removeHs=False) if ext == ".mol" else Chem.MolFromMol2File(str(path), removeHs=False)
    elif ext == ".pdb":
        # try sanitize False first (more tolerant)
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                # still usable for embedding in many cases
                pass

        if mol is None:
            # fallback: use obabel to convert to sdf
            _which("obabel")
            workdir.mkdir(parents=True, exist_ok=True)
            tmp_sdf = workdir / "ligand_obabel_tmp.sdf"
            log = _run(["obabel", "-ipdb", str(path), "-osdf", "-O", str(tmp_sdf), "--gen3d"])
            (workdir / "ligand_obabel_convert.log").write_text(log)

            suppl = Chem.SDMolSupplier(str(tmp_sdf), removeHs=False)
            mol = suppl[0] if suppl and len(suppl) > 0 else None
    else:
        # last resort: try RDKit from SDMolSupplier
        suppl = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None

    if mol is None:
        raise ValueError(f"Failed to read ligand: {path}")

    # ensure explicit Hs
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def _embed_conformers(mol: Chem.Mol, cfg: LigandConformerConfig) -> List[int]:
    # Use ETKDG parameters, but do NOT set params.maxAttempts (RDKit version dependent).
    params = AllChem.ETKDGv3()
    params.randomSeed = int(cfg.seed)
    params.pruneRmsThresh = float(cfg.prune_rmsd)
    params.numThreads = 0

    # Try embed with params, fallback to legacy signature if needed
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(cfg.n_confs), params=params))
    except TypeError:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=int(cfg.n_confs),
                randomSeed=int(cfg.seed),
                pruneRmsThresh=float(cfg.prune_rmsd),
                maxAttempts=int(cfg.max_embed_attempts),
                numThreads=0,
            )
        )

    if not conf_ids:
        raise RuntimeError("RDKit failed to embed conformers (0 conformers generated).")
    return conf_ids


def _optimize_conformers(mol: Chem.Mol, conf_ids: List[int], cfg: LigandConformerConfig) -> List[Tuple[int, float]]:
    energies: List[Tuple[int, float]] = []

    if not cfg.optimize:
        for cid in conf_ids:
            energies.append((cid, float("nan")))
        return energies

    ff = cfg.forcefield.upper()
    if ff == "MMFF":
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        for cid in conf_ids:
            ffobj = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            ffobj.Initialize()
            ffobj.Minimize(maxIts=int(cfg.max_opt_iters))
            e = ffobj.CalcEnergy()
            energies.append((cid, float(e)))
    else:
        # UFF
        for cid in conf_ids:
            ffobj = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ffobj.Initialize()
            ffobj.Minimize(maxIts=int(cfg.max_opt_iters))
            e = ffobj.CalcEnergy()
            energies.append((cid, float(e)))

    return energies


def prepare_ligand_conformers(ligand_file: Path, out_dir: Path, cfg: LigandConformerConfig) -> Tuple[List[Path], Dict]:
    """
    Generate conformers and write each as separate SDF:
      out_dir/conf_001.sdf, conf_002.sdf, ...
    Also write manifest.json.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mol = _load_ligand_any(Path(ligand_file), out_dir)
    conf_ids = _embed_conformers(mol, cfg)
    energies = _optimize_conformers(mol, conf_ids, cfg)

    # Sort conformers by energy (if available)
    def _key(t):
        cid, e = t
        return (1e30 if (e != e) else e)  # NaN -> large

    energies_sorted = sorted(energies, key=_key)

    sdf_paths: List[Path] = []
    manifest_items: List[Dict] = []

    for i, (cid, e) in enumerate(energies_sorted, start=1):
        p = out_dir / f"conf_{i:03d}.sdf"
        w = Chem.SDWriter(str(p))
        w.write(mol, confId=int(cid))
        w.close()

        sdf_paths.append(p)
        manifest_items.append(
            {"idx": i, "conf_id": int(cid), "energy": None if (e != e) else float(e), "path": str(p)}
        )

    manifest = {
        "input": str(Path(ligand_file).expanduser().resolve()),
        "n_confs_requested": int(cfg.n_confs),
        "n_confs_written": int(len(sdf_paths)),
        "seed": int(cfg.seed),
        "prune_rmsd": float(cfg.prune_rmsd),
        "forcefield": str(cfg.forcefield),
        "optimize": bool(cfg.optimize),
        "items": manifest_items,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return sdf_paths, manifest
