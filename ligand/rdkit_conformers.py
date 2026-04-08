# ligand/rdkit_conformers.py
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors


@dataclass
class LigandConformerConfig:
    n_confs: int = 20
    seed: int = 123
    prune_rmsd: float = 0.5
    forcefield: str = "MMFF"   # "MMFF" or "UFF"
    optimize: bool = True
    max_opt_iters: int = 200
    max_embed_attempts: int = 50
    # Charge settings
    charge_override: Optional[int] = None  # If set, use this instead of auto-detect
    validate_charge: bool = True


@dataclass
class LigandChargeInfo:
    """Validated charge information for a ligand."""
    formal_charge: int
    detection_method: str  # "rdkit_formal", "override", "atom_sum", "gasteiger_sum", "fallback"
    n_atoms: int
    n_heavy_atoms: int
    molecular_formula: str
    molecular_weight: float
    smiles: str
    validation_passed: bool
    validation_notes: str


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


def _detect_formal_charge(mol: Chem.Mol) -> Tuple[int, str]:
    """
    Detect formal charge of molecule using multiple methods.
    
    Returns:
        (charge, method): The detected charge and method used.
    """
    # Method 1: RDKit GetFormalCharge (most reliable)
    try:
        formal_charge = Chem.GetFormalCharge(mol)
        return formal_charge, "rdkit_formal"
    except Exception:
        pass
    
    # Method 2: Sum of atomic formal charges
    try:
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        return charge, "atom_sum"
    except Exception:
        pass
    
    # Method 3: Gasteiger charges (last resort, less accurate for formal charge)
    try:
        AllChem.ComputeGasteigerCharges(mol)
        total = 0.0
        valid = True
        for atom in mol.GetAtoms():
            gc = atom.GetDoubleProp("_GasteigerCharge")
            if gc != gc:  # NaN check
                valid = False
                break
            total += gc
        if valid:
            return int(round(total)), "gasteiger_sum"
    except Exception:
        pass
    
    # Fallback: assume neutral
    return 0, "fallback"


def _validate_charge(mol: Chem.Mol, detected_charge: int) -> Tuple[bool, str]:
    """
    Validate that detected charge is chemically reasonable.
    
    Returns:
        (is_valid, notes): Validation result and explanatory notes.
    """
    notes_parts = []
    is_valid = True
    
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol.GetNumHeavyAtoms()
    
    # Check for extreme charges relative to molecule size
    if abs(detected_charge) > n_heavy:
        notes_parts.append(f"WARNING: |charge|={abs(detected_charge)} > n_heavy_atoms={n_heavy}")
        is_valid = False
    
    # Check for very high charges on small molecules
    if n_heavy < 50 and abs(detected_charge) > 6:
        notes_parts.append(f"WARNING: high charge ({detected_charge}) for small molecule (n_heavy={n_heavy})")
    
    # Try to verify with ionizable group patterns
    try:
        # Common anionic groups
        anionic_smarts = [
            "[CX3](=O)[OX1H0-]",      # Carboxylate
            "[SX4](=O)(=O)[OX1H0-]",  # Sulfonate
            "[PX4](=O)([OX1H0-])[OX1H0-]",  # Phosphate (2-)
            "[PX4](=O)([OX1H0-])[OX2]",     # Phosphate (1-)
            "[OX1H0-]",               # Generic O-
        ]
        # Common cationic groups
        cationic_smarts = [
            "[NX4+]",    # Quaternary ammonium
            "[nX3+]",    # Aromatic N+
            "[NH3+]",    # Primary amine protonated
        ]
        
        n_negative = 0
        n_positive = 0
        
        for smarts in anionic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                n_negative += len(matches)
        
        for smarts in cationic_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                n_positive += len(matches)
        
        # Expected charge range based on ionizable groups
        expected_min = -n_negative - 1
        expected_max = n_positive + 1
        
        if not (expected_min <= detected_charge <= expected_max):
            notes_parts.append(
                f"NOTE: charge ({detected_charge}) outside expected range "
                f"[{expected_min}, {expected_max}] from ionizable groups"
            )
    except Exception as e:
        notes_parts.append(f"NOTE: ionizable group check skipped: {str(e)[:40]}")
    
    notes = "; ".join(notes_parts) if notes_parts else "OK"
    return is_valid, notes


def compute_ligand_charge_info(
    mol: Chem.Mol,
    charge_override: Optional[int] = None,
    validate: bool = True,
) -> LigandChargeInfo:
    """
    Compute and validate ligand charge information from RDKit Mol object.
    
    Args:
        mol: RDKit Mol object (should have explicit Hs).
        charge_override: If provided, use this charge instead of auto-detection.
        validate: Whether to run validation checks.
    
    Returns:
        LigandChargeInfo with validated charge and molecular properties.
    """
    # Detect or use override charge
    if charge_override is not None:
        formal_charge = int(charge_override)
        method = "override"
    else:
        formal_charge, method = _detect_formal_charge(mol)
    
    # Validate if requested
    if validate:
        is_valid, notes = _validate_charge(mol, formal_charge)
    else:
        is_valid = True
        notes = "validation_skipped"
    
    # Compute molecular properties
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        smiles = ""
    
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = ""
    
    try:
        mw = Descriptors.MolWt(mol)
    except Exception:
        mw = 0.0
    
    return LigandChargeInfo(
        formal_charge=formal_charge,
        detection_method=method,
        n_atoms=mol.GetNumAtoms(),
        n_heavy_atoms=mol.GetNumHeavyAtoms(),
        molecular_formula=formula,
        molecular_weight=mw,
        smiles=smiles,
        validation_passed=is_valid,
        validation_notes=notes,
    )


def _embed_conformers(mol: Chem.Mol, cfg: LigandConformerConfig) -> List[int]:
    # Use ETKDG parameters
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


def _write_conformer_with_charge(
    mol: Chem.Mol,
    conf_id: int,
    out_path: Path,
    charge_info: LigandChargeInfo,
) -> None:
    """Write single conformer to SDF with charge info embedded as properties."""
    # Set properties on molecule for SDF output
    mol.SetProp("_Name", out_path.stem)
    mol.SetIntProp("NET_CHARGE", charge_info.formal_charge)
    mol.SetProp("CHARGE_METHOD", charge_info.detection_method)
    mol.SetProp("MOLECULAR_FORMULA", charge_info.molecular_formula)
    mol.SetDoubleProp("MOLECULAR_WEIGHT", charge_info.molecular_weight)
    mol.SetProp("CHARGE_VALIDATION", "PASSED" if charge_info.validation_passed else "FAILED")
    mol.SetProp("CHARGE_NOTES", charge_info.validation_notes)
    
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol, confId=conf_id)
    writer.close()


def prepare_ligand_conformers(
    ligand_file: Path,
    out_dir: Path,
    cfg: LigandConformerConfig,
) -> Tuple[List[Path], Path]:
    """
    Generate conformers and write each as separate SDF with embedded charge info.
    
    Output files:
      out_dir/conf_001.sdf, conf_002.sdf, ...
      out_dir/manifest.json (includes charge_info)
      out_dir/charge.json (standalone charge info)
      out_dir/charge.txt (simple charge value)
    
    Args:
        ligand_file: Input ligand file (SDF, MOL2, PDB, etc.)
        out_dir: Output directory for conformers
        cfg: Configuration for conformer generation
    
    Returns:
        (sdf_paths, manifest_path): List of conformer SDF paths and manifest path.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = out_dir / "manifest.json"

    # Load molecule
    mol = _load_ligand_any(Path(ligand_file), out_dir)
    
    # Compute and validate charge BEFORE conformer generation
    charge_info = compute_ligand_charge_info(
        mol=mol,
        charge_override=cfg.charge_override,
        validate=cfg.validate_charge,
    )
    
    # Log charge detection
    print(f"[ligand] Charge detected: {charge_info.formal_charge} (method={charge_info.detection_method})")
    if not charge_info.validation_passed:
        print(f"[ligand] WARNING: Charge validation failed: {charge_info.validation_notes}")
    
    # Generate conformers
    conf_ids = _embed_conformers(mol, cfg)
    energies = _optimize_conformers(mol, conf_ids, cfg)

    # Sort conformers by energy (if available)
    def _key(t):
        cid, e = t
        return (1e30 if (e != e) else e)  # NaN -> large

    energies_sorted = sorted(energies, key=_key)

    sdf_paths: List[Path] = []
    manifest_items: List[Dict[str, Any]] = []

    for i, (cid, e) in enumerate(energies_sorted, start=1):
        p = out_dir / f"conf_{i:03d}.sdf"
        
        # Write conformer with embedded charge properties
        _write_conformer_with_charge(mol, int(cid), p, charge_info)

        sdf_paths.append(p)
        manifest_items.append({
            "idx": i,
            "conf_id": int(cid),
            "energy": None if (e != e) else float(e),
            "path": str(p),
        })

    # Build manifest with charge_info included
    manifest: Dict[str, Any] = {
        "input": str(Path(ligand_file).expanduser().resolve()),
        "n_confs_requested": int(cfg.n_confs),
        "n_confs_written": int(len(sdf_paths)),
        "seed": int(cfg.seed),
        "prune_rmsd": float(cfg.prune_rmsd),
        "forcefield": str(cfg.forcefield),
        "optimize": bool(cfg.optimize),
        # NEW: Include charge information
        "charge_info": asdict(charge_info),
        "items": manifest_items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    
    # Write standalone charge files for easy access by downstream tools
    charge_json_path = out_dir / "charge.json"
    charge_json_path.write_text(json.dumps(asdict(charge_info), indent=2, ensure_ascii=False))
    
    charge_txt_path = out_dir / "charge.txt"
    charge_txt_path.write_text(f"{charge_info.formal_charge}\n")
    
    print(f"[ligand] Wrote {len(sdf_paths)} conformers to {out_dir}")
    print(f"[ligand] Charge info: {charge_json_path}")

    return sdf_paths, manifest_path


# ============================================================
# Utility functions for downstream tools (e.g., rescore_mmgbsa)
# ============================================================

def get_ligand_charge_from_manifest(manifest_path: Path) -> int:
    """
    Read validated ligand charge from conformer manifest.
    
    This is the RECOMMENDED way for downstream tools to get the charge.
    
    Args:
        manifest_path: Path to manifest.json from prepare_ligand_conformers.
    
    Returns:
        Net formal charge of the ligand.
    
    Raises:
        FileNotFoundError: If manifest doesn't exist.
        KeyError: If charge info not in manifest.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    manifest = json.loads(manifest_path.read_text())
    charge_info = manifest.get("charge_info", {})
    
    if "formal_charge" not in charge_info:
        raise KeyError(f"No charge_info.formal_charge in manifest: {manifest_path}")
    
    return int(charge_info["formal_charge"])


def get_ligand_charge_from_sdf(sdf_path: Path) -> Optional[int]:
    """
    Read ligand charge from SDF NET_CHARGE property.
    
    Args:
        sdf_path: Path to conformer SDF file (written by this module).
    
    Returns:
        Net charge if found, None otherwise.
    """
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        if suppl and len(suppl) > 0:
            mol = suppl[0]
            if mol and mol.HasProp("NET_CHARGE"):
                return mol.GetIntProp("NET_CHARGE")
    except Exception:
        pass
    return None


def get_ligand_charge_from_charge_txt(run_dir: Path) -> Optional[int]:
    """
    Read charge from ligand_ensemble/charge.txt (simplest access).
    
    Args:
        run_dir: Run directory containing ligand_ensemble/.
    
    Returns:
        Net charge if found, None otherwise.
    """
    charge_txt = Path(run_dir) / "ligand_ensemble" / "charge.txt"
    if charge_txt.exists():
        try:
            return int(charge_txt.read_text().strip())
        except Exception:
            pass
    return None


def get_ligand_charge_auto(
    run_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    sdf_path: Optional[Path] = None,
    ligand_file: Optional[Path] = None,
    fallback: int = 0,
) -> Tuple[int, str]:
    """
    Automatically determine ligand charge from available sources.
    
    Priority order:
    1. manifest.json in ligand_ensemble/ (most reliable, includes validation)
    2. charge.txt in ligand_ensemble/ (simple)
    3. NET_CHARGE property in SDF
    4. Direct RDKit detection from ligand_file
    5. Fallback value
    
    Args:
        run_dir: Run directory containing ligand_ensemble/.
        manifest_path: Direct path to manifest.json.
        sdf_path: Path to conformer SDF with embedded charge.
        ligand_file: Original ligand file for direct detection.
        fallback: Value to use if all detection fails.
    
    Returns:
        (charge, source): The charge and where it came from.
    """
    # Priority 1: Manifest from run_dir
    if run_dir:
        run_dir = Path(run_dir)
        manifest_in_run = run_dir / "ligand_ensemble" / "manifest.json"
        if manifest_in_run.exists():
            try:
                charge = get_ligand_charge_from_manifest(manifest_in_run)
                return charge, "manifest"
            except Exception:
                pass
    
    # Priority 1b: Direct manifest path
    if manifest_path and Path(manifest_path).exists():
        try:
            charge = get_ligand_charge_from_manifest(Path(manifest_path))
            return charge, "manifest"
        except Exception:
            pass
    
    # Priority 2: charge.txt
    if run_dir:
        charge = get_ligand_charge_from_charge_txt(run_dir)
        if charge is not None:
            return charge, "charge_txt"
    
    # Priority 3: SDF property
    if sdf_path and Path(sdf_path).exists():
        charge = get_ligand_charge_from_sdf(Path(sdf_path))
        if charge is not None:
            return charge, "sdf_property"
    
    # Priority 4: Direct detection from ligand file
    if ligand_file and Path(ligand_file).exists():
        try:
            # Need a temp workdir for potential obabel conversion
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                mol = _load_ligand_any(Path(ligand_file), Path(tmpdir))
                info = compute_ligand_charge_info(mol)
                return info.formal_charge, f"rdkit_{info.detection_method}"
        except Exception:
            pass
    
    # Fallback
    return fallback, "fallback"


# ============================================================
# Command-line interface
# ============================================================

def main():
    """Command-line interface for ligand preparation with charge detection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare ligand conformer ensemble with charge validation"
    )
    parser.add_argument("ligand", type=str, help="Input ligand file (SDF, MOL2, PDB)")
    parser.add_argument("--out_dir", "-o", type=str, default="ligand_ensemble",
                        help="Output directory (default: ligand_ensemble)")
    parser.add_argument("--n_confs", "-n", type=int, default=20,
                        help="Number of conformers to generate (default: 20)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--prune_rmsd", type=float, default=0.5,
                        help="RMSD threshold for conformer pruning")
    parser.add_argument("--forcefield", type=str, default="MMFF",
                        choices=["MMFF", "UFF"], help="Force field for optimization")
    parser.add_argument("--no_optimize", action="store_true",
                        help="Skip geometry optimization")
    parser.add_argument("--charge", type=int, default=None,
                        help="Override net charge (auto-detect if not set)")
    parser.add_argument("--charge_only", action="store_true",
                        help="Only compute and print charge, don't generate conformers")
    
    args = parser.parse_args()
    
    ligand_path = Path(args.ligand)
    
    if args.charge_only:
        # Just compute and print charge
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mol = _load_ligand_any(ligand_path, Path(tmpdir))
            info = compute_ligand_charge_info(mol, charge_override=args.charge)
        
        print(f"Formal charge:    {info.formal_charge}")
        print(f"Detection method: {info.detection_method}")
        print(f"Validation:       {'PASSED' if info.validation_passed else 'FAILED'}")
        print(f"Notes:            {info.validation_notes}")
        print(f"Formula:          {info.molecular_formula}")
        print(f"MW:               {info.molecular_weight:.2f}")
        print(f"Heavy atoms:      {info.n_heavy_atoms}")
        print(f"SMILES:           {info.smiles}")
        return
    
    cfg = LigandConformerConfig(
        n_confs=args.n_confs,
        seed=args.seed,
        prune_rmsd=args.prune_rmsd,
        forcefield=args.forcefield,
        optimize=not args.no_optimize,
        charge_override=args.charge,
    )
    
    conf_files, manifest = prepare_ligand_conformers(
        ligand_file=ligand_path,
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )
    
    print(f"\n=== Summary ===")
    print(f"Generated {len(conf_files)} conformers")
    print(f"Manifest:    {manifest}")
    print(f"Charge file: {Path(args.out_dir) / 'charge.json'}")


if __name__ == "__main__":
    main()
