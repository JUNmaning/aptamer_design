import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# -------------------------------
# 1. Read molecules (support PDB / SDF)
# -------------------------------

def load_molecule(path):
    """
    Load molecule from PDB or SDF using RDKit.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".sdf", ".mol"]:
        mol = Chem.MolFromMolFile(path, removeHs=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if mol is None:
        raise ValueError(f"RDKit failed to load molecule: {path}")

    return mol


# -------------------------------
# 2. Obtain atomic coordinates and charges
# -------------------------------

def get_atom_positions_and_charges(mol):
    """
    Extract atomic positions (Å) and Gasteiger charges.
    """

    # ① Calculate Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)

    conf = mol.GetConformer()

    positions = []
    charges = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        positions.append([pos.x, pos.y, pos.z])

        # ② Read partial charge
        q = atom.GetProp("_GasteigerCharge")
        q = float(q) if q != "nan" else 0.0
        charges.append(q)

    return np.array(positions), np.array(charges)


# -------------------------------
# 3. Construct 3D mesh
# -------------------------------

def build_grid(positions, spacing=1.0, padding=5.0):
    """
    Build a 3D grid around the molecule.
    """
    min_coords = positions.min(axis=0) - padding
    max_coords = positions.max(axis=0) + padding

    xs = np.arange(min_coords[0], max_coords[0], spacing)
    ys = np.arange(min_coords[1], max_coords[1], spacing)
    zs = np.arange(min_coords[2], max_coords[2], spacing)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid


# -------------------------------
# 4. Calculate electrostatic potential (simplified Coulomb)
# -------------------------------

def compute_electrostatic_field(
    mol_path,
    spacing=1.0,
    padding=5.0,
    epsilon=80.0
):
    """
    Compute electrostatic potential field on a 3D grid.

    V(r) = sum(q_i / (epsilon * r_i))
    """

    # ① load molecule
    mol = load_molecule(mol_path)

    # ② Obtain atomic information
    positions, charges = get_atom_positions_and_charges(mol)

    # ③ build grid
    grid = build_grid(positions, spacing, padding)
    field = np.zeros(grid.shape[:-1])

    # ④ Atomic accumulated electrostatic potential
    for atom_pos, q in zip(positions, charges):
        if q == 0:
            continue

        diff = grid - atom_pos
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.maximum(dist, 1e-3)  

        field += q / (epsilon * dist)

    return field, grid
