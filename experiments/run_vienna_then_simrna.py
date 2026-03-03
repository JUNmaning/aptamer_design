
from structure.vienna import rnafold_dotbracket
from structure.simrna import fold_sequence_to_pdb

seq = "CAGGGCACCCTTCTAATAGGAATTAGGT"  # NA is also fine, it will automatically convert T->U

# 1) ViennaRNA gets dot-bracket
res2d = rnafold_dotbracket(seq)
print("RNA:", res2d.rna_seq)
print("Dot-bracket:", res2d.dotbracket)
print("MFE (kcal/mol):", res2d.mfe_kcal_mol)

# 2) SimRNA uses the -S constraint for folding
res3d = fold_sequence_to_pdb(
    seq=res2d.rna_seq,
    workdir="runs/vienna_then_simrna",
    out_prefix="demo",
    seed=123,
    omp_threads=8,
    n_iterations=200000,          # First, conduct a quick test using a smaller sample size
    ss_restraints=res2d.dotbracket,
    frame_to_export="13",         # Don't export frame 1 forever; try exporting later frames first
)

print("All-atom PDB:", res3d.all_atom_pdb)
