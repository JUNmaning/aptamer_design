import os
import sys
import glob
import shutil
import subprocess
import pandas as pd
from pathlib import Path

# ================= CONFIGURATION =================
RUN_DIR = r"runs/ce_run1"  # ⭐ Modify to your actual run directory
TOP_N = 50                 # Extract Top N sequences
OUTPUT_FOLDER = "final_result" # Name of the output folder
USE_SYMLINK = False        # True=Use symlinks (saves space), False=Copy files (safer, recommended)
# ===========================================

def predict_structure(seq):
    """Predicts secondary structure using RNAfold."""
    try:
        proc = subprocess.run(
            ['RNAfold', '--noPS'],
            input=seq, text=True,
            capture_output=True, timeout=5
        )
        if proc.returncode == 0:
            lines = proc.stdout.strip().split('\n')
            if len(lines) >= 2:
                return lines[1].split()[0]
    except Exception:
        pass
    return "N/A"

def find_source_files(run_dir, seq_hash, iter_num, rank):
    """
    Locate source file paths based on hash or sequence info.
    Assumes standard pipeline directory structure.
    """
    base_path = Path(run_dir)
    
    # 1. Locate MM/GBSA working directory (contains prmtop, mdcrd, sander log, etc.)
    # Path pattern: rescore/iter_XXX/mmgbsa/rank_XXX_f...
    gbsa_dirs = list(base_path.glob(f"rescore/iter_{iter_num:03d}/mmgbsa/rank_{rank:03d}_f*"))
    gbsa_dir = str(gbsa_dirs[0]) if gbsa_dirs else None
    
    return gbsa_dir

def main():
    print(f"🔍 Processing run directory: {RUN_DIR}")
    if not os.path.exists(RUN_DIR):
        print("❌ Error: Run directory does not exist!")
        sys.exit(1)

    # 1. Collect all MM/GBSA data
    print("📂 Reading MM/GBSA results from all iterations...")
    all_data = []
    
    csv_pattern = os.path.join(RUN_DIR, "rescore", "iter_*", "mmgbsa_results.csv")
    files = sorted(glob.glob(csv_pattern))
    
    if not files:
        print("❌ No mmgbsa_results.csv files found!")
        sys.exit(1)

    for f in files:
        try:
            df = pd.read_csv(f)
            # Extract iteration number
            iter_num = int(os.path.basename(os.path.dirname(os.path.dirname(f))).replace("iter_", ""))
            df['iter'] = iter_num
            df['source_file'] = f
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}")

    if not all_data:
        print("❌ No valid data found.")
        sys.exit(1)

    df_all = pd.concat(all_data, ignore_index=True)
    
    # Filter successful results
    # Note: mmgbsa_ok might be a boolean or a string 'True'/'False'
    if 'mmgbsa_ok' in df_all.columns:
        df_ok = df_all[df_all['mmgbsa_ok'].astype(str).str.lower() == 'true'].copy()
    else:
        # If column is missing, assume all read rows are successful, or filter by checking if dg_bind is not NaN
        df_ok = df_all.dropna(subset=['mmgbsa_dg_bind']).copy()
        print("⚠️ 'mmgbsa_ok' column not found. Filtering rows with empty ΔG values.")

    if len(df_ok) == 0:
        print("❌ No successful MM/GBSA calculation results found.")
        sys.exit(1)

    print(f"✅ Found {len(df_ok)} successful records.")

    # 2. Select Global Top-N
    print(f"🏆 Selecting Global Top-{TOP_N}...")
    # Sort by binding energy (lower is better)
    df_top = df_ok.nsmallest(TOP_N, 'mmgbsa_dg_bind').reset_index(drop=True)
    
    # Add global ranking
    df_top['global_rank'] = range(1, len(df_top) + 1)

    # 3. Predict secondary structures
    print("🧬 Predicting secondary structures (RNAfold)...")
    structures = []
    for idx, row in df_top.iterrows():
        seq = row['seq']
        struct = predict_structure(seq)
        structures.append(struct)
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{len(df_top)}")
    
    df_top['structure'] = structures
    df_top['pairs'] = df_top['structure'].apply(lambda x: x.count('(') if x != "N/A" else 0)
    df_top['stems'] = df_top['structure'].apply(lambda x: sum(1 for i, c in enumerate(x) if c in '()' and (i==0 or x[i-1] not in '()')) if x != "N/A" else 0)

    # 4. Prepare output folder
    out_path = Path(RUN_DIR) / OUTPUT_FOLDER
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    
    # Create subdirectories
    dirs = ['sequences', 'docking_poses', 'mmgbsa_workdirs', 'logs']
    for d in dirs:
        (out_path / d).mkdir()

    print(f"📦 Organizing files into {out_path}...")

    # List to record file mappings
    file_map = []

    for idx, row in df_top.iterrows():
        rank = int(row['global_rank'])
        seq_short = row['seq'][:10] + "..."
        iter_num = int(row['iter'])
        original_rank = int(row['rank']) # Rank in that iteration
        
        folder_name = f"Rank{rank:02d}_Iter{iter_num:03d}_DG{row['mmgbsa_dg_bind']:.2f}"
        target_dir = out_path / "mmgbsa_workdirs" / folder_name
        target_dir.mkdir()

        # --- A. Copy MM/GBSA working directory ---
        # Construct source path: rescore/iter_XXX/mmgbsa/rank_XXX_f...
        # Note: rank in filename is usually 3 digits, e.g., rank_001
        src_gbsa_pattern = os.path.join(RUN_DIR, "rescore", f"iter_{iter_num:03d}", "mmgbsa", f"rank_{original_rank:03d}_f*")
        src_gbsa_dirs = glob.glob(src_gbsa_pattern)
        
        gbsa_copied = False
        if src_gbsa_dirs:
            src_gbsa = src_gbsa_dirs[0] # Take the first match
            if USE_SYMLINK:
                os.symlink(os.path.abspath(src_gbsa), target_dir / "mmgbsa_job")
            else:
                # Copy entire folder contents
                for item in os.listdir(src_gbsa):
                    s = os.path.join(src_gbsa, item)
                    d = os.path.join(target_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, symlinks=False)
                    else:
                        shutil.copy2(s, d)
            gbsa_copied = True
            
            # Copy key logs to logs directory
            log_src = os.path.join(src_gbsa, "MMPBSA.log")
            if os.path.exists(log_src):
                shutil.copy2(log_src, out_path / "logs" / f"{folder_name}_MMPBSA.log")
        else:
            print(f"⚠️ MM/GBSA source directory not found for Rank {rank}: {src_gbsa_pattern}")

        # --- B. Find and Copy Docking Pose ---
        # Challenge: We need the hash. Usually mmgbsa_results.csv does not contain hash.
        # Strategy: Look up hash in topN.csv or history.csv based on sequence, or search in dock dir (slow)
        # Assumption: We can establish a seq -> hash map from the run's topN.csv
        # If topN.csv is missing, skip docking copy
        
        topn_csv = os.path.join(RUN_DIR, "topN.csv")
        dock_copied = False
        
        if os.path.exists(topn_csv):
            df_topn = pd.read_csv(topn_csv)
            # Try matching sequence (Note: seq in topN.csv should match mmgbsa exactly)
            match = df_topn[df_topn['seq'] == row['seq']]
            
            if not match.empty:
                # Take the first matching hash (In theory, hash for the same seq should be the same)
                seq_hash = match.iloc[0]['hash']
                
                # Construct dock path: dock/<hash>/frame_*/lig_*/vina_output.pdbqt
                dock_src_base = os.path.join(RUN_DIR, "dock", str(seq_hash))
                
                if os.path.exists(dock_src_base):
                    # Find the best pose (usually frame_1 or lowest score)
                    # For simplicity, copy the entire hash folder, or just the first frame's first ligand
                    # To save space, we copy only the best pose
                    frames = sorted(glob.glob(os.path.join(dock_src_base, "frame_*")))
                    if frames:
                        best_frame = frames[0] # Assume frame_1 is representative
                        ligands = sorted(glob.glob(os.path.join(best_frame, "lig_*")))
                        if ligands:
                            best_lig = ligands[0]
                            target_dock_dir = out_path / "docking_poses" / folder_name
                            target_dock_dir.mkdir(exist_ok=True)
                            
                            # Copy vina output
                            vina_out = os.path.join(best_lig, "vina_output.pdbqt")
                            if os.path.exists(vina_out):
                                dest_file = target_dock_dir / "best_pose.pdbqt"
                                if USE_SYMLINK:
                                    os.symlink(os.path.abspath(vina_out), dest_file)
                                else:
                                    shutil.copy2(vina_out, dest_file)
                                dock_copied = True
                                
                                # Also copy receptor for viewing
                                rec_file = os.path.join(best_lig, "receptor.pdbqt")
                                if os.path.exists(rec_file):
                                    shutil.copy2(rec_file, target_dock_dir / "receptor.pdbqt")
        
        # Record information
        file_map.append({
            "Global_Rank": rank,
            "Iter": iter_num,
            "Seq": row['seq'],
            "Structure": row['structure'],
            "Pairs": row['pairs'],
            "Stems": row['stems'],
            "MMGBSA_DG": row['mmgbsa_dg_bind'],
            "Vina_Score": row.get('vina_score', 'N/A'), # If column exists
            "GBSA_Dir_Copied": gbsa_copied,
            "Dock_Pose_Copied": dock_copied
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  File organization progress: {idx+1}/{len(df_top)}")

    # 5. Save summary table
    summary_df = pd.DataFrame(file_map)
    summary_file = out_path / "summary_top50.xlsx"
    summary_df.to_excel(summary_file, index=False)
    summary_df.to_csv(out_path / "summary_top50.csv", index=False)

    print("\n" + "="*60)
    print("✅ Processing Complete!")
    print("="*60)
    print(f"📁 Results saved to: {out_path}")
    print(f"   - sequences/: Raw sequence text files (Optional, can add code to generate)")
    print(f"   - docking_poses/: Best docking poses (PDBQT)")
    print(f"   - mmgbsa_workdirs/: Complete MM/GBSA calculation directories (includes prmtop, logs, etc.)")
    print(f"   - summary_top50.xlsx: Detailed statistical table")
    print("\n📊 Data Statistics:")
    print(f"   Avg Pairs: {summary_df['Pairs'].mean():.1f}")
    print(f"   Avg Stems: {summary_df['Stems'].mean():.1f}")
    print(f"   Best ΔG: {summary_df['MMGBSA_DG'].min():.2f} kcal/mol")
    
    # Extra: Generate FASTA file
    fasta_file = out_path / "top50_sequences.fasta"
    with open(fasta_file, 'w') as f:
        for _, row in summary_df.iterrows():
            f.write(f">Rank{row['Global_Rank']:02d}_DG{row['MMGBSA_DG']:.2f}_Struct{row['Structure']}\n")
            f.write(f"{row['Seq']}\n")
    print(f"   - top50_sequences.fasta: Sequence file (with structure annotation)")

if __name__ == "__main__":
    main()
