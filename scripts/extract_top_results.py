#!/usr/bin/env python3
"""
Extract global Top-N MM/GBSA results from a CE aptamer design run.
Optimized: Uses hash matching instead of PDBQT parsing (handles UNL residues correctly).
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
import hashlib
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_FOLDER = "final_result"
POSSIBLE_VINA_COLS = ["vina_score", "score_vina", "dock_score", "score", "vina"]


def log(msg):
    print(msg, flush=True)


def detect_vina_column(df):
    for col in POSSIBLE_VINA_COLS:
        if col in df.columns:
            log(f"[INFO] Detected docking score column: {col}")
            return col
    log("[WARN] No docking score column detected.")
    return None


def predict_structure(seq):
    try:
        proc = subprocess.run(
            ["RNAfold", "--noPS"],
            input=seq,
            text=True,
            capture_output=True,
            timeout=10
        )
        if proc.returncode != 0:
            return "N/A"
        lines = proc.stdout.strip().splitlines()
        if len(lines) >= 2:
            return lines[1].split()[0]
    except Exception:
        pass
    return "N/A"


def count_structure_features(structure):
    if not structure or structure == "N/A":
        return {"pairs": 0, "unpaired": 0, "stems": 0}
    pairs = structure.count("(")
    unpaired = structure.count(".")
    stems = 0
    in_stem = False
    for ch in structure:
        if ch in "()":
            if not in_stem:
                stems += 1
                in_stem = True
        else:
            in_stem = False
    return {"pairs": pairs, "unpaired": unpaired, "stems": stems}


def get_rescore_path(run_dir):
    run_path = Path(run_dir)
    rescore_path = run_path / "rescore"
    if not rescore_path.exists():
        alt_path = run_path / "result" / "rescore"
        if alt_path.exists():
            return alt_path
        raise FileNotFoundError(f"rescore directory not found in {run_dir}")
    return rescore_path


def load_all_mmgbsa_results(run_dir):
    rescore_path = get_rescore_path(run_dir)
    files = sorted(rescore_path.glob("iter_*/mmgbsa_results.csv"))
    if not files:
        raise FileNotFoundError(f"No mmgbsa_results.csv files found under: {rescore_path}")

    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            iter_num = int(f.parent.name.replace("iter_", ""))
            df["iter"] = iter_num
            df["source_csv"] = str(f)
            all_dfs.append(df)
        except Exception as e:
            log(f"[WARN] Failed to read {f}: {e}")

    if not all_dfs:
        raise RuntimeError("No valid MM/GBSA result tables could be loaded.")
    return pd.concat(all_dfs, ignore_index=True)


def filter_successful_mmgbsa(df):
    if "mmgbsa_ok" in df.columns:
        ok_mask = df["mmgbsa_ok"].astype(str).str.lower().isin(["true", "1"])
        return df[ok_mask].copy()
    else:
        log("[WARN] Column 'mmgbsa_ok' not found. Falling back to non-null mmgbsa_dg_bind.")
        return df.dropna(subset=["mmgbsa_dg_bind"]).copy()


def find_hash_for_sequence(seq, dock_hash_dirs):
    """
    Match sequence to dock directory by computing common hashes.
    Returns the 12-char hex hash if found, else None.
    """
    seq_clean = seq.upper().replace('T', 'U')
    # Try common hashing algorithms (first 12 chars)
    for algo in [hashlib.md5, hashlib.sha1, hashlib.sha256]:
        h = algo(seq_clean.encode()).hexdigest()[:12]
        if h in dock_hash_dirs:
            return h
    return None


def smart_copy(src, dst, use_symlink=False):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if use_symlink:
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
        else:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        return True
    except Exception as e:
        log(f"[WARN] Failed to copy/link {src} -> {dst}: {e}")
        return False


def copy_mmgbsa_directory(run_dir, iter_num, rank, target_dir, use_symlink=False):
    rescore_path = get_rescore_path(run_dir)
    pattern = os.path.join(str(rescore_path), f"iter_{iter_num:03d}", "mmgbsa", f"rank_{rank:03d}_f*")
    matches = glob.glob(pattern)
    if not matches:
        return False

    src_dir = matches[0]
    os.makedirs(target_dir, exist_ok=True)
    try:
        if use_symlink:
            link_name = os.path.join(target_dir, "mmgbsa_job")
            if not os.path.exists(link_name):
                os.symlink(os.path.abspath(src_dir), link_name)
        else:
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    if os.path.exists(d): shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        return True
    except Exception as e:
        log(f"[WARN] Failed to copy MM/GBSA directory: {e}")
        return False


def copy_docking_directory(run_dir, seq, target_dir, use_symlink=False, dock_hash_dirs=None):
    """
    Find hash for seq, then copy the entire dock/<hash> directory.
    """
    if dock_hash_dirs is None:
        return False, None

    seq_hash = find_hash_for_sequence(seq, dock_hash_dirs)
    if not seq_hash:
        return False, None

    dock_base = Path(run_dir) / "dock" / seq_hash
    if not dock_base.exists():
        return False, None

    try:
        os.makedirs(target_dir, exist_ok=True)
        target_hash_dir = os.path.join(target_dir, seq_hash)
        success = smart_copy(str(dock_base), target_hash_dir, use_symlink)
        return success, seq_hash
    except Exception as e:
        log(f"[WARN] Failed to copy docking directory for {seq_hash}: {e}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Extract global Top-N MM/GBSA results.")
    parser.add_argument("-r", "--run_dir", required=True)
    parser.add_argument("-n", "--top_n", type=int, default=50)
    parser.add_argument("-o", "--output_folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--use_symlink", action="store_true")
    parser.add_argument("--skip_structure", action="store_true")
    parser.add_argument("--unique_seq", action="store_true", help="Keep only unique sequences (best energy)")
    args = parser.parse_args()

    run_dir = args.run_dir
    top_n = args.top_n
    output_folder = args.output_folder
    use_symlink = args.use_symlink
    skip_structure = args.skip_structure
    unique_seq = args.unique_seq

    log(f"[INFO] Run directory: {run_dir}")
    log(f"[INFO] Output folder: {output_folder}")
    log(f"[INFO] Top N: {top_n}")

    # Load MM/GBSA results
    try:
        df_all = load_all_mmgbsa_results(run_dir)
    except FileNotFoundError as e:
        log(f"[ERROR] {e}"); sys.exit(1)
        
    log(f"[INFO] Loaded total rows: {len(df_all)}")
    df_ok = filter_successful_mmgbsa(df_all)
    log(f"[INFO] Successful MM/GBSA rows: {len(df_ok)}")
    if df_ok.empty:
        raise RuntimeError("No successful MM/GBSA rows found.")

    # Pre-scan dock/ directory names for fast matching
    dock_path = Path(run_dir) / "dock"
    if dock_path.exists():
        dock_hash_dirs = {d.name for d in dock_path.iterdir() if d.is_dir()}
        log(f"[INFO] Found {len(dock_hash_dirs)} docking directories. Will match via hash computation.")
    else:
        log("[WARN] dock/ directory not found. Docking files will be skipped.")
        dock_hash_dirs = set()

    # Handle unique sequences
    if unique_seq:
        df_ok = df_ok.sort_values("mmgbsa_dg_bind", ascending=True)
        df_ok = df_ok.drop_duplicates(subset=["seq"], keep="first").copy()
        log(f"[INFO] Unique-sequence rows after deduplication: {len(df_ok)}")

    # Select top N
    df_top = df_ok.nsmallest(top_n, "mmgbsa_dg_bind").reset_index(drop=True)
    df_top["global_rank"] = range(1, len(df_top) + 1)
    vina_col = detect_vina_column(df_top)

    # Predict structures
    structures = []
    if skip_structure:
        structures = ["N/A"] * len(df_top)
    else:
        log("[INFO] Predicting RNA secondary structures...")
        for i, seq in enumerate(df_top["seq"], start=1):
            structures.append(predict_structure(seq))
            if i % 10 == 0 or i == len(df_top):
                log(f"[INFO] Progress: {i}/{len(df_top)}")

    df_top["structure"] = structures
    df_top["pairs"] = [count_structure_features(s)["pairs"] for s in structures]
    df_top["unpaired"] = [count_structure_features(s)["unpaired"] for s in structures]
    df_top["stems"] = [count_structure_features(s)["stems"] for s in structures]

    # Prepare output
    out_dir = Path(run_dir) / output_folder
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    mmgbsa_out = out_dir / "mmgbsa_workdirs"
    docking_out = out_dir / "docking_poses"
    mmgbsa_out.mkdir()
    docking_out.mkdir()

    # Process each top result
    log("[INFO] Copying files...")
    records = []

    for idx, row in df_top.iterrows():
        global_rank = int(row["global_rank"])
        iter_num = int(row["iter"])
        orig_rank = int(row["rank"])
        seq = row["seq"]
        dg = row["mmgbsa_dg_bind"]

        folder_name = f"Rank{global_rank:02d}_Iter{iter_num:03d}_DG{dg:.2f}"
        mmgbsa_target = mmgbsa_out / folder_name
        docking_target = docking_out / folder_name

        mmgbsa_copied = copy_mmgbsa_directory(run_dir, iter_num, orig_rank, str(mmgbsa_target), use_symlink)
        docking_copied, dock_hash = copy_docking_directory(run_dir, seq, str(docking_target), use_symlink, dock_hash_dirs)

        record = {
            "global_rank": global_rank,
            "iter": iter_num,
            "rank_in_iter": orig_rank,
            "seq": seq,
            "dock_hash": dock_hash,
            "structure": row["structure"],
            "pairs": row["pairs"],
            "unpaired": row["unpaired"],
            "stems": row["stems"],
            "mmgbsa_dg_bind": dg,
            "mmgbsa_dir_copied": mmgbsa_copied,
            "docking_dir_copied": docking_copied,
        }
        if vina_col is not None:
            record[vina_col] = row[vina_col]
        records.append(record)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_top):
            log(f"[INFO] Progress: {idx+1}/{len(df_top)}")

    summary_df = pd.DataFrame(records)

    # Save outputs
    csv_file = out_dir / f"summary_top{top_n}.csv"
    summary_df.to_csv(csv_file, index=False)
    log(f"[INFO] Saved: {csv_file}")

    fasta_file = out_dir / f"top{top_n}_sequences.fasta"
    with open(fasta_file, "w") as f:
        for _, row in summary_df.iterrows():
            hash_str = row.get("dock_hash", "N/A")
            f.write(f">rank_{int(row['global_rank']):02d}|iter_{int(row['iter']):03d}|dg_{row['mmgbsa_dg_bind']:.2f}|hash_{hash_str}\n")
            f.write(f"{row['seq']}\n")
    log(f"[INFO] Saved: {fasta_file}")

    log("=" * 80)
    log(f"[DONE] Best MM/GBSA: {summary_df['mmgbsa_dg_bind'].min():.2f}")
    log(f"[DONE] Docking dirs copied: {summary_df['docking_dir_copied'].sum()}/{len(summary_df)}")
    log(f"[DONE] Output: {out_dir}")
    log("=" * 80)


if __name__ == "__main__":
    main()
