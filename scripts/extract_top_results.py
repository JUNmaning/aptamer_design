#!/usr/bin/env python3
"""
Extract global Top-N MM/GBSA results from a CE aptamer design run.

Features:
1. Scan all rescore/iter_*/mmgbsa_results.csv files
2. Collect successful MM/GBSA entries across all iterations
3. Select global Top-N by mmgbsa_dg_bind
4. Predict RNA secondary structure using RNAfold
5. Save summary table (CSV/XLSX)
6. Save FASTA sequences
7. Copy corresponding MM/GBSA work directories
8. Copy corresponding docking pose files (Enhanced: tries multiple sources for seq->hash mapping)

Usage examples:
    python scripts/extract_top_results.py --run_dir runs/ce_run1
    python scripts/extract_top_results.py --run_dir runs/ce_run1 --top_n 10
    python scripts/extract_top_results.py -r runs/ce_run1 -n 50 --use_symlink
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
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
    """
    Predict RNA secondary structure using RNAfold.
    Returns dot-bracket string or 'N/A' on failure.
    """
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
    """
    Count simple secondary structure features.
    """
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
    """Helper to find rescore directory (supports result/rescore alternative)."""
    run_path = Path(run_dir)
    rescore_path = run_path / "rescore"
    if not rescore_path.exists():
        alt_path = run_path / "result" / "rescore"
        if alt_path.exists():
            log(f"[WARN] Using alternative path: {alt_path}")
            return alt_path
        raise FileNotFoundError(f"rescore directory not found in {run_dir}")
    return rescore_path


def load_all_mmgbsa_results(run_dir):
    """
    Load all mmgbsa_results.csv files from rescore/iter_*.
    """
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

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


def filter_successful_mmgbsa(df):
    """
    Keep only successful MM/GBSA rows.
    """
    if "mmgbsa_ok" in df.columns:
        ok_mask = df["mmgbsa_ok"].astype(str).str.lower().isin(["true", "1"])
        return df[ok_mask].copy()
    else:
        log("[WARN] Column 'mmgbsa_ok' not found. Falling back to non-null mmgbsa_dg_bind.")
        return df.dropna(subset=["mmgbsa_dg_bind"]).copy()


def build_seq_to_hash_map(run_dir):
    """
    Build a seq -> hash map from multiple sources (in order of priority):
    1. topN.csv (if available)
    2. All manifest.jsonl files from rescore/iter_*
    
    Returns a dictionary {sequence: hash}.
    """
    seq_to_hash = {}
    
    # Source 1: topN.csv
    topn_path = Path(run_dir) / "topN.csv"
    if topn_path.exists():
        try:
            df_topn = pd.read_csv(topn_path)
            if "seq" in df_topn.columns and "hash" in df_topn.columns:
                for _, row in df_topn.iterrows():
                    seq_to_hash[row["seq"]] = row["hash"]
                log(f"[INFO] Loaded {len(df_topn)} seq->hash mappings from topN.csv")
        except Exception as e:
            log(f"[WARN] Failed to read topN.csv: {e}")
    else:
        log("[INFO] topN.csv not found, will try manifest.jsonl files")

    # Source 2: manifest.jsonl from each iteration
    rescore_path = get_rescore_path(run_dir)
    manifest_files = sorted(rescore_path.glob("iter_*/manifest.jsonl"))
    
    loaded_from_manifest = 0
    for mf in manifest_files:
        try:
            with open(mf, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        seq = entry.get("seq")
                        hash_val = entry.get("hash")
                        if seq and hash_val and seq not in seq_to_hash:
                            seq_to_hash[seq] = hash_val
                            loaded_from_manifest += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log(f"[WARN] Failed to read {mf}: {e}")
    
    if loaded_from_manifest > 0:
        log(f"[INFO] Loaded additional {loaded_from_manifest} seq->hash mappings from manifest.jsonl files")

    if not seq_to_hash:
        log("[ERROR] No seq->hash mappings found anywhere!")
        
    return seq_to_hash


def smart_copy(src, dst, use_symlink=False):
    """
    Copy a file or directory, or create a symlink if requested.
    """
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
    """
    Copy the MM/GBSA working directory for one selected result.
    """
    rescore_path = get_rescore_path(run_dir)
    
    pattern = os.path.join(
        str(rescore_path),
        f"iter_{iter_num:03d}",
        "mmgbsa",
        f"rank_{rank:03d}_f*"
    )
    matches = glob.glob(pattern)

    if not matches:
        log(f"[WARN] MMGBSA dir not found for iter {iter_num}, rank {rank}")
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
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        return True
    except Exception as e:
        log(f"[WARN] Failed to copy MM/GBSA directory {src_dir}: {e}")
        return False


def find_seq_hash_in_dock(run_dir, seq):
    """
    Fallback: Search for sequence hash by directly scanning dock/ directory.
    This is slow but serves as last resort.
    """
    dock_path = Path(run_dir) / "dock"
    if not dock_path.exists():
        return None
    
    # 遍历 dock 下的每个 hash 目录
    try:
        for hash_dir in dock_path.iterdir():
            if not hash_dir.is_dir():
                continue
            # 检查 hash 目录下是否有序列文件
            # 常见位置: frame_*/lig_*/ 或者序列信息文件
            # 这里我们简化：假设 hash 目录名就是 hash，且如果找不到更直接的方法
            # 可以通过其他元数据文件匹配
            
            # 更可靠的方法：检查是否有 .seq 文件或记录文件包含序列信息
            # 但为了效率，这里我们只返回目录名作为候选
            # 实际应用中可能需要更复杂的匹配逻辑
            pass
    except Exception as e:
        pass
    
    return None


def copy_docking_files(run_dir, seq, seq_to_hash, target_dir, use_symlink=False):
    """
    Copy docking files for a selected sequence.
    Tries to find hash from seq_to_hash map.
    """
    if seq not in seq_to_hash:
        log(f"[WARN] Sequence not found in any hash map. Cannot locate docking files.")
        return False

    seq_hash = str(seq_to_hash[seq])
    dock_base = Path(run_dir) / "dock" / seq_hash
    
    if not dock_base.exists():
        log(f"[WARN] Docking directory not found: {dock_base}")
        return False

    try:
        frames = sorted(dock_base.glob("frame_*"))
        if not frames:
            log(f"[WARN] No frame_* directories in {dock_base}")
            return False

        best_frame = frames[0]
        lig_dirs = sorted(best_frame.glob("lig_*"))
        if not lig_dirs:
            log(f"[WARN] No lig_* directories in {best_frame}")
            return False

        best_lig_dir = lig_dirs[0]
        
        os.makedirs(target_dir, exist_ok=True)
        target_lig_dir = os.path.join(target_dir, "docking_job")

        log(f"[INFO] Copying docking from: {best_lig_dir}")
        
        success = smart_copy(str(best_lig_dir), target_lig_dir, use_symlink)
        
        if success:
            log(f"[INFO] Docking files saved to: {target_lig_dir}")
        return success

    except Exception as e:
        log(f"[WARN] Failed to copy docking files for seq hash {seq_hash}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract global Top-N MM/GBSA results.")
    parser.add_argument("-r", "--run_dir", required=True, help="Run directory, e.g. runs/ce_run1")
    parser.add_argument("-n", "--top_n", type=int, default=50, help="Number of top results to extract")
    parser.add_argument("-o", "--output_folder", default=DEFAULT_OUTPUT_FOLDER, help="Output folder name inside run_dir")
    parser.add_argument("--use_symlink", action="store_true", help="Use symlinks instead of copying files")
    parser.add_argument("--skip_structure", action="store_true", help="Skip RNA secondary structure prediction")
    parser.add_argument("--unique_seq", action="store_true", help="Keep only unique sequences before selecting Top-N")
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

    # Load all MM/GBSA results
    try:
        df_all = load_all_mmgbsa_results(run_dir)
    except FileNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
        
    log(f"[INFO] Loaded total rows: {len(df_all)}")

    df_ok = filter_successful_mmgbsa(df_all)
    log(f"[INFO] Successful MM/GBSA rows: {len(df_ok)}")

    if df_ok.empty:
        raise RuntimeError("No successful MM/GBSA rows found.")

    if unique_seq:
        df_ok = df_ok.sort_values("mmgbsa_dg_bind", ascending=True)
        df_ok = df_ok.drop_duplicates(subset=["seq"], keep="first").copy()
        log(f"[INFO] Unique-sequence rows after deduplication: {len(df_ok)}")

    df_top = df_ok.nsmallest(top_n, "mmgbsa_dg_bind").reset_index(drop=True)
    df_top["global_rank"] = range(1, len(df_top) + 1)

    vina_col = detect_vina_column(df_top)

    # Predict structures
    structures = []
    if skip_structure:
        structures = ["N/A"] * len(df_top)
    else:
        log("[INFO] Predicting RNA secondary structures with RNAfold...")
        for i, seq in enumerate(df_top["seq"], start=1):
            structures.append(predict_structure(seq))
            if i % 10 == 0 or i == len(df_top):
                log(f"[INFO] Structure prediction progress: {i}/{len(df_top)}")

    df_top["structure"] = structures

    pairs_list = []
    unpaired_list = []
    stems_list = []
    for s in structures:
        stats = count_structure_features(s)
        pairs_list.append(stats["pairs"])
        unpaired_list.append(stats["unpaired"])
        stems_list.append(stats["stems"])

    df_top["pairs"] = pairs_list
    df_top["unpaired"] = unpaired_list
    df_top["stems"] = stems_list

    # Prepare output dirs
    out_dir = Path(run_dir) / output_folder
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    mmgbsa_out = out_dir / "mmgbsa_workdirs"
    docking_out = out_dir / "docking_poses"
    logs_out = out_dir / "logs"

    mmgbsa_out.mkdir()
    docking_out.mkdir()
    logs_out.mkdir()

    # Build seq->hash map (Enhanced: tries topN.csv + manifest.jsonl)
    seq_to_hash = build_seq_to_hash_map(run_dir)
    log(f"[INFO] Total seq->hash mappings available: {len(seq_to_hash)}")

    # Copy files
    log("[INFO] Copying related MM/GBSA and docking files...")
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

        mmgbsa_copied = copy_mmgbsa_directory(
            run_dir=run_dir,
            iter_num=iter_num,
            rank=orig_rank,
            target_dir=str(mmgbsa_target),
            use_symlink=use_symlink
        )

        docking_copied = copy_docking_files(
            run_dir=run_dir,
            seq=seq,
            seq_to_hash=seq_to_hash,
            target_dir=str(docking_target),
            use_symlink=use_symlink
        )

        record = {
            "global_rank": global_rank,
            "iter": iter_num,
            "rank_in_iter": orig_rank,
            "seq": seq,
            "structure": row["structure"],
            "pairs": row["pairs"],
            "unpaired": row["unpaired"],
            "stems": row["stems"],
            "mmgbsa_dg_bind": dg,
            "mmgbsa_dir_copied": mmgbsa_copied,
            "docking_files_copied": docking_copied,
        }

        if vina_col is not None:
            record[vina_col] = row[vina_col]

        records.append(record)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(df_top):
            log(f"[INFO] File collection progress: {idx+1}/{len(df_top)}")

    summary_df = pd.DataFrame(records)

    # Save summary
    csv_file = out_dir / f"summary_top{top_n}.csv"
    summary_df.to_csv(csv_file, index=False)
    log(f"[INFO] Saved summary CSV: {csv_file}")

    xlsx_file = out_dir / f"summary_top{top_n}.xlsx"
    try:
        summary_df.to_excel(xlsx_file, index=False)
        log(f"[INFO] Saved summary Excel: {xlsx_file}")
    except Exception as e:
        log(f"[WARN] Failed to save Excel file: {e}")

    # Save FASTA
    fasta_file = out_dir / f"top{top_n}_sequences.fasta"
    with open(fasta_file, "w") as f:
        for _, row in summary_df.iterrows():
            f.write(
                f">rank_{int(row['global_rank']):02d}|iter_{int(row['iter']):03d}|dg_{row['mmgbsa_dg_bind']:.2f}|structure_{row['structure']}\n"
            )
            f.write(f"{row['seq']}\n")
    log(f"[INFO] Saved FASTA: {fasta_file}")

    # Print short report
    log("=" * 80)
    log("[DONE] Extraction completed successfully.")
    log(f"[DONE] Output directory: {out_dir}")
    log(f"[DONE] Best MM/GBSA score: {summary_df['mmgbsa_dg_bind'].min():.2f}")
    log(f"[DONE] Mean pairs: {summary_df['pairs'].mean():.2f}")
    log(f"[DONE] Mean stems: {summary_df['stems'].mean():.2f}")
    
    # Summary of docking file copy status
    copied_count = summary_df['docking_files_copied'].sum()
    total_count = len(summary_df)
    log(f"[DONE] Docking files copied: {copied_count}/{total_count}")
    log("=" * 80)


if __name__ == "__main__":
    main()
