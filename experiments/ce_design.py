from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from utils.config import load_yaml, deep_update, parse_set_kv, set_by_dotted_key
from utils.meta import write_meta, write_resolved_config, update_meta_end
from utils.logging_utils import setup_logging

from optimizers.ce import CEConfig, SequenceDistribution
from structure.simrna import fold_sequence_to_pdb_with_vienna
from dock.vina import dock_one

from ligand.rdkit_conformers import LigandConformerConfig, prepare_ligand_conformers


# -----------------------------
# Helpers
# -----------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def stable_int_seed(global_seed: int, seq: str, iter_idx: int) -> int:
    h = hashlib.sha1(f"{global_seed}|{iter_idx}|{seq}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def locate_ligand(cfg: dict) -> Path:
    ligand = cfg.get("inputs", {}).get("ligand", None)
    if ligand:
        p = Path(ligand).expanduser().resolve()
        if p.exists():
            return p
    data_dir = Path("data").resolve()
    if (data_dir / "ligand.sdf").exists():
        return (data_dir / "ligand.sdf").resolve()
    if (data_dir / "ligand.pdb").exists():
        return (data_dir / "ligand.pdb").resolve()
    raise FileNotFoundError("Ligand not found. Provide inputs.ligand or put data/ligand.sdf (or data/ligand.pdb).")


def is_penalty(rec: Dict) -> bool:
    try:
        return float(rec.get("score", 1e9)) >= 1e8
    except Exception:
        return True


# -----------------------------
# Distribution accessors (robust to dist.probs being a method)
# -----------------------------
def get_probs(dist: SequenceDistribution) -> np.ndarray:
    p = getattr(dist, "probs", None)
    if callable(p):
        P = p()
    else:
        P = p
    return np.asarray(P, dtype=float)


def set_probs(dist: SequenceDistribution, P: np.ndarray) -> None:
    P = np.asarray(P, dtype=float)
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    logits = np.log(P)

    for name in ["logits", "_logits", "theta", "_theta"]:
        if hasattr(dist, name):
            try:
                setattr(dist, name, np.array(logits, dtype=float))
                return
            except Exception:
                pass

    for name in dir(dist):
        if name.startswith("_") and name not in ["_logits", "_theta"]:
            continue
        try:
            v = getattr(dist, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.shape == logits.shape:
            try:
                setattr(dist, name, np.array(logits, dtype=float))
                return
            except Exception:
                pass

    raise AttributeError("Cannot set distribution probability matrix (no writable logits-like attribute found).")


def dist_stats(dist: SequenceDistribution) -> Dict[str, float]:
    P = get_probs(dist)
    eps = 1e-12
    P_clip = np.clip(P, eps, 1.0)
    H_i = -np.sum(P_clip * np.log(P_clip), axis=1)
    maxp_i = np.max(P, axis=1)
    return {
        "dist_entropy_sum": float(np.sum(H_i)),
        "dist_entropy_mean": float(np.mean(H_i)),
        "dist_maxp_mean": float(np.mean(maxp_i)),
        "dist_maxp_frac095": float(np.mean(maxp_i >= 0.95)),
    }


def apply_uniform_mix_entropy_regularization(dist: SequenceDistribution, mix: float) -> None:
    mix = float(mix)
    if mix <= 0:
        return
    P = get_probs(dist)
    L, K = P.shape
    U = np.ones((L, K), dtype=float) / float(K)
    P2 = (1.0 - mix) * P + mix * U
    P2 = P2 / np.sum(P2, axis=1, keepdims=True)
    set_probs(dist, P2)


def hamming_norm(a: str, b: str) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    d = sum(1 for x, y in zip(a, b) if x != y)
    return float(d) / float(len(a))


def dotbracket_pairs(db: str) -> set[tuple[int, int]]:
    stack = []
    pairs = set()
    for i, ch in enumerate(db):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                pairs.add((j, i))
    return pairs


def struct_distance(db1: Optional[str], db2: Optional[str], metric: str = "bp") -> float:
    if not db1 or not db2:
        return 1.0
    if metric == "edit":
        return hamming_norm(db1, db2)
    p1 = dotbracket_pairs(db1)
    p2 = dotbracket_pairs(db2)
    if not p1 and not p2:
        return 0.0
    inter = len(p1 & p2)
    union = len(p1 | p2)
    j = inter / union if union > 0 else 0.0
    return float(1.0 - j)


def select_elites_with_constraints(
    sorted_records: List[Dict],
    elite_n: int,
    min_hamming: float,
    struct_unique: bool,
    struct_metric: str,
    struct_min_dist: float,
    logger,
) -> Tuple[List[Dict], Dict[str, float], List[Dict]]:
    elite_n = max(1, int(elite_n))
    stage_log: List[Dict] = []

    cur_struct_min = float(struct_min_dist)
    cur_struct_unique = bool(struct_unique)
    cur_min_ham = float(min_hamming)

    def try_pick(cur_min_ham_: float, cur_struct_unique_: bool, cur_struct_min_: float) -> List[Dict]:
        picked: List[Dict] = []
        picked_seqs: List[str] = []
        picked_db: List[Optional[str]] = []
        seen_db: set[str] = set()

        for rec in sorted_records:
            if len(picked) >= elite_n:
                break
            if is_penalty(rec):
                continue

            s = rec["seq"]
            db = rec.get("dot_bracket", None)

            ok = True

            if picked_seqs and cur_min_ham_ > 0:
                if min(hamming_norm(s, t) for t in picked_seqs) < cur_min_ham_:
                    ok = False

            if ok and cur_struct_unique_:
                if db and db in seen_db:
                    ok = False

            if ok and cur_struct_min_ > 0:
                if picked_db:
                    if min(struct_distance(db, tdb, metric=struct_metric) for tdb in picked_db) < cur_struct_min_:
                        ok = False

            if ok:
                picked.append(rec)
                picked_seqs.append(s)
                picked_db.append(db)
                if db:
                    seen_db.add(db)

        return picked

    max_rounds = 12
    for r in range(max_rounds):
        elites = try_pick(cur_min_ham, cur_struct_unique, cur_struct_min)
        stage_log.append(
            {
                "round": r,
                "picked": len(elites),
                "min_hamming": cur_min_ham,
                "struct_unique": cur_struct_unique,
                "struct_min_dist": cur_struct_min,
            }
        )
        if len(elites) >= elite_n:
            break

        if cur_struct_min > 0:
            cur_struct_min = max(0.0, cur_struct_min * 0.8)
            continue
        if cur_struct_unique:
            cur_struct_unique = False
            continue
        if cur_min_ham > 0:
            cur_min_ham = max(0.0, cur_min_ham * 0.5)
            continue
        break

    if logger is not None:
        msg = "elite_fallback:"
        for st in stage_log:
            msg += f" [r{st['round']} picked={st['picked']}/{elite_n} ham>={st['min_hamming']:.3f} struct_unique={st['struct_unique']} struct_min={st['struct_min_dist']:.3f}]"
        logger.info(msg)

    elites = elites[:elite_n] if elites else []
    elite_seqs = [e["seq"] for e in elites]
    elite_db = [e.get("dot_bracket", None) for e in elites]

    ham_mean = 0.0
    ham_min = 0.0
    unique_frac = 1.0
    uniq_db = 0

    if len(elite_seqs) >= 2:
        dists = []
        for i in range(len(elite_seqs)):
            for j in range(i + 1, len(elite_seqs)):
                dists.append(hamming_norm(elite_seqs[i], elite_seqs[j]))
        ham_mean = float(np.mean(dists)) if dists else 0.0
        ham_min = float(np.min(dists)) if dists else 0.0

    if elite_seqs:
        unique_frac = float(len(set(elite_seqs)) / len(elite_seqs))

    uniq_db = len(set([d for d in elite_db if d])) if elite_db else 0

    metrics = {
        "elite_hamming_mean": ham_mean,
        "elite_hamming_min": ham_min,
        "elite_unique_frac": unique_frac,
        "n_unique_dotbracket": float(uniq_db),
    }
    return elites, metrics, stage_log


# -----------------------------
# Eval config + cache
# -----------------------------
@dataclass
class EvalConfig:
    omp_threads: int = 1
    simrna_iterations: int = 200000
    simrna_frames: str = "1"
    decoy_agg: str = "min"
    vina_exhaustiveness: int = 8
    vina_padding: float = 8.0
    vina_cpu: int = 1

    ligand_conformers: int = 0
    ligand_seed: int = 123
    ligand_prune_rmsd: float = 0.5
    ligand_forcefield: str = "MMFF"
    ligand_optimize: bool = True
    ligand_max_opt_iters: int = 200


@dataclass
class MDConfig:
    enabled: bool = False
    topX: int = 5
    gmx: str = "gmx"
    mdp_dir: str = "mdp"
    forcefield: str = "amber14sb"
    water: str = "tip3p"
    strip_5prime_phosphate: bool = True
    strip_3prime_phosphate: bool = False
    box_type: str = "cubic"
    box_distance_nm: float = 1.0
    salt_conc_m: float = 0.15
    neutralize: bool = True
    threads: int = 0
    gpu: bool = False
    skip_if_done: bool = True


class DiskCache:
    """Simple JSONL cache; key is seq."""
    def __init__(self, path: Path):
        self.path = path
        self.map: Dict[str, Dict] = {}
        if path.exists():
            with path.open("r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "seq" in rec:
                            self.map[rec["seq"]] = rec
                    except Exception:
                        continue

    def get(self, seq: str) -> Optional[Dict]:
        return self.map.get(seq)

    def put(self, rec: Dict) -> None:
        self.map[rec["seq"]] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------
# Defaults
# -----------------------------
def default_cfg() -> dict:
    return {
        "run": {
            "run_dir": "runs/ce_default",
            "seed": 123,
            "log_level": "INFO",
            "workers": 1,
            "resume": True,
        },
        "inputs": {"ligand": "data/ligand.sdf"},
        "ce": {
            "length": 30,
            "alphabet": "RNA",
            "batch": 24,
            "iters": 10,
            "elite_frac": 0.2,
            "alpha": 0.4,
            "min_prob": 0.02,
            "temperature": 1.0,
            "topN": 20,
        },
        "regularize": {
            "entropy_uniform_mix": 0.0,
            "elite_min_hamming": 0.0,
            "struct_unique": False,
            "struct_metric": "bp",
            "struct_min_dist": 0.0,
        },
        "monitor": {
            "penalty_skip": {
                "enabled": True,
                "frac_thr": 0.95,
                "action": "skip",
                "reset_patience": 2,
            },
            "early_stop": {
                "enabled": False,
                "patience": 6,
                "min_improve": 0.05,
            },
            "collapse": {
                "enabled": False,
                "max_prob": 0.95,
                "frac_thr": 0.60,
                "patience": 4,
            },
        },
        "rescore": {
            "enabled": False,
            "top_frac": None,
            "elite_frac": None,
            "include_penalty": False,
        },
        "eval": {
            "omp_threads": 1,
            "simrna_iterations": 200000,
            "simrna_frames": "1",
            "decoy_agg": "min",
            "vina_exhaustiveness": 8,
            "vina_cpu": 1,
            "grid_padding": 8.0,
            "ligand_conformers": 0,
            "ligand_seed": 123,
            "ligand_prune_rmsd": 0.5,
            "ligand_forcefield": "MMFF",
            "ligand_optimize": True,
            "ligand_max_opt_iters": 200,
        },
        "md": {
            "enabled": False,
            "topX": 5,
            "gmx": "gmx",
            "mdp_dir": "mdp",
            "forcefield": "amber14sb",
            "water": "tip3p",
            "box_type": "cubic",
            "box_distance_nm": 1.0,
            "salt_conc_m": 0.15,
            "neutralize": True,
            "threads": 0,
            "gpu": False,
            "skip_if_done": True,
        },
    }


# -----------------------------
# Docking call compatibility
# -----------------------------
def dock_one_compat(
    name: str,
    receptor_pdb: Path,
    ligand_sdf: Path,
    workdir: Path,
    padding: float,
    exhaustiveness: int,
    cpu: int,
):
    try:
        return dock_one(
            name=name,
            receptor_pdb=receptor_pdb,
            ligand_sdf=ligand_sdf,
            workdir=workdir,
            padding_angstrom=float(padding),
            exhaustiveness=int(exhaustiveness),
            cpu=int(cpu),
        )
    except TypeError:
        return dock_one(
            name=name,
            receptor_pdb=receptor_pdb,
            ligand_sdf=ligand_sdf,
            workdir=workdir,
            padding_angstrom=float(padding),
            exhaustiveness=int(exhaustiveness),
        )


# -----------------------------
# Ligand ensemble preparation (with charge preservation)
# -----------------------------
def prepare_ligand_ensemble_if_needed(
    ligand_file: Path,
    run_dir: Path,
    eval_cfg: EvalConfig,
    logger,
) -> Tuple[List[Path], Optional[Path]]:
    """
    Returns (ligand_sdfs, manifest_path)
    If ligand_conformers==0 => [ligand_file], None
    If ligand_ensemble already exists with charge info, SKIP regeneration
    to preserve user-specified charge settings.
    """
    n = int(getattr(eval_cfg, "ligand_conformers", 0) or 0)
    if n <= 0:
        return [ligand_file], None

    out_dir = (run_dir / "ligand_ensemble").resolve()
    manifest_file = out_dir / "manifest.json"
    charge_file = out_dir / "charge.txt"

    # If user already prepared ligand ensemble (e.g. with --charge -1), reuse it
    if manifest_file.exists() and charge_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text())
            items = manifest_data.get("items", [])
            existing_files = [Path(item["path"]) for item in items]
            if existing_files and all(p.exists() for p in existing_files):
                charge_info = manifest_data.get("charge_info", {})
                charge_val = charge_info.get("formal_charge", "N/A")
                charge_method = charge_info.get("detection_method", "N/A")
                logger.info(
                    f"Ligand ensemble already exists: {len(existing_files)} conformers, "
                    f"charge={charge_val} (method={charge_method}). "
                    f"Reusing existing ensemble (will NOT overwrite)."
                )
                return existing_files, manifest_file
        except Exception as e:
            logger.warning(f"Existing ligand ensemble corrupted, regenerating: {e}")

    # No pre-existing ensemble found, generate new one
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LigandConformerConfig(
        n_confs=n,
        seed=int(eval_cfg.ligand_seed),
        prune_rmsd=float(eval_cfg.ligand_prune_rmsd),
        forcefield=str(eval_cfg.ligand_forcefield),
        optimize=bool(eval_cfg.ligand_optimize),
        max_opt_iters=int(eval_cfg.ligand_max_opt_iters),
    )

    conf_sdfs, manifest = prepare_ligand_conformers(ligand_file, out_dir, cfg)
    logger.info(f"Ligand ensemble: {len(conf_sdfs)} conformers (manifest={manifest})")
    return conf_sdfs, manifest


# -----------------------------
# Score aggregation across frames
# -----------------------------
def aggregate_scores(scores: List[float], method: str = "min", boltz_beta: float = 1.0) -> Dict[str, float]:
    if not scores:
        return {
            "score": 1e9,
            "score_min": 1e9,
            "score_mean": 1e9,
            "score_p10": 1e9,
            "score_boltz": 1e9,
        }
    arr = np.array(scores, dtype=float)
    smin = float(np.min(arr))
    smean = float(np.mean(arr))
    sp10 = float(np.quantile(arr, 0.10))

    x = arr - smin
    w = np.exp(-float(boltz_beta) * x)
    w = w / (np.sum(w) + 1e-12)
    sboltz = float(np.sum(w * arr))

    method = (method or "min").lower()
    if method == "mean":
        s = smean
    elif method in ["p10", "p10q", "q10"]:
        s = sp10
    elif method in ["boltz", "boltzmann", "softmin"]:
        s = sboltz
    else:
        s = smin

    return {
        "score": float(s),
        "score_min": smin,
        "score_mean": smean,
        "score_p10": sp10,
        "score_boltz": sboltz,
    }


# -----------------------------
# Core evaluation (sequence -> fold frames -> dock -> aggregate)
# -----------------------------
def evaluate_sequence(
    seq: str,
    ligand_sdfs: List[Path],
    work_root: Path,
    eval_cfg: EvalConfig,
    seed: int,
) -> Dict:
    tag = sha1(seq)
    fold_dir = work_root / "fold" / tag

    res2d, res3d = fold_sequence_to_pdb_with_vienna(
        seq=seq,
        workdir=fold_dir,
        out_prefix="aptamer",
        omp_threads=int(eval_cfg.omp_threads),
        n_iterations=int(eval_cfg.simrna_iterations),
        frames_to_export=str(eval_cfg.simrna_frames),
        seed=int(seed),
    )

    dotb = getattr(res2d, "dotbracket", None)
    pdbs = [Path(p) for p in getattr(res3d, "all_atom_pdbs", [])]
    if not pdbs:
        p0 = getattr(res3d, "all_atom_pdb", None)
        if p0:
            pdbs = [Path(p0)]

    if not pdbs:
        raise RuntimeError("SimRNA produced no exported PDB frames.")

    frame_scores: List[float] = []
    frame_best: List[Dict] = []

    for fi, receptor_pdb in enumerate(pdbs, start=1):
        best_sc = 1e9
        best_item: Optional[Dict] = None

        for li, lig in enumerate(ligand_sdfs, start=1):
            dock_dir = work_root / "dock" / tag / f"frame_{fi:03d}" / f"lig_{li:03d}"
            dres = dock_one_compat(
                name=f"{tag}_f{fi:03d}_l{li:03d}",
                receptor_pdb=receptor_pdb,
                ligand_sdf=lig,
                workdir=dock_dir,
                padding=float(eval_cfg.vina_padding),
                exhaustiveness=int(eval_cfg.vina_exhaustiveness),
                cpu=int(eval_cfg.vina_cpu),
            )
            sc = float(dres.affinity_kcal)
            if sc < best_sc:
                best_sc = sc
                best_item = {
                    "frame": fi,
                    "ligand_index": li,
                    "ligand_sdf": str(lig),
                    "receptor_pdb": str(receptor_pdb),
                    "dock_pose_pdbqt": str(dres.out_pdbqt),
                    "dock_log": str(dres.log_file),
                    "dock_dir": str(dock_dir),
                }

        frame_scores.append(best_sc)
        frame_best.append(best_item or {"frame": fi, "ligand_index": "", "ligand_sdf": "", "receptor_pdb": str(receptor_pdb)})

    agg = aggregate_scores(frame_scores, method=str(eval_cfg.decoy_agg), boltz_beta=1.0)
    best_frame_idx = int(np.argmin(np.array(frame_scores, dtype=float))) if frame_scores else 0
    best_frame_info = frame_best[best_frame_idx] if frame_best else {}

    rec = {
        "seq": seq,
        "eval_hash": sha1(seq),
        "dot_bracket": dotb,
        "score": float(agg["score"]),
        "decoy_agg": str(eval_cfg.decoy_agg),
        "n_frames": int(len(frame_scores)),
        "score_min": float(agg["score_min"]),
        "score_mean": float(agg["score_mean"]),
        "score_p10": float(agg["score_p10"]),
        "score_boltz": float(agg["score_boltz"]),
        "best_frame": int(best_frame_info.get("frame", best_frame_idx + 1)) if best_frame_info else (best_frame_idx + 1),
        "best_score": float(frame_scores[best_frame_idx]) if frame_scores else float(agg["score"]),
        "best_receptor_pdb": str(best_frame_info.get("receptor_pdb", "")),
        "best_dock_pose_pdbqt": str(best_frame_info.get("dock_pose_pdbqt", "")),
        "best_dock_log": str(best_frame_info.get("dock_log", "")),
        "best_dock_dir": str(best_frame_info.get("dock_dir", "")),
        "best_ligand_sdf": str(best_frame_info.get("ligand_sdf", "")),
        "fold_dir": str(fold_dir),
        "error": "",
    }
    return rec


def evaluate_sequence_safe(
    seq: str,
    ligand_sdfs: List[Path],
    work_root: Path,
    eval_cfg: EvalConfig,
    seed: int,
) -> Dict:
    try:
        return evaluate_sequence(seq, ligand_sdfs, work_root, eval_cfg, seed)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        tag = sha1(seq)
        return {
            "seq": seq,
            "eval_hash": sha1(seq),
            "dot_bracket": None,
            "score": 1e9,
            "decoy_agg": getattr(eval_cfg, "decoy_agg", "min"),
            "n_frames": 0,
            "score_min": 1e9,
            "score_mean": 1e9,
            "score_p10": 1e9,
            "score_boltz": 1e9,
            "best_frame": "",
            "best_score": 1e9,
            "best_receptor_pdb": "",
            "best_dock_pose_pdbqt": "",
            "best_dock_log": "",
            "best_dock_dir": "",
            "best_ligand_sdf": "",
            "fold_dir": str((work_root / "fold" / tag).resolve()),
            "error": repr(e),
        }


# -----------------------------
# Parallel batch evaluation
# -----------------------------
def _worker_init_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _worker_task(args: Tuple[str, List[str], str, EvalConfig, int]) -> Dict:
    seq, ligand_sdfs, work_root, eval_cfg, seed = args
    lig_paths = [Path(x) for x in ligand_sdfs]
    return evaluate_sequence_safe(seq, lig_paths, Path(work_root), eval_cfg, seed)


def evaluate_batch_parallel(
    seqs: List[str],
    ligand_sdfs: List[Path],
    run_dir: Path,
    eval_cfg: EvalConfig,
    global_seed: int,
    iter_idx: int,
    workers: int,
    logger,
) -> List[Dict]:
    tasks = []
    ligand_sdfs_s = [str(p) for p in ligand_sdfs]
    for s in seqs:
        seed = stable_int_seed(global_seed, s, iter_idx)
        tasks.append((s, ligand_sdfs_s, str(run_dir), eval_cfg, seed))

    if workers <= 1:
        return [_worker_task(t) for t in tasks]

    ctx = mp.get_context("spawn")
    recs: List[Dict] = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
        initializer=_worker_init_env,
    ) as ex:
        futs = [ex.submit(_worker_task, t) for t in tasks]
        for fu in as_completed(futs):
            rec = fu.result()
            recs.append(rec)
            logger.info(f"Finished eval: score={float(rec['score']):.3f} seq={rec['seq'][:12]}...")
    return recs


# -----------------------------
# Rescore export hook (Top-M)
# -----------------------------
def export_rescore_topM(
    run_dir: Path,
    iter_idx: int,
    ranked_records: List[Dict],
    topM: int,
    include_penalty: bool,
    logger,
) -> Path:
    out_dir = (run_dir / "rescore" / f"iter_{iter_idx:03d}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    picked: List[Dict] = []
    for rec in ranked_records:
        if len(picked) >= int(topM):
            break
        if (not include_penalty) and is_penalty(rec):
            continue
        if not rec.get("best_receptor_pdb") or not rec.get("best_dock_pose_pdbqt"):
            continue
        picked.append(rec)

    with manifest.open("w") as f:
        for k, rec in enumerate(picked, start=1):
            item_dir = out_dir / f"rank_{k:03d}_{rec.get('eval_hash','')}"
            item_dir.mkdir(parents=True, exist_ok=True)

            src_receptor = Path(rec["best_receptor_pdb"])
            src_pose = Path(rec["best_dock_pose_pdbqt"])
            src_log = Path(rec.get("best_dock_log", "")) if rec.get("best_dock_log") else None
            src_lig = Path(rec.get("best_ligand_sdf", "")) if rec.get("best_ligand_sdf") else None

            dst_receptor = item_dir / src_receptor.name if src_receptor.exists() else item_dir / "receptor.pdb"
            dst_pose = item_dir / src_pose.name if src_pose.exists() else item_dir / "pose.pdbqt"
            dst_log = item_dir / (src_log.name if src_log and src_log.exists() else "vina.log")
            dst_lig = item_dir / (src_lig.name if src_lig and src_lig.exists() else "ligand.sdf")

            if src_receptor.exists():
                shutil.copy2(src_receptor, dst_receptor)
            if src_pose.exists():
                shutil.copy2(src_pose, dst_pose)
            if src_log and src_log.exists():
                shutil.copy2(src_log, dst_log)
            if src_lig and src_lig.exists():
                shutil.copy2(src_lig, dst_lig)

            out = {
                "iter": iter_idx,
                "rank": k,
                "seq": rec["seq"],
                "score_vina": float(rec.get("score", 1e9)),
                "best_score_frame": float(rec.get("best_score", rec.get("score", 1e9))),
                "best_frame": rec.get("best_frame", ""),
                "decoy_agg": rec.get("decoy_agg", ""),
                "n_frames": rec.get("n_frames", 0),
                "dot_bracket": rec.get("dot_bracket", None),
                "receptor_pdb": str(dst_receptor),
                "pose_pdbqt": str(dst_pose),
                "vina_log": str(dst_log),
                "ligand_sdf": str(dst_lig),
                "source_best_dock_dir": rec.get("best_dock_dir", ""),
                "source_fold_dir": rec.get("fold_dir", ""),
                "error": rec.get("error", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info(f"Rescore export: wrote {len(picked)} items -> {manifest}")
    return manifest


# -----------------------------
# Meta end update (signature compatible)
# -----------------------------
def safe_update_meta_end(meta_path: Path, run_dir: Path, status: str, best: Optional[Dict]):
    sig = None
    try:
        sig = inspect.signature(update_meta_end)
    except Exception:
        sig = None

    if sig is None:
        try:
            update_meta_end(meta_path, status=status, best=best)
            return
        except Exception:
            update_meta_end(run_dir, status=status, best=best)
            return

    params = list(sig.parameters.keys())
    if len(params) >= 1 and params[0] in ["meta_path", "path"]:
        update_meta_end(meta_path, status=status, best=best)
        return
    if "run_dir" in params:
        update_meta_end(run_dir=run_dir, status=status, best=best)
        return
    try:
        update_meta_end(meta_path, status=status, best=best)
    except Exception:
        update_meta_end(run_dir, status=status, best=best)


# -----------------------------
# Post-run MD refinement (GROMACS) for top-X sequences
# -----------------------------
_GMX_HELP_CACHE = {}

def _gmx_supports(gmx: str, subcmd: str, flag: str) -> bool:
    key = (gmx, subcmd)
    if key not in _GMX_HELP_CACHE:
        try:
            cp = subprocess.run([gmx, subcmd, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, check=False)
            _GMX_HELP_CACHE[key] = cp.stdout or ""
        except Exception:
            _GMX_HELP_CACHE[key] = ""
    return flag in _GMX_HELP_CACHE[key]


def _entry_score(rec: Dict) -> float:
    for k in ("score", "agg_score", "dock_score", "vina_score", "best_score"):
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                pass
    return float("inf")


def _run_cmd(cmd: List[str], cwd: Path, logger, input_text: Optional[str] = None) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    logger.info(f"[md] $ (cd {cwd}) {' '.join(cmd)}")
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=(input_text.encode('utf-8') if input_text is not None else None),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Command not found: {cmd[0]}") from e

    out = p.stdout.decode('utf-8', errors='ignore')
    (cwd / "md_cmd.log").open("a", encoding="utf-8").write(out + "\n")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (rc={p.returncode}): {' '.join(cmd)}\n---\n{out}")


def _mdrun_cmd(gmx: str, deffnm: str, threads: int, gpu: bool) -> List[str]:
    cmd: List[str] = [gmx, "mdrun", "-deffnm", deffnm]
    ntmpi = 1
    cmd += ["-ntmpi", str(ntmpi)]
    if int(threads) > 1:
        cmd += ["-ntomp", str(int(threads))]
    cmd += ["-v"]
    cmd += ["-pin", "on"]
    if gpu:
        cmd += ["-nb", "gpu"]
    return cmd


def _run_md_analysis(gmx: str, workdir: str, logger: logging.Logger) -> None:
    workdir = Path(workdir)
    tpr = workdir / "md.tpr"
    xtc = workdir / "md.xtc"
    rmsd_xvg = workdir / "rmsd.xvg"
    rmsf_xvg = workdir / "rmsf.xvg"

    if not tpr.exists() or not xtc.exists():
        logger.warning("[md] md.tpr or md.xtc missing, skip analysis")
        return

    if not rmsd_xvg.exists():
        try:
            logger.info("[md] RMSD: calculating (RNA group index=1)")
            cmd = [gmx, "rms", "-s", "md.tpr", "-f", "md.xtc", "-o", "rmsd.xvg", "-tu", "ns"]
            _run_cmd(cmd, workdir, logger, input_text="1\n1\n")
        except Exception as e:
            logger.warning(f"[md] RMSD failed: {e}")
    else:
        logger.info("[md] RMSD: exists, skip")

    if not rmsf_xvg.exists():
        try:
            logger.info("[md] RMSF: calculating (RNA group index=1)")
            cmd = [gmx, "rmsf", "-s", "md.tpr", "-f", "md.xtc", "-o", "rmsf.xvg", "-res"]
            _run_cmd(cmd, workdir, logger, input_text="1\n")
        except Exception as e:
            logger.warning(f"[md] RMSF failed: {e}")
    else:
        logger.info("[md] RMSF: exists, skip")


def _sanitize_rna_pdb(
    in_pdb: Path,
    out_pdb: Path,
    *,
    strip_5prime_phosphate: bool = True,
    strip_3prime_phosphate: bool = False,
) -> None:
    phosphate_atoms = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}
    lines = in_pdb.read_text().splitlines(True)

    residues: list[tuple[str, str, str]] = []
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        chain = line[21].strip() or " "
        resseq = line[22:26].strip()
        icode = line[26].strip() or " "
        key = (chain, resseq, icode)
        if not residues or residues[-1] != key:
            residues.append(key)

    first = residues[0] if residues else None
    last = residues[-1] if residues else None

    def match_res(line: str, key) -> bool:
        if key is None:
            return False
        chain, resseq, icode = key
        return ((line[21].strip() or " ") == chain
                and line[22:26].strip() == resseq
                and (line[26].strip() or " ") == icode)

    out = []
    for line in lines:
        if (line.startswith("ATOM") or line.startswith("HETATM")):
            atom = line[12:16].strip()
            if strip_5prime_phosphate and match_res(line, first) and atom in phosphate_atoms:
                continue
            if strip_3prime_phosphate and match_res(line, last) and atom in phosphate_atoms:
                continue
        out.append(line)

    out_pdb.write_text("".join(out))


def run_md_one_target(
    workdir: Path,
    gmx: str,
    mdp_dir: Path,
    input_pdb: Path,
    forcefield: str,
    water: str,
    strip_5prime_phosphate: bool,
    strip_3prime_phosphate: bool,
    box_type: str,
    box_distance_nm: float,
    salt_conc_m: float,
    neutralize: bool,
    threads: int,
    gpu: bool,
    logger,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for fn in ("em.mdp", "ions.mdp", "nvt.mdp", "npt.mdp", "md.mdp"):
        src = mdp_dir / fn
        if not src.exists():
            raise RuntimeError(f"Missing MDP: {src}")
        shutil.copy2(src, workdir / fn)

    sanitized_pdb = workdir / "input_sanitized.pdb"
    _sanitize_rna_pdb(
        input_pdb,
        sanitized_pdb,
        strip_5prime_phosphate=strip_5prime_phosphate,
        strip_3prime_phosphate=strip_3prime_phosphate,
    )
    input_pdb = sanitized_pdb

    _run_cmd(
        [gmx, "pdb2gmx", "-f", str(input_pdb), "-o", "processed.gro", "-p", "topol.top",
         "-ff", forcefield, "-water", water, "-ignh"],
        cwd=workdir, logger=logger,
    )
    _run_cmd(
        [gmx, "editconf", "-f", "processed.gro", "-o", "boxed.gro", "-c", "-d", str(box_distance_nm), "-bt", box_type],
        cwd=workdir, logger=logger,
    )
    _run_cmd(
        [gmx, "solvate", "-cp", "boxed.gro", "-cs", "spc216.gro", "-o", "solv.gro", "-p", "topol.top"],
        cwd=workdir, logger=logger,
    )
    _run_cmd(
        [gmx, "grompp", "-f", "ions.mdp", "-c", "solv.gro", "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "2"],
        cwd=workdir, logger=logger,
    )
    genion_cmd = [gmx, "genion", "-s", "ions.tpr", "-o", "solv_ions.gro", "-p", "topol.top"]
    if neutralize:
        genion_cmd += ["-neutral"]
    if salt_conc_m and float(salt_conc_m) > 0:
        genion_cmd += ["-conc", str(float(salt_conc_m))]
    genion_cmd += ["-pname", "NA", "-nname", "CL"]
    _run_cmd(genion_cmd, cwd=workdir, logger=logger, input_text="SOL\n")

    _run_cmd(
        [gmx, "grompp", "-f", "em.mdp", "-c", "solv_ions.gro", "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "2"],
        cwd=workdir, logger=logger,
    )
    _run_cmd(_mdrun_cmd(gmx, "em", threads, gpu), cwd=workdir, logger=logger)

    _run_cmd(
        [gmx, "grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro", "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "2"],
        cwd=workdir, logger=logger,
    )
    _run_cmd(_mdrun_cmd(gmx, "nvt", threads, gpu), cwd=workdir, logger=logger)

    _run_cmd(
        [gmx, "grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro", "-t", "nvt.cpt", "-p", "topol.top", "-o", "npt.tpr", "-maxwarn", "2"],
        cwd=workdir, logger=logger,
    )
    _run_cmd(_mdrun_cmd(gmx, "npt", threads, gpu), cwd=workdir, logger=logger)

    _run_cmd(
        [gmx, "grompp", "-f", "md.mdp", "-c", "npt.gro", "-t", "npt.cpt", "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "2"],
        cwd=workdir, logger=logger,
    )
    _run_cmd(_mdrun_cmd(gmx, "md", threads, gpu), cwd=workdir, logger=logger)
    _run_md_analysis(gmx=gmx, workdir=workdir, logger=logger)


def run_md_postprocess_topX(run_dir: Path, cache: "DiskCache", md_cfg: MDConfig, logger) -> None:
    if not md_cfg.enabled:
        logger.info("[md] md.enabled=false; skip")
        return

    md_root = run_dir / "md"
    md_root.mkdir(parents=True, exist_ok=True)

    mdp_dir = Path(md_cfg.mdp_dir).expanduser().resolve()
    if not mdp_dir.exists():
        raise RuntimeError(f"mdp_dir not found: {mdp_dir}")

    items = list(cache.map.items())
    items.sort(key=lambda kv: _entry_score(kv[1]))
    top_items = items[: int(md_cfg.topX)]

    logger.info(f"[md] postprocess: global top-{len(top_items)} -> {md_root}")

    summary_rows: List[Dict] = []
    for rank, (seq, rec) in enumerate(top_items, start=1):
        score = _entry_score(rec)
        tag = sha1(seq)
        workdir = md_root / f"rank_{rank:03d}_{tag}"
        status = "ok"
        err = ""

        pdb_path = None
        for k in ("best_receptor_pdb", "receptor_pdb", "pdb", "best_pdb"):
            if k in rec and rec[k]:
                p = Path(str(rec[k])).expanduser()
                if not p.is_absolute():
                    p = (run_dir / p).resolve()
                if p.exists():
                    pdb_path = p
                    break
        if pdb_path is None:
            status = "skip"
            err = "no_pdb_in_cache"
        else:
            xtc_file = workdir / "md.xtc"
            gro_file = workdir / "md.gro"

            if md_cfg.skip_if_done and xtc_file.exists() and gro_file.exists():
                logger.info(f"[md] rank={rank:03d} MD exists, skip simulation, run analysis only")
                try:
                    _run_md_analysis(md_cfg.gmx, str(workdir), logger)
                except Exception as e:
                    logger.warning(f"[md] rank={rank:03d} analysis failed: {e}")
                status = "skip"
                err = "analysis_checked"
            else:
                try:
                    t0 = time.time()
                    run_md_one_target(
                        workdir=workdir,
                        gmx=md_cfg.gmx,
                        mdp_dir=mdp_dir,
                        input_pdb=pdb_path,
                        forcefield=md_cfg.forcefield,
                        water=md_cfg.water,
                        strip_5prime_phosphate=bool(md_cfg.strip_5prime_phosphate),
                        strip_3prime_phosphate=bool(md_cfg.strip_3prime_phosphate),
                        box_type=md_cfg.box_type,
                        box_distance_nm=float(md_cfg.box_distance_nm),
                        salt_conc_m=float(md_cfg.salt_conc_m),
                        neutralize=bool(md_cfg.neutralize),
                        threads=int(md_cfg.threads),
                        gpu=bool(md_cfg.gpu),
                        logger=logger,
                    )
                    dt = time.time() - t0
                    logger.info(f"[md] rank={rank:03d} done in {dt:.1f}s -> {workdir}")
                except Exception as e:
                    status = "fail"
                    err = str(e).splitlines()[0][:240]
                    logger.error(f"[md] rank={rank:03d} FAILED: {e}")

                    summary_rows.append({
                        "rank": rank,
                        "seq": seq,
                        "score": score,
                        "workdir": str(workdir),
                        "status": status,
                        "error": err,
                    })

                write_csv(md_root / "md_summary.csv", summary_rows, fieldnames=["rank", "seq", "score", "workdir", "status", "error"])
                logger.info(f"[md] summary -> {md_root / 'md_summary.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--md_only", action="store_true", help="Only run MD postprocess on global top-X and exit.")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    cfg = default_cfg()
    if args.config:
        deep_update(cfg, load_yaml(args.config))
    if args.iters is not None:
        cfg.setdefault("ce", {})
        cfg["ce"]["iters"] = int(args.iters)
    for k, v in parse_set_kv(args.set):
        set_by_dotted_key(cfg, k, v)

    run_dir = Path(cfg["run"]["run_dir"]).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    write_resolved_config(run_dir, cfg)
    logger = setup_logging(run_dir=run_dir, name="ce_design", level=str(cfg["run"].get("log_level", "INFO")))

    # --- md-only mode ---
    if args.md_only:
        md_block = cfg.get("md", {})
        md_cfg = MDConfig(
            enabled=bool(md_block.get("enabled", False)),
            topX=int(md_block.get("topX", 5)),
            gmx=str(md_block.get("gmx", "gmx")),
            mdp_dir=str(md_block.get("mdp_dir", "mdp")),
            forcefield=str(md_block.get("forcefield", "amber14sb")),
            water=str(md_block.get("water", "tip3p")),
            box_type=str(md_block.get("box_type", "cubic")),
            box_distance_nm=float(md_block.get("box_distance_nm", 1.0)),
            salt_conc_m=float(md_block.get("salt_conc_m", 0.15)),
            neutralize=bool(md_block.get("neutralize", True)),
            threads=int(md_block.get("threads", 0)),
            gpu=bool(md_block.get("gpu", False)),
            skip_if_done=bool(md_block.get("skip_if_done", True)),
        )

        if not md_cfg.enabled:
            raise SystemExit("md_only set but md.enabled is false.")

        cache_path = run_dir / "cache.jsonl"
        if not cache_path.exists():
            raise SystemExit(f"md_only requires existing cache.jsonl: {cache_path}")

        cache = DiskCache(cache_path)
        logger.info(f"[md_only] cache entries: {len(cache.map)}")
        logger.info(f"[md_only] running MD for global top-{int(md_cfg.topX)}")

        run_md_postprocess_topX(run_dir=run_dir, cache=cache, md_cfg=md_cfg, logger=logger)
        logger.info("[md_only] DONE")
        return

    ligand_file = locate_ligand(cfg)

    meta_path = write_meta(run_dir=run_dir, cfg=cfg, ligand_file=ligand_file, status="running")
    logger.info(f"Meta: {meta_path}")
    t0 = time.time()

    try:
        workers = int(cfg["run"].get("workers", 1))
        global_seed = int(cfg["run"].get("seed", 123))
        resume = bool(cfg["run"].get("resume", True))

        eval_cfg = EvalConfig(
            omp_threads=int(cfg["eval"].get("omp_threads", 1)),
            simrna_iterations=int(cfg["eval"]["simrna_iterations"]),
            simrna_frames=str(cfg["eval"].get("simrna_frames", "1")),
            decoy_agg=str(cfg["eval"].get("decoy_agg", "min")),
            vina_exhaustiveness=int(cfg["eval"]["vina_exhaustiveness"]),
            vina_padding=float(cfg["eval"]["grid_padding"]),
            vina_cpu=int(cfg["eval"].get("vina_cpu", 1)),
            ligand_conformers=int(cfg["eval"].get("ligand_conformers", 0)),
            ligand_seed=int(cfg["eval"].get("ligand_seed", 123)),
            ligand_prune_rmsd=float(cfg["eval"].get("ligand_prune_rmsd", 0.5)),
            ligand_forcefield=str(cfg["eval"].get("ligand_forcefield", "MMFF")),
            ligand_optimize=bool(cfg["eval"].get("ligand_optimize", True)),
            ligand_max_opt_iters=int(cfg["eval"].get("ligand_max_opt_iters", 200)),
        )

        if workers > 1 and eval_cfg.omp_threads != 1:
            logger.info("workers>1: force eval.omp_threads=1 to avoid oversubscription")
            eval_cfg.omp_threads = 1

        # Ligand ensemble: reuse existing if present (preserves user-set charge)
        ligand_sdfs, ligand_manifest = prepare_ligand_ensemble_if_needed(ligand_file, run_dir, eval_cfg, logger)

        ce_cfg = CEConfig(
            length=int(cfg["ce"]["length"]),
            alphabet=str(cfg["ce"].get("alphabet", "RNA")),
            batch_size=int(cfg["ce"]["batch"]),
            elite_frac=float(cfg["ce"]["elite_frac"]),
            smoothing_alpha=float(cfg["ce"]["alpha"]),
            temperature=float(cfg["ce"].get("temperature", 1.0)),
            min_prob=float(cfg["ce"].get("min_prob", 0.02)),
            seed=global_seed,
        )

        dist_path = run_dir / "dist.json"
        cache_path = run_dir / "cache.jsonl"
        history_csv = run_dir / "history.csv"
        top_csv = run_dir / "topN.csv"
        checkpoint = run_dir / "checkpoint.json"

        if (not resume) and dist_path.exists():
            dist_path.unlink(missing_ok=True)
        if (not resume) and cache_path.exists():
            cache_path.unlink(missing_ok=True)

        if dist_path.exists():
            dist = SequenceDistribution.from_file(dist_path)
            ce_cfg = dist.cfg
        else:
            dist = SequenceDistribution(ce_cfg)

        cache = DiskCache(cache_path)

        history: List[Dict] = []
        if history_csv.exists():
            with history_csv.open("r") as f:
                for row in csv.DictReader(f):
                    history.append(row)

        rng = np.random.default_rng(int(ce_cfg.seed))

        global_best: Optional[Dict] = None
        for rec in cache.map.values():
            try:
                sc = float(rec["score"])
            except Exception:
                continue
            if global_best is None or sc < float(global_best["score"]):
                global_best = rec

        history_fields = [
            "iter", "batch", "elite_n",
            "best_score", "best_seq", "mean_score",
            "penalty_frac",
            "dist_entropy_mean", "dist_maxp_mean", "dist_maxp_frac095",
            "elite_hamming_mean", "elite_hamming_min", "elite_unique_frac",
            "n_unique_dotbracket",
            "early_stop_streak", "collapse_streak", "penalty_streak",
            "stop_reason",
        ]
        top_fields = [
            "seq", "eval_hash", "score", "decoy_agg", "n_frames",
            "score_min", "score_mean", "score_p10", "score_boltz",
            "best_frame", "best_score",
            "best_receptor_pdb", "best_dock_pose_pdbqt", "best_dock_log", "best_dock_dir",
            "best_ligand_sdf",
            "dot_bracket", "fold_dir", "error",
        ]

        iters = int(cfg["ce"]["iters"])
        topN_keep = int(cfg["ce"].get("topN", 20))

        ck: Dict = {}
        start_iter = 1
        if resume and checkpoint.exists():
            try:
                ck = json.loads(checkpoint.read_text())
                start_iter = int(ck.get("iter_completed", 0)) + 1
            except Exception:
                ck = {}
                start_iter = 1

        if start_iter > 1 and history:
            try:
                history = [r for r in history if int(r.get("iter", 0)) < start_iter]
            except Exception:
                pass

        if start_iter > iters:
            logger.info(
                f"Nothing to do: checkpoint iter_completed={start_iter-1} >= target iters={iters} "
                f"(run_dir={run_dir})"
            )
            safe_update_meta_end(Path(meta_path), run_dir, status="done", best=global_best)
            return

        reg = cfg.get("regularize", {})
        entropy_mix = float(reg.get("entropy_uniform_mix", 0.0))
        elite_min_ham = float(reg.get("elite_min_hamming", 0.0))
        struct_unique = bool(reg.get("struct_unique", False))
        struct_metric = str(reg.get("struct_metric", "bp"))
        struct_min_dist = float(reg.get("struct_min_dist", 0.0))

        mon = cfg.get("monitor", {})
        pen_cfg = mon.get("penalty_skip", {})
        pen_enabled = bool(pen_cfg.get("enabled", True))
        pen_thr = float(pen_cfg.get("frac_thr", 0.95))
        pen_action = str(pen_cfg.get("action", "skip")).lower()
        pen_reset_pat = int(pen_cfg.get("reset_patience", 2))

        es_cfg = mon.get("early_stop", {})
        es_enabled = bool(es_cfg.get("enabled", False))
        es_pat = int(es_cfg.get("patience", 6))
        es_min_improve = float(es_cfg.get("min_improve", 0.05))

        col_cfg = mon.get("collapse", {})
        col_enabled = bool(col_cfg.get("enabled", False))
        col_max_prob = float(col_cfg.get("max_prob", 0.95))
        col_frac_thr = float(col_cfg.get("frac_thr", 0.60))
        col_pat = int(col_cfg.get("patience", 4))

        # -------- rescore configuration --------
        rs_cfg = cfg.get("rescore", {})
        rs_enabled = bool(rs_cfg.get("enabled", False))
        rs_include_penalty = bool(rs_cfg.get("include_penalty", False))

        batch_size = int(cfg["ce"]["batch"])

        top_frac = rs_cfg.get("top_frac", None)
        if top_frac is None:
            raise ValueError("rescore.top_frac must be defined in YAML (ratio-based rescoring is required).")

        rs_topM = max(1, int(math.ceil(batch_size * float(top_frac))))

        logger.info(
            f"[rescore] enabled={rs_enabled}, "
            f"top_frac={top_frac}, batch={batch_size}, "
            f"computed topM={rs_topM}"
        )

        md_block = cfg.get("md", {})
        md_cfg = MDConfig(
            enabled=bool(md_block.get("enabled", False)),
            topX=int(md_block.get("topX", 5)),
            gmx=str(md_block.get("gmx", "gmx")),
            mdp_dir=str(md_block.get("mdp_dir", "mdp")),
            forcefield=str(md_block.get("forcefield", "amber14sb")),
            water=str(md_block.get("water", "tip3p")),
            box_type=str(md_block.get("box_type", "cubic")),
            box_distance_nm=float(md_block.get("box_distance_nm", 1.0)),
            salt_conc_m=float(md_block.get("salt_conc_m", 0.15)),
            neutralize=bool(md_block.get("neutralize", True)),
            threads=int(md_block.get("threads", 0)),
            gpu=bool(md_block.get("gpu", False)),
            skip_if_done=bool(md_block.get("skip_if_done", True)),
        )

        early_streak = int(ck.get("early_streak", 0)) if isinstance(ck, dict) else 0
        collapse_streak = int(ck.get("collapse_streak", 0)) if isinstance(ck, dict) else 0
        penalty_streak = int(ck.get("penalty_streak", 0)) if isinstance(ck, dict) else 0

        if isinstance(ck, dict) and ck.get("best_prev", None) is not None:
            try:
                best_prev = float(ck.get("best_prev"))
            except Exception:
                best_prev = float(global_best["score"]) if global_best else 1e9
        else:
            best_prev = float(global_best["score"]) if global_best else 1e9

        stop_reason = str(ck.get("stop_reason", "")) if isinstance(ck, dict) else ""

        for iter_idx in range(start_iter, iters + 1):
            seqs = dist.sample(int(ce_cfg.batch_size), rng)

            seen = set()
            uniq = []
            for s in seqs:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)

            records: List[Dict] = []
            todo: List[str] = []
            for s in uniq:
                c = cache.get(s)
                if c is not None:
                    records.append(c)
                else:
                    todo.append(s)

            logger.info(f"[iter {iter_idx}] batch={len(uniq)} cached={len(records)} todo={len(todo)} workers={workers}")

            if todo:
                new_recs = evaluate_batch_parallel(
                    seqs=todo,
                    ligand_sdfs=ligand_sdfs,
                    run_dir=run_dir,
                    eval_cfg=eval_cfg,
                    global_seed=global_seed,
                    iter_idx=iter_idx,
                    workers=workers,
                    logger=logger,
                )
                for r in new_recs:
                    cache.put(r)
                records.extend(new_recs)

            records.sort(key=lambda x: float(x["score"]))
            mean_score = float(np.mean([float(r["score"]) for r in records])) if records else 1e9
            best = records[0] if records else {"seq": "", "score": 1e9}

            if global_best is None or float(best["score"]) < float(global_best["score"]):
                global_best = best

            penalty_frac = float(np.mean([1.0 if is_penalty(r) else 0.0 for r in records])) if records else 1.0

            ds = dist_stats(dist)

            elite_n = max(1, int(math.ceil(float(ce_cfg.elite_frac) * len(records)))) if records else 1

            elites, elite_metrics, stage_log = select_elites_with_constraints(
                sorted_records=records,
                elite_n=elite_n,
                min_hamming=elite_min_ham,
                struct_unique=struct_unique,
                struct_metric=struct_metric,
                struct_min_dist=struct_min_dist,
                logger=logger,
            )
            elite_seqs = [e["seq"] for e in elites]

            did_update = True
            if pen_enabled and penalty_frac >= pen_thr:
                penalty_streak += 1
                did_update = False
                logger.info(f"[iter {iter_idx}] penalty_frac={penalty_frac:.3f} >= {pen_thr:.3f} -> SKIP CE UPDATE (streak={penalty_streak})")

                if pen_action == "reset_uniform" and penalty_streak >= pen_reset_pat:
                    P = get_probs(dist)
                    L, K = P.shape
                    U = np.ones((L, K), dtype=float) / float(K)
                    set_probs(dist, U)
                    logger.info(f"[iter {iter_idx}] penalty_streak >= {pen_reset_pat} -> RESET distribution to uniform")
                    penalty_streak = 0
            else:
                penalty_streak = 0

            if did_update:
                dist.update_from_elites(elite_seqs)
                if entropy_mix > 0:
                    apply_uniform_mix_entropy_regularization(dist, mix=entropy_mix)

            dist.save(dist_path)

            ds2 = dist_stats(dist)
            maxp_frac095 = float(ds2["dist_maxp_frac095"])
            if col_enabled:
                Pnow = get_probs(dist)
                maxp_i = np.max(Pnow, axis=1)
                frac = float(np.mean(maxp_i >= col_max_prob))
                if frac >= col_frac_thr:
                    collapse_streak += 1
                else:
                    collapse_streak = 0
            else:
                collapse_streak = 0

            if es_enabled and global_best is not None:
                best_now = float(global_best["score"])
                improve = float(best_prev - best_now)
                if improve < es_min_improve:
                    early_streak += 1
                else:
                    early_streak = 0
                    best_prev = best_now
            else:
                early_streak = 0

            stop_reason = ""
            if es_enabled and early_streak >= es_pat:
                stop_reason = f"early_stop(patience={es_pat}, min_improve={es_min_improve})"
            if col_enabled and collapse_streak >= col_pat:
                stop_reason = f"collapse(patience={col_pat}, frac_thr={col_frac_thr}, max_prob={col_max_prob})"

            history.append(
                {
                    "iter": str(iter_idx),
                    "batch": str(len(records)),
                    "elite_n": str(elite_n),
                    "best_score": f"{float(best['score']):.3f}",
                    "best_seq": best.get("seq", ""),
                    "mean_score": f"{mean_score:.3f}",
                    "penalty_frac": f"{penalty_frac:.3f}",
                    "dist_entropy_mean": f"{float(ds2['dist_entropy_mean']):.4f}",
                    "dist_maxp_mean": f"{float(ds2['dist_maxp_mean']):.4f}",
                    "dist_maxp_frac095": f"{float(ds2['dist_maxp_frac095']):.4f}",
                    "elite_hamming_mean": f"{float(elite_metrics.get('elite_hamming_mean', 0.0)):.4f}",
                    "elite_hamming_min": f"{float(elite_metrics.get('elite_hamming_min', 0.0)):.4f}",
                    "elite_unique_frac": f"{float(elite_metrics.get('elite_unique_frac', 1.0)):.4f}",
                    "n_unique_dotbracket": str(int(elite_metrics.get("n_unique_dotbracket", 0.0))),
                    "early_stop_streak": str(early_streak),
                    "collapse_streak": str(collapse_streak),
                    "penalty_streak": str(penalty_streak),
                    "stop_reason": stop_reason,
                }
            )
            write_csv(history_csv, history, history_fields)

            all_recs = list(cache.map.values())
            all_recs.sort(key=lambda x: float(x["score"]))
            write_csv(top_csv, all_recs[:topN_keep], top_fields)

            if rs_enabled:
                export_rescore_topM(
                    run_dir=run_dir,
                    iter_idx=iter_idx,
                    ranked_records=all_recs,
                    topM=rs_topM,
                    include_penalty=rs_include_penalty,
                    logger=logger,
                )

            checkpoint.write_text(
                json.dumps(
                    {
                        "iter_completed": iter_idx,
                        "run_dir": str(run_dir),
                        "ligand": str(ligand_file),
                        "ligand_manifest": str(ligand_manifest) if ligand_manifest else "",
                        "best": global_best,
                        "best_prev": float(best_prev),
                        "early_streak": int(early_streak),
                        "collapse_streak": int(collapse_streak),
                        "penalty_streak": int(penalty_streak),
                        "stop_reason": stop_reason,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

            logger.info(
                f"[iter {iter_idx}] best={float(best['score']):.3f} mean={mean_score:.3f} elite={elite_n} "
                f"penalty_frac={penalty_frac:.3f} global_best={float(global_best['score']):.3f}"
            )

            if stop_reason:
                logger.info(f"STOP: {stop_reason}")
                break

        logger.info("DONE")

        if getattr(md_cfg, "enabled", False):
            try:
                logger.info(f"[md] enabled: running MD for global top-{int(md_cfg.topX)} sequences")
                run_md_postprocess_topX(run_dir=run_dir, cache=cache, md_cfg=md_cfg, logger=logger)
            except Exception as e:
                logger.error(f"[md] postprocess failed: {repr(e)}", exc_info=True)

        if global_best:
            logger.info(f"Global best score: {float(global_best['score'])}")
            logger.info(f"Global best seq: {global_best['seq']}")
            logger.info(f"Artifacts: {run_dir}")

        safe_update_meta_end(Path(meta_path), run_dir, status="done", best=global_best)

    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        safe_update_meta_end(Path(meta_path), run_dir, status="interrupted", best=None)
        raise
    except Exception as e:
        logger.error("Run failed with exception.")
        logger.error(repr(e), exc_info=True)
        safe_update_meta_end(Path(meta_path), run_dir, status="failed", best=None)
        raise
    finally:
        elapsed = time.time() - t0
        logger.info(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
