from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

# 你的 CE 实现：SequenceDistribution + CEConfig
from optimizers.ce import CEConfig, SequenceDistribution  # :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}


@dataclass
class GbsaRow:
    seq: str
    score: float
    ok: bool
    source: str


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            yield r


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, object]]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _as_bool(v: object, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "ok"):
        return True
    if s in ("false", "0", "no", "n", "bad"):
        return False
    return default


def _as_float(v: object) -> Optional[float]:
    try:
        x = float(v)  # type: ignore[arg-type]
        if np.isfinite(x):
            return float(x)
        return None
    except Exception:
        return None


def _pick_score(row: Dict[str, object]) -> Tuple[Optional[str], Optional[float], bool]:
    """
    兼容多种字段：
      1) 真实 MMGBSA: mmgbsa_dg_bind (+ mmgbsa_ok)
      2) stub:        stub_mmgbsa
      3) fallback:    dock_score / score
    """
    if "mmgbsa_dg_bind" in row:
        sc = _as_float(row.get("mmgbsa_dg_bind"))
        ok = _as_bool(row.get("mmgbsa_ok"), default=True)
        return "mmgbsa_dg_bind", sc, ok

    if "stub_mmgbsa" in row:
        sc = _as_float(row.get("stub_mmgbsa"))
        # stub 默认当作 ok
        ok = _as_bool(row.get("mmgbsa_ok"), default=True)
        return "stub_mmgbsa", sc, ok

    if "dock_score" in row:
        sc = _as_float(row.get("dock_score"))
        ok = _as_bool(row.get("ok"), default=True)
        return "dock_score", sc, ok

    if "score" in row:
        sc = _as_float(row.get("score"))
        ok = _as_bool(row.get("ok"), default=True)
        return "score", sc, ok

    return None, None, True


def load_results(results_path: Path) -> List[GbsaRow]:
    if not results_path.exists():
        return []

    rows: List[GbsaRow] = []

    if results_path.suffix.lower() == ".csv":
        it = _iter_csv_rows(results_path)
    else:
        it = _iter_jsonl_rows(results_path)

    for r in it:
        # 统一把 key/value 当 object
        rr: Dict[str, object] = dict(r)  # type: ignore[arg-type]

        seq = (rr.get("seq") or rr.get("sequence") or rr.get("SEQ") or "").__str__().strip().upper()
        seq = seq.replace("T", "U")
        if not seq:
            continue

        col, sc, ok = _pick_score(rr)
        if col is None or sc is None:
            continue

        rows.append(GbsaRow(seq=seq, score=float(sc), ok=bool(ok), source=col))

    return rows


def choose_elites(rows: List[GbsaRow], elite_n: int, require_ok: bool, minimize: bool) -> List[GbsaRow]:
    cand = [x for x in rows if (x.ok or not require_ok)]
    if not cand:
        return []
    cand.sort(key=lambda x: x.score, reverse=(not minimize))
    elite_n = max(1, min(elite_n, len(cand)))
    return cand[:elite_n]


def _load_or_init_dist(dist_path: Path, seq_len: int, batch: int, seed: int,
                       smoothing_alpha: float, min_prob: float, temperature: float) -> SequenceDistribution:
    if dist_path.exists():
        return SequenceDistribution.from_file(dist_path)  # :contentReference[oaicite:3]{index=3}

    cfg = CEConfig(
        length=seq_len,
        alphabet="RNA",
        batch_size=batch,
        elite_frac=0.1,
        smoothing_alpha=smoothing_alpha,
        temperature=temperature,
        min_prob=min_prob,
        seed=seed,
    )
    return SequenceDistribution(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/ce_run1")
    ap.add_argument("--iter", type=int, required=True, help="iteration index, e.g. 1 -> iter_001")
    ap.add_argument("--elite_n", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--require_ok", action="store_true", help="only use rows with ok=True / mmgbsa_ok=True")
    ap.add_argument("--minimize", action="store_true", default=True, help="smaller score is better (default)")
    ap.add_argument("--alpha", type=float, default=0.4, help="CE smoothing_alpha")
    ap.add_argument("--min_prob", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--results_path", default="", help="override results file path (csv/jsonl)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    iter_dir = run_dir / "rescore" / f"iter_{args.iter:03d}"
    dist_path = run_dir / "dist.json"

    # 默认优先读 results.csv，不存在则读 results.jsonl
    if args.results_path:
        results_path = Path(args.results_path)
    else:
        results_path = iter_dir / "results.csv"
        if not results_path.exists():
            results_path = iter_dir / "results.jsonl"

    rows = load_results(results_path)
    if not rows:
        raise RuntimeError(
            f"[ERR] No usable rows parsed from {results_path}.\n"
            f"      Check file exists and has one of score columns: mmgbsa_dg_bind / stub_mmgbsa / dock_score.\n"
            f"      You can run: head -n 3 {results_path}"
        )

    elites = choose_elites(rows, elite_n=args.elite_n, require_ok=args.require_ok, minimize=args.minimize)
    if not elites:
        # 给出明确诊断，不再 IndexError
        cols_hint = sorted({r.source for r in rows})
        raise RuntimeError(
            f"[ERR] elites is empty after filtering (require_ok={args.require_ok}).\n"
            f"      Parsed rows={len(rows)}, parsed score sources={cols_hint}.\n"
            f"      Suggest: re-run without --require_ok, or ensure mmgbsa_ok/ok is True."
        )

    #  elites
    elites_path = iter_dir / "elites_for_ce.txt"
    elites_path.write_text("".join([e.seq + "\n" for e in elites]))

    # Update the distribution and sample the next batch
    seq_len = len(elites[0].seq)
    dist = _load_or_init_dist(
        dist_path=dist_path,
        seq_len=seq_len,
        batch=args.batch,
        seed=args.seed,
        smoothing_alpha=args.alpha,
        min_prob=args.min_prob,
        temperature=args.temperature,
    )

    dist.update_from_elites([e.seq for e in elites])  # :contentReference[oaicite:4]{index=4}
    dist.save(dist_path)  # :contentReference[oaicite:5]{index=5}

    rng = np.random.default_rng(args.seed)
    next_seqs = dist.sample(args.batch, rng)  # :contentReference[oaicite:6]{index=6}

    next_batch_path = run_dir / "next_batch.txt"
    next_batch_path.write_text("".join([s + "\n" for s in next_seqs]))

    # Print Summary
    best = elites[0]
    print(f"[OK] read results: {results_path}")
    print(f"[OK] elites written: {elites_path} (n={len(elites)}) best={best.score:.4f} source={best.source}")
    print(f"[OK] dist updated:   {dist_path}")
    print(f"[OK] next batch:     {next_batch_path} (n={len(next_seqs)})")


if __name__ == "__main__":
    main()
