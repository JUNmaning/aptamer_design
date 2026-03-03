
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_run.py
- Plot CE history + MMGBSA trends (existing)
- NEW: plot MD RMSD / RMSF from runs/<run>/md/rank_*/(rmsd.xvg, rmsf.xvg)
- Publication style: CEJ / Elsevier-like (Times New Roman, bold ticks/labels, thick spines)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# CEJ / Elsevier-like publication style
# ============================================================
def apply_cej_style(font_family: str = "Times New Roman", base_fontsize: int = 20) -> None:
    plt.rcParams.update({
        "font.family": font_family,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize + 2,
        "axes.titlesize": base_fontsize + 3,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,

        "axes.linewidth": 1.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,

        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3.5,
        "ytick.minor.size": 3.5,

        "legend.frameon": True,
        "legend.framealpha": 0.95,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.55,
    })


def style_axes_cej(
    ax,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    spine_lw: float = 2.2,
    tick_lw: float = 2.0,
    add_minor: bool = True,
    bold: bool = True,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold" if bold else "normal")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold" if bold else "normal")

    ax.grid(False)

    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(spine_lw)

    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)
    ax.tick_params(axis="both", which="major", width=tick_lw)
    ax.tick_params(axis="both", which="minor", width=max(tick_lw - 0.5, 1.0))

    if add_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    if bold:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _annotate_empty(ax, title: str, msg: str) -> None:
    ax.axis("off")
    ax.text(
        0.5, 0.55, title,
        ha="center", va="center", fontsize=16, fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.40, msg,
        ha="center", va="center", fontsize=12,
        transform=ax.transAxes,
    )


def _safe_legend(ax, loc="best", frameon=False) -> None:
    try:
        leg = ax.legend(loc=loc, frameon=frameon)
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontweight("bold")
    except Exception:
        pass


# ------------------------------
# CE run plots
# ------------------------------
def read_history(run_dir: Path) -> Tuple[Optional[pd.DataFrame], str]:
    fp = run_dir / "history.csv"
    if not fp.exists():
        return None, f"Missing {fp}"

    try:
        df = pd.read_csv(fp)
    except Exception as e:
        return None, f"Failed to read {fp}: {e}"

    # Heuristic column names (your pipeline uses these)
    for c in ["iter", "best_score", "mean_score"]:
        if c not in df.columns:
            return None, f"history.csv missing column: {c}"

    df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
    df["best_score"] = pd.to_numeric(df["best_score"], errors="coerce")
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df = df.dropna(subset=["iter", "best_score", "mean_score"])
    if df.empty:
        return None, "history.csv has no numeric rows after parsing."
    df["iter"] = df["iter"].astype(int)
    return df, ""


def plot_history_best_mean(df: Optional[pd.DataFrame], err: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    if df is None or df.empty:
        _annotate_empty(ax, out_base.name + ".png", "No plottable history data.\n" + err)
        fig.savefig(str(out_base) + ".png", dpi=300)
        plt.close(fig)
        return

    x = df["iter"].to_numpy()
    ax.plot(x, df["best_score"].to_numpy(dtype=float), marker="o", label="Best (min) score")
    ax.plot(x, df["mean_score"].to_numpy(dtype=float), marker="s", label="Mean score")

    ax.set_xticks(x)
    style_axes_cej(ax, xlabel="Iteration", ylabel="Docking score")
    _safe_legend(ax, loc="best", frameon=False)

    fig.savefig(str(out_base) + ".png", dpi=300)
    fig.savefig(str(out_base) + ".pdf")
    fig.savefig(str(out_base) + ".tif", dpi=600)
    plt.close(fig)


def plot_history_penalty(df: Optional[pd.DataFrame], err: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    if df is None or df.empty:
        _annotate_empty(ax, out_base.name + ".png", "No plottable history data.\n" + err)
        fig.savefig(str(out_base) + ".png", dpi=300)
        plt.close(fig)
        return

    if "penalty_frac" not in df.columns:
        _annotate_empty(ax, out_base.name + ".png", "history.csv missing column: penalty_frac")
        fig.savefig(str(out_base) + ".png", dpi=300)
        plt.close(fig)
        return

    x = df["iter"].to_numpy()
    y = pd.to_numeric(df["penalty_frac"], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, y, marker="o", label="Penalty fraction")
    ax.set_xticks(x)
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(y) * 1.05)))

    style_axes_cej(ax, xlabel="Iteration", ylabel="Penalty fraction")
    _safe_legend(ax, loc="best", frameon=False)

    fig.savefig(str(out_base) + ".png", dpi=300)
    fig.savefig(str(out_base) + ".pdf")
    fig.savefig(str(out_base) + ".tif", dpi=600)
    plt.close(fig)


def read_mmgbsa_series(run_dir: Path, topk: int = 5) -> Tuple[Optional[pd.DataFrame], str]:
    pattern = str(run_dir / "rescore" / "iter_*" / "mmgbsa_results.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, f"No mmgbsa_results.csv found under {run_dir/'rescore'}/iter_*/"

    rows = []
    for fp in files:
        p = Path(fp)
        try:
            it = int(p.parent.name.split("_")[-1])
        except Exception:
            continue

        try:
            d = pd.read_csv(p)
        except Exception:
            continue

        if d.empty or "mmgbsa_dg_bind" not in d.columns:
            continue

        d["mmgbsa_dg_bind"] = pd.to_numeric(d["mmgbsa_dg_bind"], errors="coerce")
        d = d.dropna(subset=["mmgbsa_dg_bind"])
        if d.empty:
            continue

        d_sorted = d.sort_values("mmgbsa_dg_bind", ascending=True)
        best = float(d_sorted.iloc[0]["mmgbsa_dg_bind"])
        mean_topk = float(d_sorted.head(int(topk))["mmgbsa_dg_bind"].mean())
        rows.append({"iter": it, "best_dg": best, "mean_topk_dg": mean_topk})

    if not rows:
        return None, "mmgbsa_results.csv found, but mmgbsa_dg_bind could not be parsed."
    out = pd.DataFrame(rows).sort_values("iter")
    out["iter"] = out["iter"].astype(int)
    return out, ""


def plot_mmgbsa(df: Optional[pd.DataFrame], err: str, out_base: Path, topk: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    if df is None or df.empty:
        _annotate_empty(ax, out_base.name + ".png", "No plottable MMGBSA data.\n" + err)
        fig.savefig(str(out_base) + ".png", dpi=300)
        plt.close(fig)
        return

    x = df["iter"].to_numpy()
    ax.plot(x, df["best_dg"].to_numpy(dtype=float), marker="o", label="Best ΔG (lower is better)")
    ax.plot(x, df["mean_topk_dg"].to_numpy(dtype=float), marker="s", label=f"Mean top-{topk} ΔG")
    ax.set_xticks(x)

    style_axes_cej(ax, xlabel="Iteration", ylabel=r"MM/GBSA $\Delta G_\mathrm{bind}$ (kcal/mol)")
    _safe_legend(ax, loc="best", frameon=False)

    fig.savefig(str(out_base) + ".png", dpi=300)
    fig.savefig(str(out_base) + ".pdf")
    fig.savefig(str(out_base) + ".tif", dpi=600)
    plt.close(fig)


# ------------------------------
# MD RMSD / RMSF plots
# ------------------------------
def _load_xvg(fp: Path) -> np.ndarray:
    # GROMACS .xvg: comments begin with @ and #
    return np.loadtxt(fp, comments=("#", "@"))


def _moving_average(y: np.ndarray, window_pts: int) -> Optional[np.ndarray]:
    if window_pts <= 1 or y.size < window_pts + 1:
        return None
    kernel = np.ones(window_pts, dtype=float) / float(window_pts)
    return np.convolve(y, kernel, mode="valid")


def plot_rmsd_xvg(fp: Path, out_base: Path) -> None:
    data = _load_xvg(fp)
    if data.ndim != 2 or data.shape[1] < 2:
        return

    t = data[:, 0].astype(float)       # with -tu ns in pipeline, this is ns
    rmsd_nm = data[:, 1].astype(float)
    rmsd_A = rmsd_nm * 10.0

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(t, rmsd_A, linewidth=2.0, alpha=0.90, label="RMSD")

    # 1 ns moving average (only if we have enough points)
    if t.size >= 3:
        dt = float(np.mean(np.diff(t)))
        if dt > 0:
            win = max(1, int(round(1.0 / dt)))
            ma = _moving_average(rmsd_A, win)
            if ma is not None:
                ax.plot(t[win - 1:], ma, linewidth=2.0, alpha=0.95, label="1 ns Moving Avg")

    ax.set_xlim(left=0.0, right=float(np.max(t)))
    # Let y autoscale but keep floor at 0
    ax.set_ylim(bottom=0.0)

    style_axes_cej(ax, xlabel="Time (ns)", ylabel=r"RMSD ($\mathrm{\AA}$)")
    _safe_legend(ax, loc="best", frameon=False)

    fig.savefig(str(out_base) + ".png", dpi=300)
    fig.savefig(str(out_base) + ".pdf")
    fig.savefig(str(out_base) + ".tif", dpi=600)
    plt.close(fig)


def plot_rmsf_xvg(fp: Path, out_base: Path) -> None:
    data = _load_xvg(fp)
    if data.ndim != 2 or data.shape[1] < 2:
        return

    x = data[:, 0].astype(float)      # residue index (often integer)
    rmsf_nm = data[:, 1].astype(float)
    rmsf_A = rmsf_nm * 10.0

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot(x, rmsf_A, linewidth=2.0, alpha=0.92, label="RMSF")

    ax.set_xlim(left=float(np.min(x)), right=float(np.max(x)))
    ax.set_ylim(bottom=0.0)

    style_axes_cej(ax, xlabel="Residue index", ylabel=r"RMSF ($\mathrm{\AA}$)")
    _safe_legend(ax, loc="best", frameon=False)

    fig.savefig(str(out_base) + ".png", dpi=300)
    fig.savefig(str(out_base) + ".pdf")
    fig.savefig(str(out_base) + ".tif", dpi=600)
    plt.close(fig)


def _rank_id_from_dir(d: Path) -> str:
    # rank_001_xxxxx -> 001
    m = re.match(r"rank_(\d+)_", d.name)
    return m.group(1) if m else d.name


def plot_md_rmsd_rmsf(run_dir: Path, plots_dir: Path) -> List[Path]:
    """
    Look for run_dir/md/rank_*/rmsd.xvg and rmsf.xvg.
    Write plots into plots_dir/md/.
    Returns list of generated image paths.
    """
    out_paths: List[Path] = []
    md_dir = run_dir / "md"
    if not md_dir.exists():
        return out_paths

    md_plots = plots_dir / "md"
    _ensure_dir(md_plots)

    rank_dirs = sorted([p for p in md_dir.glob("rank_*") if p.is_dir()])
    for rd in rank_dirs:
        rid = _rank_id_from_dir(rd)

        rmsd = rd / "rmsd.xvg"
        if rmsd.exists():
            out_base = md_plots / f"rank_{rid}_rmsd"
            try:
                plot_rmsd_xvg(rmsd, out_base)
                out_paths.append(Path(str(out_base) + ".png"))
            except Exception:
                pass

        rmsf = rd / "rmsf.xvg"
        if rmsf.exists():
            out_base = md_plots / f"rank_{rid}_rmsf"
            try:
                plot_rmsf_xvg(rmsf, out_base)
                out_paths.append(Path(str(out_base) + ".png"))
            except Exception:
                pass

    return out_paths


# ------------------------------
# CLI
# ------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to runs/<run_name> (e.g., ~/aptamer_design/runs/ce_run1)")
    ap.add_argument("--topk_mmgbsa", type=int, default=5, help="Top-k for mean ΔG in mmgbsa_by_iter plot")
    ap.add_argument("--no_md_plots", action="store_true", help="Skip RMSD/RMSF plotting even if xvg exist")
    args = ap.parse_args()

    run_dir = Path(os.path.expanduser(args.run_dir)).resolve()
    plots_dir = run_dir / "plots"
    _ensure_dir(plots_dir)

    apply_cej_style(font_family="Times New Roman", base_fontsize=20)

    # 1) history
    hist, herr = read_history(run_dir)
    plot_history_best_mean(hist, herr, plots_dir / "history_best_mean")
    print(f"[OK] wrote: {plots_dir/'history_best_mean.png'}")

    plot_history_penalty(hist, herr, plots_dir / "history_penalty")
    print(f"[OK] wrote: {plots_dir/'history_penalty.png'}")

    # 2) mmgbsa by iter
    mm, merr = read_mmgbsa_series(run_dir, topk=args.topk_mmgbsa)
    plot_mmgbsa(mm, merr, plots_dir / "mmgbsa_by_iter", topk=args.topk_mmgbsa)
    print(f"[OK] wrote: {plots_dir/'mmgbsa_by_iter.png'}")

    # 3) MD rmsd/rmsf
    if not args.no_md_plots:
        outs = plot_md_rmsd_rmsf(run_dir, plots_dir)
        if outs:
            for p in outs:
                print(f"[OK] wrote: {p}")
        else:
            print("[INFO] no RMSD/RMSF xvg found under run_dir/md/rank_*/ (skipping).")


if __name__ == "__main__":
    main()
