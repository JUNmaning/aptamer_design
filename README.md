# Aptamer Design Pipeline (CE + Multi-frame SimRNA + Vina Docking + MMGBSA Feedback Loop + Optional MD)

Author: maning

This repository implements an end-to-end, reproducible RNA aptamer design pipeline driven by a Cross-Entropy (CE) optimizer.
At each CE iteration, the pipeline samples RNA sequences from a probability distribution, evaluates them via RNA structure sampling (SimRNA) and docking (AutoDock Vina) using multiple RNA decoy frames and multiple ligand conformers, then optionally runs MMGBSA rescoring on top candidates and feeds those scores back to update the CE distribution for the next iteration.

The workflow is restartable/resumable and supports cheap smoke tests before expensive rescoring.

---

## Table of Contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [Key outputs](#2-key-outputs-per-run-directory)
3. [Environment and dependencies](#3-environment-and-dependencies)
4. [Ligand charge handling (IMPORTANT)](#4-ligand-charge-handling-important)
5. [How to run (production)](#5-how-to-run-production)
6. [Quick Start (5-minute test)](#6-quick-start-5-minute-test)
7. [Configuration explained](#7-configuration-explained)
8. [Smoke test](#8-smoke-test)
9. [How to analyze results](#9-how-to-analyze-results)
10. [Common failure modes](#10-common-failure-modes)
11. [Visualization](#11-visualization)
12. [CE theory mapping](#12-ce-theory-mapping)
13. [Reproducibility notes](#13-reproducibility-notes)

---

## 1) What this pipeline does

Given a ligand structure (SDF/PDB), the pipeline performs:

1. Ligand ensemble generation (RDKit conformers with charge validation)
2. RNA structure sampling (ViennaRNA + SimRNA trajectory)
3. Multi-frame decoy export (SimRNA trafl to multiple all-atom PDBs)
4. Docking (Vina) across RNA frames x ligand conformers
5. Aggregation across frames/conformers (e.g. min)
6. CE update using docking elites
7. Optional MMGBSA rescoring for top candidates
8. Feedback update: MMGBSA results update dist.json
9. Resume to the next CE iteration until stopping criteria are met

---

## 2) Key outputs (per run directory)

A run directory (e.g. runs/ce_run1/) contains:

    runs/<RUN_NAME>/
    ├── config_resolved.json          # resolved configuration for reproducibility
    ├── meta.json                     # run metadata
    ├── logs/ce_design.log            # main log
    ├── checkpoint.json               # resume/stop state
    ├── dist.json                     # CE distribution (core state)
    ├── history.csv                   # per-iteration summary metrics
    ├── topN.csv                      # exported top ranked candidates
    ├── next_batch.txt                # next batch generated after GBSA feedback
    ├── ligand_ensemble/              # ligand conformers + charge info
    │   ├── manifest.json             # includes charge_info
    │   ├── charge.json               # standalone charge information
    │   ├── charge.txt                # simple charge value (e.g. -1)
    │   └── conf_*.sdf               # individual conformers with NET_CHARGE property
    ├── fold/<eval_hash>/             # SimRNA outputs per sequence
    │   ├── aptamer.trafl
    │   ├── aptamer-000001_AA.pdb
    │   └── ...
    ├── dock/<eval_hash>/frame_XXX/lig_YYY/
    │   ├── *.dock.pdbqt
    │   └── *.vina.log
    ├── rescore/iter_XXX/
    │   ├── manifest.jsonl            # candidates selected for MMGBSA
    │   ├── mmgbsa_results.csv        # raw MMGBSA results
    │   ├── results.csv               # normalized file for GBSA feedback
    │   ├── elites_for_ce.txt         # elites selected in GBSA feedback
    │   └── feedback.done             # marker for completed feedback
    └── md/                           # optional MD results
        ├── rank_00x_<eval_hash>/
        └── md_summary.csv

### File purpose summary

| File | Purpose | When to inspect |
|------|---------|----------------|
| dist.json | CE distribution state | Debug convergence issues |
| checkpoint.json | Resume state | Check current iteration |
| history.csv | Per-iteration metrics | Monitor training progress |
| topN.csv | Best docking candidates | Quick result inspection |
| cache.jsonl | Evaluation cache | Debug duplicate evaluations |
| next_batch.txt | Next sequences to evaluate | After GBSA feedback |
| ligand_ensemble/manifest.json | Ligand conformers + charge info | Verify charge detection |
| ligand_ensemble/charge.txt | Net charge (simple) | Quick charge check |
| rescore/iter_XXX/mmgbsa_results.csv | MMGBSA outputs | Final ranking |
| rescore/iter_XXX/elites_for_ce.txt | GBSA feedback elites | Verify feedback update |
| md/md_summary.csv | MD results summary | Post-run refinement |

Two files matter most for the feedback loop:
- dist.json: current CE distribution (updated each iteration)
- checkpoint.json: current iteration/termination state (controls resume)

---

## 3) Environment and dependencies

### 3.1 Recommended: mamba

This project includes an environment.yml.

    micromamba create -f environment.yml -y
    micromamba shell init -s bash -r /root/.local/share/mamba
    source ~/.bashrc
    micromamba activate aptamer_design

### 3.2 External tools required

The pipeline assumes these tools are available in PATH:

- SimRNA (including SimRNA and SimRNA_trafl2pdbs)
- AutoDock Vina (vina)
- AmberTools (tleap, antechamber, parmchk2, MMPBSA.py)
- OpenBabel (obabel)
- ViennaRNA (RNAfold)
- Optional MD: GROMACS (gmx)

Quick check:

    which RNAfold || echo "RNAfold not found"
    which SimRNA || echo "SimRNA not found"
    which SimRNA_trafl2pdbs || echo "SimRNA_trafl2pdbs not found"
    which obabel || echo "obabel not found"
    which vina || echo "vina not found"
    which tleap || echo "tleap not found"
    which MMPBSA.py || echo "MMPBSA.py not found"

### 3.3 Preflight check

    bash scripts/preflight.sh

---

## 4) Ligand charge handling (IMPORTANT)

### Why charge matters

Ligand charge critically affects:
- Antechamber parameterization (AM1-BCC charge assignment)
- MMGBSA binding free energy (electrostatics dominate)
- Docking pose selection (charged groups prefer polar pockets)

Wrong charge will cause all MMGBSA calculations to fail or give meaningless results.

### Automatic charge detection (priority order)

The pipeline auto-detects ligand charge from:

1. CLI override (--charge in ligand prep, or --net_charge in MMGBSA)
2. ligand_ensemble/manifest.json (charge_info.formal_charge) -- most reliable
3. ligand_ensemble/charge.json (standalone charge info)
4. ligand_ensemble/charge.txt (simple text file)
5. SDF NET_CHARGE property (embedded in conformer SDF files)
6. RDKit formal charge from original ligand file
7. PDBQT Gasteiger sum (last resort, unreliable)
8. Fallback to 0 (with warning)

### How to check your ligand charge

    python ligand/rdkit_conformers.py data/your_ligand.sdf --charge_only

Example output:

    Formal charge:    0
    Detection method: rdkit_formal
    Validation:       PASSED
    Formula:          C6H2F13O3P
    MW:               400.03

### Common ligand charge values at pH 7.4

- Phosphoric acid (-PO(OH)2): charge = -1 or -2
- ATP: charge = -3 or -4
- Neutral ligands: charge = 0
- Amines: charge = +1

### How to set ligand charge

Before running the main pipeline, prepare ligand conformers with the correct charge:

    cd ~/autodl-tmp/aptamer_design

    python ligand/rdkit_conformers.py data/PFHxS.sdf \
        --out_dir runs/ce_run1/ligand_ensemble \
        --n_confs 10 \
        --charge n     #Set it yourself, otherwise the small molecule file will be automatically read to determine the charge
   
    # Verify:
    cat runs/ce_run1/ligand_ensemble/charge.txt
    # Should output: n

This charge will be automatically used by all downstream steps including
CE docking evaluation, MMGBSA rescoring, and all analysis.

### Override charge during MMGBSA only

If you need to re-run MMGBSA with a different charge without regenerating conformers:

    python -m experiments.rescore_mmgbsa \
        --run_dir runs/ce_run1 \
        --iter 1 \
        --net_charge -2

---

## 5) How to run (production)

### Standard workflow 

    cd ~/autodl-tmp/aptamer_design

    # Step 1: Prepare ligand ensemble with correct charge
    python ligand/rdkit_conformers.py data/PFHxS.sdf \
        --out_dir runs/ce_run1/ligand_ensemble \
        --n_confs 10 \
        --charge -1

    # Step 2: Verify charge
    cat runs/ce_run1/ligand_ensemble/charge.txt

    # Step 3: Run the full CE + GBSA feedback loop
    bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml

### One-line command 

    python ligand/rdkit_conformers.py data/PFHxS.sdf \
        --out_dir runs/ce_run1/ligand_ensemble \
        --n_confs 10 \
        --charge -1 && \
    bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml

### What the loop does

The run_ce_gbsa_loop.sh script will:

1. Read checkpoint.json to determine the last completed iteration
2. Run CE to iter_completed + 1
3. Run MMGBSA rescoring for that iteration (auto-detects charge from ligand_ensemble/)
4. Run GBSA feedback to update dist.json (only uses mmgbsa_ok=True entries)
5. Repeat until:
   - ce.iters reached, OR
   - stop_reason in checkpoint.json becomes non-empty

### Resume after interruption

Just re-run the same command:

    bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml

The loop will:
1. Read checkpoint.json to find the last completed iteration
2. Skip iterations where feedback.done exists
3. Continue from the next incomplete iteration

### Restart from scratch

    rm -rf runs/ce_run1
    # Then re-run from Step 1

### Update ligand charge for an existing run

If you already ran some iterations and want to change the charge:

    cd ~/autodl-tmp/aptamer_design

    # 1. Update ligand ensemble
    python ligand/rdkit_conformers.py data/PFHxS.sdf \
        --out_dir runs/ce_run1/ligand_ensemble \
        --n_confs 10 \
        --charge -1

    # 2. Clear current iteration MMGBSA results
    rm -rf runs/ce_run1/rescore/iter_001/mmgbsa
    rm -f runs/ce_run1/rescore/iter_001/mmgbsa_results.*
    rm -f runs/ce_run1/rescore/iter_001/feedback.done

    # 3. Continue
    bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml

### Stopping criteria (automatic)

The loop stops when any of these conditions is met:

1. ce.iters reached (e.g. 50 iterations)
2. monitor.early_stop triggered (docking plateau)
3. monitor.collapse triggered (distribution collapsed)
4. External GBSA plateau (monitor.external_stop.gbsa_patience)

Check which criterion stopped the run:

    cat runs/ce_run1/checkpoint.json

---

## 6) Quick Start 
    cd ~/autodl-tmp/aptamer_design

    # 1. Prepare ligand
    python ligand/rdkit_conformers.py data/PFHxS.sdf \
        --out_dir runs/test/ligand_ensemble \
        --n_confs 2 \
        --charge -1

    # 2. Create minimal config
    cat > configs/test.yaml <<EOF
    run:
      run_dir: runs/test
      seed: 123
      workers: 1

    inputs:
      ligand: data/PFHxS.sdf

    ce:
      length: 30
      batch: 2
      iters: 1
      elite_frac: 0.5

    eval:
      simrna_iterations: 5000
      simrna_frames: "1"
      ligand_conformers: 2
      vina_exhaustiveness: 2

    rescore:
      enabled: false
    EOF

    # 3. Run
    python -m experiments.ce_design --config configs/test.yaml

    # 4. Check
    head -n 5 runs/test/topN.csv

---

## 7) Configuration explained

### 7.1 run
- run_dir: Output directory for all run artifacts
- seed: Global random seed for reproducibility
- workers: Number of parallel workers for sequence evaluation
- log_level: Logging verbosity (INFO, DEBUG, WARNING)
- resume: Whether to resume from existing checkpoint

### 7.2 inputs
- ligand: Path to ligand file (.sdf or .pdb)

### 7.3 ce
- length: RNA sequence length
- alphabet: Sequence alphabet (RNA)
- batch: Number of sequences sampled per CE iteration
- iters: Maximum number of CE iterations
- elite_frac: Fraction of batch used as CE elites
- alpha: CE update smoothing factor
- min_prob: Minimum probability floor per position
- temperature: Sampling temperature
- topN: Number of top candidates exported to topN.csv (for inspection only)

### 7.4 eval
- simrna_config: Path to SimRNA configuration file
- omp_threads: OpenMP threads for folding (keep 1 when workers > 1)
- simrna_iterations: Number of SimRNA sampling steps
- simrna_frames: Which frames to export (e.g. "1" or "1-3")
- decoy_agg: Aggregation method (min, mean, p10, boltz)
- vina_exhaustiveness: Vina exhaustiveness parameter
- vina_cpu: CPU threads per Vina task
- grid_padding: Docking box padding in angstroms
- ligand_conformers: Number of ligand conformers to generate
- ligand_seed: Random seed for conformer generation
- ligand_prune_rmsd: RMSD threshold for pruning conformers
- ligand_forcefield: Force field for optimization (MMFF or UFF)
- ligand_optimize: Whether to optimize conformers
- ligand_max_opt_iters: Maximum optimization iterations

### 7.5 regularize
- entropy_uniform_mix: Uniform-mixing regularization strength
- elite_min_hamming: Minimum Hamming distance between elites
- struct_unique: Prevent duplicate dot-bracket structures
- struct_metric: Structural distance metric (bp or edit)
- struct_min_dist: Minimum structural distance between elites

### 7.6 monitor

#### Penalty handling
- penalty_skip.enabled: Skip CE updates when too many sequences fail
- penalty_skip.frac_thr: Penalty fraction threshold
- penalty_skip.action: Action on repeated penalties (skip or reset_uniform)
- penalty_skip.reset_patience: Iterations before reset

#### Early stopping
- early_stop.enabled: Stop when improvement stalls
- early_stop.patience: Consecutive iterations without improvement
- early_stop.min_improve: Minimum required improvement

#### Collapse detection
- collapse.enabled: Monitor distribution collapse
- collapse.max_prob: Probability threshold for collapsed position
- collapse.frac_thr: Fraction of positions exceeding threshold
- collapse.patience: Iterations before stopping

#### External loop stopping
- external_stop.gbsa_patience: GBSA plateau detection window
- external_stop.gbsa_min_improve: Minimum GBSA improvement
- external_stop.dock_patience: Docking plateau window
- external_stop.dock_min_improve: Minimum docking improvement

### 7.7 rescore
- enabled: Whether MMGBSA rescoring is enabled
- top_frac: Fraction of batch selected for rescoring
- elite_frac: Fraction of rescored candidates used for feedback
- include_penalty: Allow penalty candidates into rescoring

### 7.8 md
- enabled: Whether MD postprocessing is enabled
- topX: Number of top sequences for MD
- gmx: GROMACS executable name
- mdp_dir: Directory containing .mdp files
- forcefield: Force field for topology generation
- water: Water model
- box_type: Box shape (cubic)
- box_distance_nm: Solute-to-box boundary distance
- salt_conc_m: Salt concentration (mol/L)
- neutralize: Whether to neutralize the system
- threads: CPU threads for MD (0 = auto)
- gpu: Enable GPU acceleration
- skip_if_done: Skip if output files exist
- strip_5prime_phosphate: Remove 5' phosphate before pdb2gmx
- strip_3prime_phosphate: Remove 3' phosphate before pdb2gmx

---

## 8) Smoke test

Before running expensive rescoring, do a cheap smoke test:

    python -m experiments.ce_design --config configs/report_smoke.yaml 2>&1 | tee /tmp/report_smoke.log

Validate multi-frame export:

    RUN=runs/report_smoke
    HASH=$(ls "$RUN/fold" | head -n 1)

    python - <<PY
    from pathlib import Path
    p = Path("$RUN/fold/$HASH/aptamer.trafl")
    n = sum(1 for line in p.read_text(errors="ignore").splitlines() if len(line.split()) == 5)
    print("frames_in_trafl =", n)
    PY

    ls -lh "$RUN/fold/$HASH/"*_AA.pdb
    ls -1 "$RUN/dock/$HASH" | head -10
    head -n 5 "$RUN/topN.csv"

Expected:
- frames_in_trafl >= 2
- At least two PDB files exported
- Docking folder includes frame_001 and frame_002
- topN.csv contains n_frames >= 2

---

## 9) How to analyze results

### 9.1 Workflow overview

Inner loop (within each CE iteration):
1. Sample ce.batch sequences from current dist.json
2. Evaluate: SimRNA -> dock each frame x conformer -> aggregate
3. Rank by docking score
4. Select docking elites using ce.elite_frac
5. Update dist.json from docking elites

Outer loop (MMGBSA feedback):
1. Export top candidates for rescoring
2. Run MMGBSA on selected candidates
3. Filter by mmgbsa_ok=True (exclude failed calculations)
4. Select GBSA elites
5. Update dist.json again using GBSA feedback
6. Generate next batch from updated distribution

### 9.2 Vina results: topN.csv

    head -n 10 runs/ce_run1/topN.csv

### 9.3 MMGBSA results: rescore/iter_XXX/mmgbsa_results.csv

    python -c "
    import pandas as pd
    df = pd.read_csv('runs/ce_run1/rescore/iter_001/mmgbsa_results.csv')
    n_ok = df['mmgbsa_ok'].astype(str).str.lower().isin(['true','1']).sum()
    print(f'MMGBSA success: {n_ok}/{len(df)}')
    if n_ok > 0:
        ok = df[df['mmgbsa_ok'].astype(str).str.lower().isin(['true','1'])]
        print(ok.nsmallest(5, 'mmgbsa_dg_bind')[['rank','seq','mmgbsa_dg_bind']].to_string(index=False))
    "

### 9.4 Check charge used in MMGBSA

    python -c "
    import pandas as pd
    df = pd.read_csv('runs/ce_run1/rescore/iter_001/mmgbsa_results.csv')
    print(df[['rank','mmgbsa_charge','mmgbsa_charge_source']].head(10).to_string(index=False))
    "

### 9.5 Export top candidates by MMGBSA

    python scripts/export_mmgbsa_best.py --run_dir runs/ce_run1

### 9.6 MD results

    cat runs/ce_run1/md/md_summary.csv

---

## 10) Common failure modes

### 10.1 All scores are 1e9 (penalty)

Check:
1. topN.csv error column
2. logs/ce_design.log tail
3. Whether fold/<hash>/*_AA.pdb exists
4. Whether dock/<hash>/frame_*/lig_*/ contains .vina.log

### 10.2 Multi-frame export not working

If frames_in_trafl = 1, ensure SimRNA outputs multiple frames and simrna_frames
requests the right range.

### 10.3 MMGBSA all failed (mmgbsa_ok=False)

Common causes:

1. Wrong ligand charge:

       cat runs/ce_run1/ligand_ensemble/charge.txt
       # Fix:
       python ligand/rdkit_conformers.py data/ligand.sdf \
           --out_dir runs/ce_run1/ligand_ensemble \
           --charge -1

2. Antechamber failed (check antechamber.log):

       tail -50 runs/ce_run1/rescore/iter_001/mmgbsa/rank_001_f*/antechamber.log

3. tleap failed (check tleap.log):

       tail -30 runs/ce_run1/rescore/iter_001/mmgbsa/rank_001_f*/tleap.log

Diagnosis command:

    python -c "
    import pandas as pd
    df = pd.read_csv('runs/ce_run1/rescore/iter_001/mmgbsa_results.csv')
    failed = df[df['mmgbsa_ok'].astype(str).str.lower() == 'false']
    print(f'Failed: {len(failed)}/{len(df)}')
    if len(failed) > 0:
        print(failed[['rank','mmgbsa_error']].head(3).to_string(index=False))
    "

### 10.4 Resume does not advance

Check:
- checkpoint.json for current iteration state
- history.csv for new rows after resume

---

## 11) Visualization

### 11.1 Plot convergence

    python scripts/visualize_run.py --run_dir runs/ce_run1

Output plots:
- plots/history_best_mean.png
- plots/history_penalty.png
- plots/mmgbsa_by_iter.png
- plots/mmgbsa_vs_vina.png
- plots/md/rank_00x_rmsd.png
- plots/md/rank_00x_rmsf.png

---

## 12) CE theory mapping

- A. Initialize distribution -> dist.json
- B. Sample batch -> ce.batch
- C. Evaluate samples -> SimRNA + Vina + aggregation
- D. Select elites -> elite_frac + diversity constraints
- E. Rescore topM -> MMGBSA via rescore/iter_XXX/
- F. Update distribution -> feedback updates dist.json

---

## 13) Reproducibility notes

To reproduce results:
- Keep seed fixed and store config_resolved.json
- Archive: config file, dist.json, checkpoint.json, topN.csv, history.csv
- Archive: ligand_ensemble/manifest.json (includes charge info)
- Record software versions (RDKit, Vina, SimRNA, AmberTools)

Minimum debugging set:
- topN.csv (first 5 lines + error column)
- logs/ce_design.log (last 100 lines)
- ligand_ensemble/charge.txt
- One fold/<hash>/ listing
