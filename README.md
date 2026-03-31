# Aptamer Design Pipeline (CE + Multi-frame SimRNA + Vina Docking + MMGBSA Feedback Loop + MD)

Author: maning

This repository implements an end-to-end, reproducible RNA aptamer design pipeline driven by a Cross-Entropy (CE) optimizer. Each CE iteration samples RNA sequences from a probability distribution, evaluates them via structure sampling (SimRNA) and docking (AutoDock Vina) using multiple RNA decoy frames and multiple ligand conformers, then optionally runs MMGBSA rescoring on top candidates and feeds those scores back to update the CE distribution for the next iteration.

The workflow is designed to be restartable/resumable and to support cheap smoke tests before running expensive rescoring.

---

## 1) What this pipeline does

Given a ligand structure (SDF/PDB), the pipeline performs:

1. Ligand ensemble generation (RDKit conformers)
2. RNA structure sampling (SimRNA trajectory)
3. Multi-frame decoy export (SimRNA `.trafl` → multiple all-atom PDBs)
4. Docking (Vina) across RNA frames (decoys) × ligand conformers
5. Aggregation across frames/conformers (e.g., `min`)
6. Top-N ranking → `topN.csv`
7. Optional MMGBSA rescoring for top-M
8. Feedback update: rescoring results update `dist.json`
9. Resume to next CE iteration until termination criteria are met

---

## 2) Key outputs (per run directory)

A run directory (e.g. `runs/ce_run1/`) contains:

```
runs/<RUN_NAME>/
├── config_resolved.json          # resolved configuration for reproducibility
├── meta.json                     # run metadata
├── logs/ce_design.log            # main log
├── checkpoint.json               # resume/stop state
├── dist.json                     # CE distribution (core state)
├── history.csv                   # per-iteration summary metrics
├── topN.csv                      # detailed ranked candidates (post-aggregation)
├── ligand_ensemble/              # ligand conformers + manifest.json
├── fold/<eval_hash>/             # SimRNA outputs per sequence
│   ├── aptamer.trafl             # SimRNA trajectory
│   ├── aptamer-000001_AA.pdb     # frame_001 exported all-atom PDB
│   ├── aptamer-000002_AA.pdb     # frame_002 (if enabled)
│   └── ...
├── dock/<eval_hash>/frame_XXX/   # docking results per frame
│   └── lig_YYY/                  # docking results per ligand conformer
│       ├── *_dock.pdbqt
│       ├── *.vina.log
│       └── ...
│ ── md/rank_00x_<eval_hash>/     # MD results per frame
│   └── lig_YYY/                  
│       ├── *_dock.pdbqt
│       ├── *md.log
│       └── ...
└── rescore/iter_XXX/             # rescoring directory (if enabled)
    ├── manifest.jsonl            # topM jobs derived from topN.csv
    ├── mmgbsa_results.csv        # MMGBSA results (real rescoring)
    ├── results.csv               # normalized results for feedback step
    ├── results.jsonl             # same as above (jsonl form)
    └── elites_for_ce.txt         # elites selected for CE update (traceability)
```

Two files matter most for the feedback loop:
- `dist.json`: current CE distribution (updated each iteration)
- `checkpoint.json`: current iteration/termination state (controls resume)

---

## 3) Environment & dependencies

### 3.1 Recommended: mamba

This project includes an `environment.yml`.

```bash
micromamba create -f environment.yml -y
micromamba shell init -s bash -r /root/.local/share/mamba
source ~/.bashrc
micromamba activate aptamer_design
```

### 3.2 External tools required on your machine

The pipeline assumes these tools are available in `PATH` (or configured by absolute path in code/config):

- SimRNA (including `SimRNA` and `SimRNA_trafl2pdbs`)
- AutoDock Vina (`vina`)
- MMGBSA toolchain (AmberTools / `MMPBSA.py` or your MMGBSA implementation)

Quick check:

```bash
which SimRNA || echo "SimRNA not found"
which SimRNA_trafl2pdbs || echo "SimRNA_trafl2pdbs not found"
which vina || echo "vina not found"
```

---

## 4) Configuration explained (`run_default.yaml`)

Example config:

```yaml
run:
  run_dir: runs/ce_run1
  seed: 123
  log_level: INFO
  workers: 4

inputs:
  ligand: data/ligand.sdf

ce:
  length: 30
  alphabet: RNA
  batch: 24
  iters: 30
  elite_frac: 0.2
  alpha: 0.4
  min_prob: 0.02
  temperature: 1.0
  topN: 20

eval:
  omp_threads: 1
  simrna_iterations: 20000
  simrna_frames: "1-3"
  decoy_agg: "min"
  vina_exhaustiveness: 8
  vina_cpu: 1
  grid_padding: 8.0

  ligand_conformers: 20
  ligand_seed: 123
  ligand_prune_rmsd: 0.5
  ligand_forcefield: "MMFF"
  ligand_optimize: true
  ligand_max_opt_iters: 200

regularize:
  entropy_uniform_mix: 0.02
  diversity_weight: 0.10
  elite_min_hamming: 0.10
  elite_diverse: true

monitor:
  penalty_threshold: 1.0e8

  early_stop:
    enabled: true
    patience: 5
    min_improve: 0.05

  collapse:
    enabled: true
    max_prob: 0.95
    frac_thr: 0.60
    patience: 3

rescore:
  enabled: true
  topM: 5
  include_penalty: false
md:
  enabled: true
  topX: 10
  gmx: gmx
  mdp_dir: mdp
  forcefield: amber99sb-ildn
  water: tip3p
  box_type: cubic
  box_distance_nm: 1.0
  salt_conc_m: 0.15
  neutralize: true
  threads: 0
  gpu: false
  skip_if_done: true
  strip_5prime_phosphate: true
  strip_3prime_phosphate: false
```

### 4.1 `run`
- `run_dir`: output folder for all artifacts
- `seed`: RNG seed (reproducibility)
- `workers`: parallel evaluation workers
- `log_level`: logging verbosity

### 4.2 `inputs`
- `ligand`: ligand file path (SDF/PDB,It's best to name it 'ligand')

### 4.3 `ce` (Cross-Entropy optimizer)
- `length`: RNA length
- `batch`: sequences evaluated per iteration
- `iters`: max CE iterations
- `elite_frac`: fraction of batch selected as elites
- `alpha`: update step size (aggressiveness)
- `min_prob`: floor probability to avoid collapse
- `temperature`: sampling temperature
- `topN`: how many top candidates to persist in `topN.csv`

### 4.4 `eval` (SimRNA + docking + aggregation)
- `simrna_iterations`: SimRNA sampling steps (higher = slower)
- `simrna_frames`: which frames to export (e.g., `"1-3"`)
- `decoy_agg`: how to aggregate multi-frame/multi-conformer docking scores (`min` recommended)
- `vina_exhaustiveness`, `vina_cpu`, `grid_padding`: docking parameters
- `ligand_conformers`: how many ligand conformers to generate
- `ligand_*`: RDKit conformer generation and optimization settings

### 4.5 `regularize` (avoid collapse / enforce diversity)
- `entropy_uniform_mix`: mix in uniform distribution to keep entropy
- `elite_min_hamming`: minimum Hamming distance among elites
- `elite_diverse`: whether to enforce diversity selection
- `diversity_weight`: strength of diversity regularization

### 4.6 `monitor` (termination criteria)
- `penalty_threshold`: scores above this are treated as penalties/failures (e.g., 1e9)
- `early_stop`: stop if no meaningful improvement for `patience` iterations
- `collapse`: stop if distribution collapses (max probability too high for too many positions)

### 4.7 `rescore` (MMGBSA rescoring)
- `enabled`: turn rescoring on/off
- `topM`: number of candidates selected from `topN.csv` for MMGBSA
- `include_penalty`: whether penalty candidates are allowed into rescoring (usually false)
  
### 4.8`MD` 
- `topX`: Set the first X sequences to run MD
-  `forcefield`:Set force field type
-  `water`:Set up a water model
-  `box_distance_nm`:Solvent boundary distance
-  `salt_conc_m`:salt concentration
-  `threads`:CPU thread count (0=automatic)
-  `gpu`:Whether to use GPU or not
-  `skip_if_done`:Completed or skipped
---

## 5) Smoke test (verify multi-frame + multi-conformer + aggregation)

Before running expensive rescoring, do a cheap smoke test by reducing parameters:
- `ce.iters: 1`
- `ce.batch: 2-4`
- `simrna_frames: "1-2"`
- `ligand_conformers: 2-3`
- `vina_exhaustiveness: 2`

Run:

```bash
python -m experiments.ce_design --config configs/report_smoke.yaml 2>&1 | tee /tmp/report_smoke.log
```

Validate multi-frame export:

```bash
RUN=runs/report_smok
HASH=$(ls "$RUN/fold" | head -n 1)

python - <<PY
from pathlib import Path
p=Path("$RUN/fold/$HASH/aptamer.trafl")
n=0
for line in p.read_text(errors="ignore").splitlines():
    if len(line.split())==5:
        n+=1
print("frames_in_trafl =", n)
PY

ls -lh "$RUN/fold/$HASH/"*_AA.pdb
ls -1 "$RUN/dock/$HASH" | sed -n '1,20p'
head -n 5 "$RUN/topN.csv"
```

Expected:
- `frames_in_trafl >= 2`
- at least two files: `aptamer-000001_AA.pdb` and `aptamer-000002_AA.pdb`
- docking folder includes `frame_001` and `frame_002`
- `topN.csv` contains `n_frames >= 2`

---

## 6) How to analyze results 

### 6.1 Convergence trend: `history.csv`

```bash
tail -n 20 runs/ce_run1/history.csv
```

Typical signals:
- `best` improves (more negative is better for docking score)
- `penalty_frac` stays low (near 0)
- `global_best` monotonically improves

### 6.2 Candidate details: `topN.csv`

```bash
head -n 5 runs/ce_run1/topN.csv
```

Focus fields:
- `score`: aggregated score across frames/conformers
- `n_frames`: number of frames used (should match export)
- `best_frame`: which frame gave best score
- `best_receptor_pdb`: best `_AA.pdb`
- `best_dock_pose_pdbqt`: best pose
- `best_ligand_sdf`: best ligand conformer
- `error`: if present, explains penalty/exception

### 6.3 Verify multi-frame decoys per sequence

```bash
ls -lh runs/ce_run1/fold/<eval_hash>/*_AA.pdb
```

Count frames in `.trafl`:

```bash
python - <<'PY'
from pathlib import Path
p=Path("runs/ce_run1/fold/<eval_hash>/aptamer.trafl")
n=0
for line in p.read_text(errors="ignore").splitlines():
    if len(line.split())==5:
        n+=1
print("frames_in_trafl =", n)
PY
```

### 6.4 Rescoring outputs 

Per iteration:

```bash
ls runs/ce_run1/rescore/iter_001/
```

Key files:
- `manifest.jsonl`: rescoring job list (topM)
- `mmgbsa_results.csv`: MMGBSA results (real)
- `results.csv`: normalized input to feedback
- `elites_for_ce.txt`: elites used to update distribution

---

## 7) Common failure modes

### 7.1 All scores are 1e9 (penalty)

Usually means fold/dock failed.
Check:
1. `topN.csv` → `error` column
2. `logs/ce_design.log` tail
3. whether `fold/<hash>/*_AA.pdb` exists
4. whether `dock/<hash>/frame_*/lig_*/` contains `.vina.log`

### 7.2 Multi-frame export not working

If `frames_in_trafl = 1`, the trajectory contains only one frame, so exporting multiple PDBs will not work. Ensure SimRNA is configured to output a trajectory with multiple frames, and that the pipeline is exporting the requested frame range (`simrna_frames`).

### 7.3 Resume does not advance

Check:
- `checkpoint.json` for current iteration state
- whether the run is started with resume enabled (if your config supports it)
- whether iteration counters in `history.csv` advance after a resume run

---

## 8) CE theory mapping 

This pipeline matches the standard CE loop:

- A. Initialize distribution → `dist.json`
- B. Sample batch → `ce.batch`
- C. Evaluate samples → SimRNA multi-frame + ligand multi-conformer + Vina + aggregation
- D. Select elites → `elite_frac` + diversity constraints
- E. Rescore topM → MMGBSA via `rescore/iter_XXX/`
- F. Update distribution → feedback updates `dist.json`, then resume to next iteration

When you successfully generate new sequences in iteration 2 after rescoring feedback, the “E → F → next B/C” chain is proven.

---

## 9) Reproducibility notes 

To reproduce my results reliably:
- Keep `seed` fixed and store `config_resolved.json`
- Always archive:
  - the exact configuration file used
  - `dist.json` and `checkpoint.json`
  - `topN.csv` and `history.csv`
  - software versions (RDKit, Vina, SimRNA build)

For debugging, the minimum set I use to triage failures is:
- `topN.csv` (first 5 lines + `error`)
- `logs/ce_design.log` (last 100 lines)
- one `fold/<hash>/` listing and `frames_in_trafl` count

---

## 10) How to run (the commands you actually type)

> TL;DR: **first preflight**, then run either **one-round** (debug) or **full loop** (production).

### 10.1 Preflight (sanity check your environment)

```bash
bash scripts/preflight.sh
```

What it checks (high level):
- Conda env / python imports
- external binaries in PATH (SimRNA, trafl2pdbs, vina, AmberTools: tleap/antechamber/parmchk2/sander/MMPBSA.py)
- write permissions + run directory layout
- (if enabled) summary csv generation hooks


### 10.2 Run the full multi-iteration CE+MMGBSA feedback loop (until termination)

```bash
bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml
```

This loop will:
1) read `checkpoint.json` to know what iteration is completed
2) run CE to `iter_completed + 1`
3) run MMGBSA for that iteration
4) run feedback to update `dist.json`
5) repeat until:
   - `ce.iters` reached **OR**
   - `stop_reason` in `checkpoint.json` becomes non-empty (early stop / collapse, etc.)

**Important:** if `ce.iters=30`, the loop performs MMGBSA+feedback for every iteration it successfully completes (1..30).  
If you stop the loop manually and rerun later, it will continue from `checkpoint.json`.

### 10.3 Restart / resume rules 

The pipeline is designed to be restartable:
- `dist.json` = your CE distribution state
- `checkpoint.json` = how many iters are finished + stop reason

If those are present, rerunning the scripts will **resume** rather than start from scratch, unless you delete the run directory.

To force a clean run:
```bash
rm -rf runs/ce_run1
```

---

## 11) What to look at for final results

There are two “views”:

### 11.1 Vina (always available): `topN.csv`

```bash
head -n 10 runs/ce_run1/topN.csv
```

Use this for:
- quick ranking
- sanity check (frames/conformers counted)
- locate best pose PDBQT and best receptor PDB

### 11.2 MMGBSA ( rescoring): `rescore/iter_XXX/mmgbsa_results.csv`

Per iteration:
```bash
ls runs/ce_run1/rescore/iter_001/
head -n 10 runs/ce_run1/rescore/iter_001/mmgbsa_results.csv
```

Across the whole run (if you enabled summaries in preflight):
- `runs/ce_run1/summary_by_vina.csv`
- `runs/ce_run1/summary_by_mmgbsa.csv`

Recommended final selection strategy:
1) take the **top K by MMGBSA** (more negative is better, if you’re computing ΔGbind that way)
2) check their **vina score**, best_frame, and dock pose quality
3) keep a small diverse set (avoid near-duplicates)

---
### 11.3 Molecular Dynamics : `/runs/ce_run1/md`
The final summary results are presented in this document
- `/runs/ce_run1/md/md_summary.csv`
Contains information such as ranking, sequence, score, etc

## 12) Visualization

### 12.1 Plot convergence from `history.csv`

```bash
tail -n 20 runs/ce_run1/history.csv
python scripts/visualize_run.py --run_dir runs/ce_run1
```

Outputs:
- `runs/ce_run1/plots/history_best_mean.png`
- `runs/ce_run1/plots/history_penalty.png` 
- `runs/ce_run1/plots/mmgbsa_by_iter.png` 
- `runs/ce_run1/plots/mmgbsa_vs_vina.png`
- `runs/ce_run1/plots/md/rank_00x_rmsd.png`
- `runs/ce_run1/plots/md/rank_00x_rmsf.png`

