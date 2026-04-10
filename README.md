# Aptamer Design Pipeline (CE + Multi-frame SimRNA + Vina Docking + MM/GBSA Feedback Loop + Optional GB Minimization + Optional MD)

Author: maning

This repository implements an end-to-end, reproducible RNA aptamer design pipeline driven by a Cross-Entropy (CE) optimizer.

At each CE iteration, the pipeline samples RNA sequences from a probability distribution, evaluates them via RNA structure sampling (SimRNA) and docking (AutoDock Vina) using multiple RNA decoy frames and multiple ligand conformers, then runs MM/GBSA rescoring on top candidates and feeds those scores back to update the CE distribution for the next iteration.

**NEW:** MM/GBSA rescoring supports an **optional short implicit-solvent GB minimization (AmberTools `sander`)** on the complex before running `MMPBSA.py`, which can reduce clashes and make rescoring more stable.

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
3. Multi-frame decoy export (SimRNA `.trafl` to multiple all-atom PDBs)
4. Docking (Vina) across RNA frames × ligand conformers
5. Aggregation across frames/conformers (e.g. min)
6. CE update using docking elites
7. MM/GBSA rescoring for top candidates
8. **(Optional NEW)** GB minimization before MM/GBSA (`sander` + restraints)
9. Feedback update: MM/GBSA results update `dist.json`
10. Resume next CE iteration until stopping criteria are met
11. Optional MD postprocessing (GROMACS)

---

## 2) Key outputs (per run directory)

A run directory (e.g. `runs/ce_run1/`) contains:
runs/<RUN_NAME>/
├── config_resolved.json
├── meta.json
├── logs/
│   └── ce_design.log
├── checkpoint.json
├── dist.json
├── history.csv
├── topN.csv
├── next_batch.txt
├── ligand_ensemble/
│   ├── manifest.json
│   ├── charge.json
│   ├── charge.txt
│   └── conf_*.sdf
├── fold/
│   └── <eval_hash>/
│       ├── aptamer.trafl
│       ├── aptamer-000001_AA.pdb
│       └── ...
├── dock/
│   └── <eval_hash>/
│       └── frame_XXX/
│           └── lig_YYY/
│               ├── *.dock.pdbqt
│               └── *.vina.log
├── rescore/
│   └── iter_XXX/
│       ├── manifest.jsonl
│       ├── mmgbsa_results.csv
│       ├── results.csv
│       ├── elites_for_ce.txt
│       └── feedback.done
└── md/
    ├── rank_00x_<eval_hash>/
    └── md_summary.csv
### File purpose summary

| File | Purpose |
|------|---------|
| `dist.json` | CE distribution state |
| `checkpoint.json` | Resume state & stop_reason |
| `history.csv` | Per-iteration metrics |
| `topN.csv` | Best docking candidates |
| `ligand_ensemble/manifest.json` | Charge detection info |
| `rescore/iter_XXX/mmgbsa_results.csv` | Final rescoring results |
| `rescore/iter_XXX/results.csv` | Feedback input |
| `rescore/iter_XXX/feedback.done` | Completion marker |

---

## 3) Environment and dependencies

### 3.1 Recommended: mamba

This project includes an `environment.yml`.

```bash
micromamba create -f environment.yml -y
micromamba shell init -s bash -r /root/.local/share/mamba
source ~/.bashrc
micromamba activate aptamer_design
```

### 3.2 External tools required

The pipeline assumes these tools are available in PATH:

- SimRNA (`SimRNA`, `SimRNA_trafl2pdbs`)
- AutoDock Vina (`vina`)
- AmberTools: `tleap`, `antechamber`, `parmchk2`, `cpptraj`, `pdb4amber`, `MMPBSA.py`
- **(NEW: only if minimization enabled)** AmberTools `sander`
- OpenBabel (`obabel`)
- ViennaRNA (`RNAfold`)
- Optional MD: GROMACS (`gmx`)

Quick check:

```bash
which RNAfold || echo "RNAfold not found"
which SimRNA || echo "SimRNA not found"
which SimRNA_trafl2pdbs || echo "SimRNA_trafl2pdbs not found"
which obabel || echo "obabel not found"
which vina || echo "vina not found"
which tleap || echo "tleap not found"
which MMPBSA.py || echo "MMPBSA.py not found"
which sander || echo "sander not found (only needed if minimization enabled)"
```

### 3.3 Preflight check

```bash
bash scripts/preflight.sh
```

---

## 4) Ligand charge handling (IMPORTANT)

### Why charge matters

Ligand charge critically affects:
- Antechamber parameterization (AM1-BCC charge assignment)
- MM/GBSA binding free energy
- Pose preference & docking stability

Wrong charge may cause MM/GBSA to fail or produce meaningless results.

### Automatic charge detection (priority order)

The pipeline auto-detects ligand charge in this order:

1. CLI override `--net_charge` (MM/GBSA only)
2. `ligand_ensemble/manifest.json` (`charge_info.formal_charge`)
3. `ligand_ensemble/charge.json`
4. `ligand_ensemble/charge.txt`
5. SDF `NET_CHARGE` property (conformer SDFs)
6. RDKit formal charge from ligand SDF
7. PDBQT Gasteiger sum (last resort, unreliable)
8. Fallback to 0 (with warning)

### How to check your ligand charge

```bash
python ligand/rdkit_conformers.py data/your_ligand.sdf --charge_only
```

### How to set ligand charge

Prepare ligand conformers with the correct charge:

```bash
cd ~/autodl-tmp/aptamer_design

python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/ce_run1/ligand_ensemble \
  --n_confs 10 \
  --charge -1

cat runs/ce_run1/ligand_ensemble/charge.txt
```

### Override charge during MM/GBSA only

```bash
python -m experiments.rescore_mmgbsa \
  --run_dir runs/ce_run1 \
  --iter 1 \
  --net_charge -2
```

---

## 5) How to run (production)

### Standard workflow

```bash
cd ~/autodl-tmp/aptamer_design

# Step 1: Prepare ligand ensemble with correct charge
python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/ce_run1/ligand_ensemble \
  --n_confs 10 \
  --charge -1

# Step 2: Verify charge
cat runs/ce_run1/ligand_ensemble/charge.txt

# Step 3: Run CE + docking + MM/GBSA feedback loop
bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml
```

### One-line command

```bash
python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/ce_run1/ligand_ensemble \
  --n_confs 10 \
  --charge -1 && \
bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml
```

### What the loop does (`scripts/run_ce_gbsa_loop.sh`)

1. Read `checkpoint.json` to determine last completed iteration
2. Run CE to `iter_completed + 1`
3. Run MM/GBSA rescoring for that iteration
4. **(Optional NEW)** If enabled, run short GB minimization before `MMPBSA.py`
5. Run feedback (`experiments.gbsa_feedback_once`) to update `dist.json` and generate `next_batch.txt`
6. Repeat until stop condition reached (`ce.iters`, collapse, early stop, external plateau, etc.)

### Resume after interruption

Just re-run the same command:

```bash
bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml
```

The loop will skip finished iterations (using `feedback.done`) and continue.

### Restart from scratch

```bash
rm -rf runs/ce_run1
# then re-run from ligand prep
```

---

## 6) Quick Start (5-minute test)

```bash
cd ~/autodl-tmp/aptamer_design

# 1) Prepare ligand
python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/test/ligand_ensemble \
  --n_confs 2 \
  --charge -1

# 2) Minimal config
cat > configs/test.yaml <<'TESTEOF'
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
TESTEOF

# 3) Run
python -m experiments.ce_design --config configs/test.yaml

# 4) Inspect
head -n 5 runs/test/topN.csv
```

---

## 7) Configuration explained

### 7.1 `run`
- `run_dir`: output directory
- `seed`: random seed
- `workers`: parallel evaluation workers
- `log_level`: INFO/DEBUG/...
- `resume`: resume from checkpoint

### 7.2 `inputs`
- `ligand`: ligand file path (SDF/PDB)

### 7.3 `ce`
- `length`, `alphabet`
- `batch`, `iters`
- `elite_frac`
- `alpha`, `min_prob`, `temperature`
- `topN`: export topN docking results for inspection

### 7.4 `eval`
- `simrna_config`, `omp_threads`, `simrna_iterations`
- `simrna_frames`: e.g. `"1"`, `"2-3"`
- `decoy_agg`: `min`, `mean`, `p10`, `boltz`
- `vina_exhaustiveness`, `vina_cpu`, `grid_padding`
- ligand conformer knobs: `ligand_conformers`, `ligand_seed`, `ligand_prune_rmsd`, `ligand_forcefield`, `ligand_optimize`, `ligand_max_opt_iters`

### 7.5 `regularize`
- `entropy_uniform_mix`
- `elite_min_hamming`
- `struct_unique`, `struct_metric`, `struct_min_dist`

### 7.6 `monitor`
- `penalty_skip.*`
- `early_stop.*`
- `collapse.*`
- `external_stop.*`

### 7.7 `rescore` (MM/GBSA feedback)

Core keys:
- `enabled`: enable MM/GBSA
- `top_frac`: fraction of docking batch to rescore
- `elite_frac`: fraction of rescored set used for feedback elites
- `include_penalty`: whether penalty samples can be rescored
- `gbsa_workers`: number of parallel MM/GBSA jobs
- `gbsa_charge_method`: antechamber method: `bcc` or `gas`

GB model keys:
- `mmgbsa_igb`: GB model (commonly 5; GBn/GBn2 are 7/8)
- `mmgbsa_saltcon`: salt concentration (M)

**NEW: Minimization before MMPBSA**
- `mmgbsa_minimize`: true/false (enable `sander` minimization)
- `mmgbsa_min_restraint`: true/false (recommended true)
- `mmgbsa_min_maxcyc`: default 1500
- `mmgbsa_min_ncyc`: default 500
- `mmgbsa_min_restraint_wt`: default 2.0
- `mmgbsa_min_restraint_mask`: recommended `"!:LIG & !@H="`

Important YAML note: the mask contains `!`, so keep it quoted:

```yaml
mmgbsa_min_restraint_mask: "!:LIG & !@H="
```

### 7.8 `md`
- Optional MD configuration

---

## 8) Smoke test

Before expensive rescoring, do a cheap smoke test:

```bash
python -m experiments.ce_design --config configs/report_smoke.yaml 2>&1 | tee /tmp/report_smoke.log
```

Validate multi-frame export:

```bash
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
```

Expected:
- `frames_in_trafl >= 2`
- at least two PDB files exported
- docking folder includes multiple frames
- `topN.csv` shows `n_frames >= 2`

---

## 9) How to analyze results

### 9.1 Docking results (`topN.csv`)

```bash
head -n 10 runs/ce_run1/topN.csv
```

### 9.2 MM/GBSA results (`rescore/iter_XXX/mmgbsa_results.csv`)

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/ce_run1/rescore/iter_001/mmgbsa_results.csv')
n_ok = df['mmgbsa_ok'].astype(str).str.lower().isin(['true','1']).sum()
print(f'MMGBSA success: {n_ok}/{len(df)}')
if n_ok > 0:
    ok = df[df['mmgbsa_ok'].astype(str).str.lower().isin(['true','1'])]
    print(ok.nsmallest(5, 'mmgbsa_dg_bind')[['rank','seq','mmgbsa_dg_bind']].to_string(index=False))
"
```

### 9.3 Check charge used in MM/GBSA

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/ce_run1/rescore/iter_001/mmgbsa_results.csv')
cols = [c for c in ['rank','mmgbsa_charge','mmgbsa_charge_source'] if c in df.columns]
print(df[cols].head(10).to_string(index=False))
"
```

### 9.4 Check whether minimization was used

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/ce_run1/rescore/iter_001/mmgbsa_results.csv')
cols = [c for c in ['rank','mmgbsa_minimized','mmgbsa_min_maxcyc','mmgbsa_min_ncyc','mmgbsa_min_restraint'] if c in df.columns]
print(df[cols].head(10).to_string(index=False))
"
```

### 9.5 Manually re-run MM/GBSA with minimization

```bash
python -m experiments.rescore_mmgbsa \
  --run_dir runs/ce_run1 \
  --iter 1 \
  --topM 5 \
  --workers 1 \
  --igb 5 \
  --saltcon 0.150 \
  --minimize \
  --min_maxcyc 1500 \
  --min_ncyc 500 \
  --min_restraint \
  --min_restraint_wt 2.0 \
  --min_restraint_mask '!:LIG & !@H='
```

---

## 10) Common failure modes

### 10.1 All scores are `1e9` (penalty)

Check:
1. `topN.csv` error column
2. `logs/ce_design.log` tail
3. whether `fold/<hash>/*_AA.pdb` exists
4. whether `dock/<hash>/frame_*/lig_*/` contains `.vina.log`

### 10.2 MM/GBSA all failed (`mmgbsa_ok=False`)

Common causes:

**1) Wrong ligand charge:**

```bash
cat runs/ce_run1/ligand_ensemble/charge.txt
```

**2) Antechamber failed:**

```bash
tail -50 runs/ce_run1/rescore/iter_001/mmgbsa/rank_001_f*/antechamber.log
```

**3) tleap failed:**

```bash
tail -30 runs/ce_run1/rescore/iter_001/mmgbsa/rank_001_f*/tleap.log
```

**4) Minimization failed (only if enabled; requires `sander`):**

```bash
tail -80 runs/ce_run1/rescore/iter_001/mmgbsa/rank_001_f*/sander_min.log
```

---

## 11) Visualization

```bash
python scripts/visualize_run.py --run_dir runs/ce_run1
```

---

## 12) CE theory mapping

- A. Initialize distribution → `dist.json`
- B. Sample batch → `ce.batch`
- C. Evaluate samples → SimRNA + Vina + aggregation
- D. Select elites → `elite_frac` + diversity constraints
- E. Rescore topM → MM/GBSA via `rescore/iter_XXX/`
- F. Update distribution → feedback updates `dist.json`

---

## 13) Reproducibility notes

To reproduce results:
- Keep seed fixed and store `config_resolved.json`
- Archive: config file, `dist.json`, `checkpoint.json`, `topN.csv`, `history.csv`
- Archive: `ligand_ensemble/manifest.json` (includes charge info)
- Record software versions (RDKit, Vina, SimRNA, AmberTools)

Minimum debugging set:
- `topN.csv` (first 5 lines + error column)
- `logs/ce_design.log` (last 100 lines)
- `ligand_ensemble/charge.txt`
- one `fold/<hash>/` listing
- one failing MM/GBSA workdir logs (`antechamber.log`, `tleap.log`, `mmpbsa.log`, and if enabled: `sander_min.log`)
```
