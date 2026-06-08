### Aptamer Design Pipeline (CE + Multi-frame SimRNA + Vina Docking + MM/GBSA Feedback Loop + GB Minimization + Optional MD)

This repository implements an end-to-end, reproducible RNA aptamer design pipeline driven by a Cross-Entropy (CE) optimizer.

At each CE iteration, the pipeline samples RNA sequences from a probability distribution, evaluates them via RNA structure sampling (SimRNA) and docking (AutoDock Vina) using multiple RNA decoy frames and multiple ligand conformers, then runs MM/GBSA rescoring on top candidates and feeds those scores back to update the CE distribution for the next iteration.

MM/GBSA rescoring supports an **optional short implicit-solvent GB minimization (AmberTools `sander`)** on the complex before running `MMPBSA.py`, which can reduce clashes and make rescoring more stable.

The workflow is restartable/resumable and supports cheap smoke tests before expensive rescoring.

---

## Table of Contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [Environment setup (complete guide)](#2-environment-setup-complete-guide)
3. [Ligand charge handling (IMPORTANT)](#3-ligand-charge-handling-important)
4. [How to run (production)](#4-how-to-run-production)
5. [Quick Start (5-minute test)](#5-quick-start-5-minute-test)
6. [Configuration explained](#6-configuration-explained)
7. [How to analyze results](#7-how-to-analyze-results)
8. [Common failure modes](#8-common-failure-modes)
9. [Visualization](#9-visualization)
10. [Adapting to new molecules](#10-adapting-to-new-molecules)
11. [CE theory mapping](#11-ce-theory-mapping)
12. [Reproducibility notes](#12-reproducibility-notes)

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

## 2) Environment setup (complete guide)

### 2.1 Install micromamba

```bash
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
sudo mv bin/micromamba /usr/local/bin/
```

### 2.2 Create conda environment

```bash
micromamba create -f environment.yml -y
micromamba shell init -s bash
source ~/.bashrc
micromamba activate aptamer_design
```

### 2.3 Install bioinformatics tools

```bash
micromamba install -n aptamer_design -c bioconda -c conda-forge \
  viennarna autodock-vina ambertools acpype -y
```

Verify installation:
```bash
which RNAfold || echo "RNAfold not found"
which vina || echo "Vina not found"
which tleap || echo "tleap not found"
which MMPBSA.py || echo "MMPBSA.py not found"
which sander || echo "sander not found (only needed if minimization enabled)"
```

### 2.4 Install SimRNA

Download SimRNA from [official website](http://genesilico.pl/software/stand-alone/simrna) and extract:

```bash
# Example installation to ~/tools/SimRNA
mkdir -p ~/tools
cd ~/tools
tar -xzf SimRNA_64bitIntel_Linux.tgz
mv SimRNA_64bitIntel_Linux SimRNA
```

Set environment variables (add to `~/.bashrc`):

```bash
export SIMRNA_HOME=~/tools/SimRNA
export SIMRNA_CONFIG=$SIMRNA_HOME/config.dat
export SIMRNA_DATA=$SIMRNA_HOME/data
export PATH=$SIMRNA_HOME:$PATH
```


Make executables runnable:

```bash
chmod +x $SIMRNA_HOME/SimRNA
chmod +x $SIMRNA_HOME/SimRNA_trafl2pdbs
chmod +x $SIMRNA_HOME/clustering
chmod +x $SIMRNA_HOME/calc_rmsd_to_1st_frame
chmod +x $SIMRNA_HOME/traflView
```
Apply changes:
```bash
source ~/.bashrc
```

### 2.5 Update YAML configuration

Edit your config file (e.g. `configs/run_default.yaml`) to point to your SimRNA installation:

```yaml
eval:
  simrna_config: ~/tools/SimRNA/config.dat  # or your installation path
```

### 2.6 Activate environment (for every new session)

```bash
micromamba activate aptamer_design
```

### 2.7 Verify all tools

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

---

## 3) Ligand charge handling (IMPORTANT)

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

### Common ligand charges at pH 7.4

| Molecule type | Typical charge | Examples |
|--------------|----------------|----------|
| Phosphate compounds | -1 to -4 | ATP (-4), ADP (-3), phosphate ester (-1/-2) |
| Carboxylic acids | -1 to -2 | Glutamate (-1), oxalate (-2) |
| Sulfonic acids | -1 | PFHxS (-1), sulfonamide (-1) |
| Neutral molecules | 0 | Caffeine, glucose |
| Amines | +1 | Dopamine (+1), arginine (+1) |

### How to check your ligand charge

```bash
python ligand/rdkit_conformers.py data/your_ligand.sdf --charge_only
```

### How to set ligand charge

Prepare ligand conformers with the correct charge:

```bash
python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/my_run/ligand_ensemble \
  --n_confs 10 \
  --charge -1

# Verify
cat runs/my_run/ligand_ensemble/charge.txt
```

### Override charge during MM/GBSA only

```bash
python -m experiments.rescore_mmgbsa \
  --run_dir runs/my_run \
  --iter 1 \
  --net_charge -2
```

---

## 4) How to run (production)

### Standard workflow

```bash
# Step 1: Prepare ligand ensemble with correct charge
python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/my_run/ligand_ensemble \
  --n_confs 10 \
  --charge -1

# Step 2: Verify charge
cat runs/my_run/ligand_ensemble/charge.txt

# Step 3: Run CE + docking + MM/GBSA feedback loop
bash scripts/run_ce_gbsa_loop.sh configs/run_default.yaml
```

### One-line command

```bash
python ligand/rdkit_conformers.py data/PFHxS.sdf \
  --out_dir runs/my_run/ligand_ensemble \
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
rm -rf runs/my_run
# then re-run from ligand prep
```

### Stopping criteria (automatic)

The loop stops when any of these conditions is met:

1. `ce.iters` reached (e.g. 30 iterations)
2. `monitor.early_stop` triggered (docking plateau)
3. `monitor.collapse` triggered (distribution collapsed)
4. External plateau detection (both GBSA and docking plateaus)

Check which criterion stopped the run:

```bash
cat runs/my_run/checkpoint.json
```

---

## 5) Quick Start (5-minute test)

```bash
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
  simrna_config: ~/tools/SimRNA/config.dat  # adjust to your path
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

## 6) Configuration explained

### 6.1 `run`
- `run_dir`: output directory
- `seed`: random seed
- `workers`: parallel evaluation workers
- `log_level`: INFO/DEBUG/...

### 6.2 `inputs`
- `ligand`: ligand file path (SDF/PDB)

### 6.3 `ce`
- `length`: RNA sequence length
- `alphabet`: RNA
- `batch`: sequences sampled per iteration
- `iters`: maximum iterations
- `elite_frac`: fraction used as elites
- `alpha`, `min_prob`, `temperature`: CE update parameters
- `topN`: export top N for inspection

### 6.4 `eval`
- `simrna_config`: path to SimRNA config.dat (e.g. `~/tools/SimRNA/config.dat`)
- `omp_threads`: 1 (when workers > 1)
- `simrna_iterations`: SimRNA sampling steps
- `simrna_frames`: e.g. `"1"`, `"2-3"`
- `decoy_agg`: `min`, `mean`, `p10`, `boltz`
- `vina_exhaustiveness`, `vina_cpu`, `grid_padding`
- Ligand conformer options: `ligand_conformers`, `ligand_seed`, `ligand_prune_rmsd`, `ligand_forcefield`, `ligand_optimize`

### 6.5 `regularize`
- `entropy_uniform_mix`
- `elite_min_hamming`
- `struct_unique`, `struct_metric`, `struct_min_dist`

### 6.6 `monitor`
- `penalty_skip.*`: skip updates when too many failures
- `early_stop.*`: stop on docking plateau
- `collapse.*`: detect distribution collapse
- `external_stop.*`: dual plateau detection

### 6.7 `rescore` (MM/GBSA feedback)

Core keys:
- `enabled`: true/false
- `top_frac`: fraction of batch to rescore
- `elite_frac`: fraction of rescored for feedback
- `include_penalty`: allow penalty candidates
- `gbsa_workers`: parallel MM/GBSA jobs
- `gbsa_charge_method`: `bcc` or `gas`

GB model:
- `mmgbsa_igb`: 5 (standard), 7/8 (GBn/GBn2)
- `mmgbsa_saltcon`: salt concentration (M)

**NEW: Minimization before MMPBSA**
- `mmgbsa_minimize`: true/false
- `mmgbsa_min_restraint`: true (recommended)
- `mmgbsa_min_maxcyc`: 1500
- `mmgbsa_min_ncyc`: 500
- `mmgbsa_min_restraint_wt`: 2.0
- `mmgbsa_min_restraint_mask`: `"!:LIG & !@H="`

Example configuration:

```yaml
rescore:
  enabled: true
  top_frac: 0.1
  elite_frac: 0.5
  gbsa_workers: 30
  gbsa_charge_method: bcc
  mmgbsa_igb: 5
  mmgbsa_saltcon: 0.150
  mmgbsa_minimize: true
  mmgbsa_min_restraint: true
  mmgbsa_min_maxcyc: 1500
  mmgbsa_min_ncyc: 500
  mmgbsa_min_restraint_wt: 2.0
  mmgbsa_min_restraint_mask: "!:LIG & !@H="
```

### 6.8 `md` (optional)
- `enabled`: true/false
- `topX`: number of sequences for MD
- GROMACS parameters

---

## 7) How to analyze results

### 7.1 Monitor progress

Check iteration state:
```bash
cat runs/my_run/checkpoint.json
```

View history:
```bash
tail -20 runs/my_run/history.csv
```

Real-time log:
```bash
tail -f runs/my_run/logs/ce_design.log
```

### 7.2 Docking results

```bash
head -n 10 runs/my_run/topN.csv
```

### 7.3 MM/GBSA results

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/my_run/rescore/iter_001/mmgbsa_results.csv')
n_ok = df['mmgbsa_ok'].astype(str).str.lower().isin(['true','1']).sum()
print(f'MMGBSA success: {n_ok}/{len(df)}')
if n_ok > 0:
    ok = df[df['mmgbsa_ok'].astype(str).str.lower().isin(['true','1'])]
    print(ok.nsmallest(5, 'mmgbsa_dg_bind')[['rank','seq','mmgbsa_dg_bind']].to_string(index=False))
"
```

### 7.4 Check charge and minimization

```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/my_run/rescore/iter_001/mmgbsa_results.csv')
cols = [c for c in ['rank','mmgbsa_charge','mmgbsa_charge_source','mmgbsa_minimized'] if c in df.columns]
print(df[cols].head(10).to_string(index=False))
"
```

### 7.5 Manually re-run MM/GBSA

```bash
python -m experiments.rescore_mmgbsa \
  --run_dir runs/my_run \
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
### 7.6 Extract the top scores of PBSA
Extract global Top-50:
```bash
python scripts/extract_top_results.py --run_dir runs/ce_run1 --top_n 50 --use_symlink
```
If you do not want repeated identical sequences in the Top-N:
```bash
python scripts/extract_top_results.py -r runs/ce_run3 -n 50 --unique_seq --use_symlink
```
This is often useful if your run has converged and the same sequence appears many times.

---

## 8) Common failure modes

### 8.1 All scores are `1e9` (penalty)

Check:
1. `topN.csv` error column
2. `logs/ce_design.log` tail
3. whether `fold/<hash>/*_AA.pdb` exists
4. whether `dock/<hash>/frame_*/lig_*/` contains `.vina.log`

### 8.2 MM/GBSA all failed (`mmgbsa_ok=False`)

**1) Wrong ligand charge:**
```bash
cat runs/my_run/ligand_ensemble/charge.txt
```

**2) Antechamber failed:**
```bash
tail -50 runs/my_run/rescore/iter_001/mmgbsa/rank_001_f*/antechamber.log
```

**3) tleap failed:**
```bash
tail -30 runs/my_run/rescore/iter_001/mmgbsa/rank_001_f*/tleap.log
```

**4) Minimization failed (only if enabled):**
```bash
tail -80 runs/my_run/rescore/iter_001/mmgbsa/rank_001_f*/sander_min.log
```

**Diagnosis:**
```bash
python -c "
import pandas as pd
df = pd.read_csv('runs/my_run/rescore/iter_001/mmgbsa_results.csv')
failed = df[df['mmgbsa_ok'].astype(str).str.lower() == 'false']
print(f'Failed: {len(failed)}/{len(df)}')
if len(failed) > 0:
    cols = [c for c in ['rank','mmgbsa_error','mmgbsa_workdir'] if c in failed.columns]
    print(failed[cols].head(3).to_string(index=False))
"
```

---

## 9) Visualization

```bash
python scripts/visualize_run.py --run_dir runs/my_run
```

Output plots:
- `plots/history_best_mean.png`
- `plots/history_penalty.png`
- `plots/mmgbsa_by_iter.png`
- `plots/mmgbsa_vs_vina.png`

---

## 10) Adapting to new molecules

### Quick adaptation (3 steps)

**Step 1: Prepare new ligand file**
```bash
# Place your ligand in data/
data/your_new_ligand.sdf
```

**Step 2: Determine charge**

| Molecule type | Typical charge |
|--------------|----------------|
| Phosphate compounds | -1 to -4 |
| Carboxylic acids | -1 to -2 |
| Sulfonic acids | -1 |
| Neutral molecules | 0 |
| Amines | +1 |

Check automatically:
```bash
python ligand/rdkit_conformers.py data/your_new_ligand.sdf --charge_only
```

**Step 3: Run with new parameters**
```bash
# Generate conformers
python ligand/rdkit_conformers.py data/your_new_ligand.sdf \
  --out_dir runs/my_new_run/ligand_ensemble \
  --n_confs 10 \
  --charge <YOUR_CHARGE>

# Update config
sed 's|data/PFHxS.sdf|data/your_new_ligand.sdf|' configs/run_default.yaml > configs/my_new_ligand.yaml
sed -i 's|runs/my_run|runs/my_new_run|' configs/my_new_ligand.yaml

# Run
bash scripts/run_ce_gbsa_loop.sh configs/my_new_ligand.yaml
```

### Optional tuning

**Large molecules:**
```yaml
eval:
  grid_padding: 12.0  # default 8.0
```

**Flexible molecules:**
```bash
--n_confs 20  # increase conformers
```

**High charge molecules (e.g. ATP -4):**
```yaml
rescore:
  mmgbsa_saltcon: 0.200  # increase salt
```

### Not supported (require special handling)

- Metal complexes (Fe³⁺, Zn²⁺)
- Covalent binders
- Very large molecules (>100 atoms)

---

## 11) CE theory mapping

- A. Initialize distribution → `dist.json`
- B. Sample batch → `ce.batch`
- C. Evaluate samples → SimRNA + Vina + aggregation
- D. Select elites → `elite_frac` + diversity
- E. Rescore topM → MM/GBSA
- F. Update distribution → feedback

---

## 12) Reproducibility notes

To reproduce results:
- Keep seed fixed and store `config_resolved.json`
- Archive: config file, `dist.json`, `checkpoint.json`, `topN.csv`, `history.csv`
- Archive: `ligand_ensemble/manifest.json` (includes charge info)
- Record software versions (RDKit, Vina, SimRNA, AmberTools)

Minimum debugging set:
- `topN.csv` (first 5 lines + error column)
- `logs/ce_design.log` (last 100 lines)
- `ligand_ensemble/charge.txt`
- One failing MM/GBSA workdir logs (`antechamber.log`, `tleap.log`, `mmpbsa.log`, and if enabled: `sander_min.log`)

---

## 13) Post-Design: Aptamer-Target Complex MD Analysis
After identifying top candidates via MM/GBSA rescoring, you can perform full-atom Molecular Dynamics (MD) simulations on the Aptamer-Ligand complex to evaluate binding stability, conformational changes, and hydrogen bond persistence.
The pipeline uses complex_md_amber.py, which automates the Amber-to-GROMACS topology conversion and running the MD production.

**How to run:**

```bash
# Example: Run MD for Top 1 candidates using 16 threads and GPU
python -m experiments.complex_md_amber \
    --run_dir runs/ce_run3 \
    --top 1 \
    --mdp_dir mdp \
    --threads 16 \
    --gpu
# If you want to view the estimated running time, please open a new terminal
# Enter the corresponding rank directory
cd runs/ce_run3/complex_md_amber/rank_001/

# View the currently running step
tail -f md.log
```
**Automated Workflow:**

1.Ligand Parameterization: Converts docking PDBQT to SDF, then to Mol2 using Antechamber (BCC charges, GAFF2).
2.Topology Assembly: Combines RNA (Amber OL3) and Ligand into a unified Amber topology via tleap.
3.GROMACS Conversion: Uses ParmEd to generate .top and .gro files with injected TIP3P water and ion definitions.
4.MD Protocol:
- Energy Minimization (Steepest Descent)
- NVT Equilibration (Temperature coupling)
- NPT Equilibration (Pressure coupling)
- Production MD (Length determined by your md.mdp)
Analysis: Automatically calculates RMSD and RMSF.

**Key Outputs:**
Each candidate gets a folder runs/run_name/complex_md_amber/rank_00x/:

- md.xtc: The simulation trajectory.
- complex.top & complex.gro: The starting topology and structure.

It is recommended to manually complete the RMSD and RMSF analysis in the future. Running the xvg files provided for RMSD and RMSF may result in selecting the wrong group.

---

## 14) Multi-Ligand Redocking + MM/GBSA（No Folding）
Utility script:
```
scripts/redock_and_mmgbsa_batch.py
```
Allows redocking multiple ligands against the best aptamer structure from a completed CE run, after 7.6 Extract the top scores of PBSA，select final_desult/summary_top50.csv，followed by automatic MM/GBSA rescoring.

No refolding is performed (uses best_receptor_pdb).
**Usage:**
```
python -m scripts.redock_and_mmgbsa_batch \
    --run_dir runs/ce_run1 \
    --ligands data/ligand1.sdf data/ligand2.sdf.sdf data/ligand3.sdf.sdf \
    --exhaustiveness 16 \
    --aptamer_topN 3 \
    --cpu 12 \
    --workers 16 \
    --minimize
```
Results are written to:
```
runs/<run_name>/redock_mmgbsa_batch/final_ranking.csv
```
Sorted by mmgbsa_dg_bind (lower = stronger binding).
