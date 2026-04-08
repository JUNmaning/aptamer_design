# Aptamer Design Pipeline (CE + Multi-frame SimRNA + Vina Docking + MMGBSA Feedback Loop + Optional MD)

Author: maning

This repository implements an end-to-end, reproducible RNA aptamer design pipeline driven by a Cross-Entropy (CE) optimizer.  
At each CE iteration, the pipeline samples RNA sequences from a probability distribution, evaluates them via RNA structure sampling (SimRNA) and docking (AutoDock Vina) using multiple RNA decoy frames and multiple ligand conformers, then optionally runs MMGBSA rescoring on top candidates and feeds those scores back to update the CE distribution for the next iteration.

The workflow is restartable/resumable and supports cheap smoke tests before expensive rescoring.

---

## 1) What this pipeline does

Given a ligand structure (SDF/PDB), the pipeline performs:

1. Ligand ensemble generation (RDKit conformers)
2. RNA structure sampling (ViennaRNA + SimRNA trajectory)
3. Multi-frame decoy export (SimRNA `.trafl` → multiple all-atom PDBs)
4. Docking (Vina) across RNA frames × ligand conformers
5. Aggregation across frames/conformers (e.g. `min`)
6. CE update using docking elites
7. Optional MMGBSA rescoring for top candidates
8. Feedback update: MMGBSA results update `dist.json`
9. Resume to the next CE iteration until stopping criteria are met

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
├── topN.csv                      # exported top ranked candidates (for inspection)
├── next_batch.txt                # next batch generated after GBSA feedback
├── ligand_ensemble/              # ligand conformers + manifest
├── fold/<eval_hash>/             # SimRNA outputs per sequence
│   ├── aptamer.trafl
│   ├── aptamer-000001_AA.pdb
│   ├── aptamer-000002_AA.pdb
│   └── ...
├── dock/<eval_hash>/frame_XXX/
│   └── lig_YYY/
│       ├── *.dock.pdbqt
│       ├── *.vina.log
│       └── ...
├── rescore/iter_XXX/
│   ├── manifest.jsonl            # candidates selected for MMGBSA
│   ├── mmgbsa_results.csv        # raw MMGBSA results
│   ├── mmgbsa_results.jsonl
│   ├── results.csv               # normalized file for GBSA feedback
│   ├── elites_for_ce.txt         # elites selected in GBSA feedback
│   └── feedback.done             # marker file for completed feedback
└── md/                           # optional MD results
    ├── rank_00x_<eval_hash>/
    └── md_summary.csv
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
- OpenBabel (obabel)
- ViennaRNA (RNAfold)
- Optional MD: GROMACS (gmx)

Quick check:

```bash
which RNAfold || echo "RNAfold not found"
which SimRNA || echo "SimRNA not found"
which SimRNA_trafl2pdbs || echo "SimRNA_trafl2pdbs not found"
which obabel || echo "obabel not found"
which vina || echo "vina not found"
which tleap || echo "tleap not found"
which MMPBSA.py || echo "MMPBSA.py not found"
```

---

## 4) Configuration explained (`run_default.yaml`)

Example config:

```yaml
run:
  run_dir: runs/ce_run1
  seed: 123
  log_level: INFO
  workers: 30

inputs:
  ligand: data/PFHxS.sdf

ce:
  length: 30
  alphabet: RNA
  batch: 500
  iters: 50
  elite_frac: 0.1
  alpha: 0.4
  min_prob: 0.02
  temperature: 1.0
  topN: 50

eval:
  simrna_config: /root/autodl-tmp/SimRNA/config.dat
  omp_threads: 1
  simrna_iterations: 20000
  simrna_frames: "1-3"
  decoy_agg: "min"
  vina_exhaustiveness: 8
  vina_cpu: 1
  grid_padding: 8.0

  ligand_conformers: 2
  ligand_seed: 123
  ligand_prune_rmsd: 0.5
  ligand_forcefield: "MMFF"
  ligand_optimize: true
  ligand_max_opt_iters: 200

regularize:
  entropy_uniform_mix: 0.02
  elite_min_hamming: 0.10
  struct_unique: false
  struct_metric: bp
  struct_min_dist: 0.0

monitor:
  penalty_skip:
    enabled: true
    frac_thr: 0.95
    action: skip
    reset_patience: 2

  early_stop:
    enabled: true
    patience: 4
    min_improve: 0.01

  collapse:
    enabled: true
    max_prob: 0.95
    frac_thr: 0.60
    patience: 5

  external_stop:
    gbsa_patience: 5
    gbsa_min_improve: 0.1

rescore:
  enabled: true
  top_frac: 0.1
  elite_frac: 0.5
  include_penalty: false

md:
  enabled: false
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

## Configuration Parameters

The pipeline is controlled by a YAML configuration file. The main sections are described below.

---

### 4.1 `run`
Controls general run behavior.

- `run_dir`  
  Output directory for all run artifacts, including logs, checkpoints, history, rescoring results, and optional MD outputs.

- `seed`  
  Global random seed used for reproducibility.

- `workers`  
  Number of parallel workers used for sequence evaluation.  
  Each worker evaluates different sequences independently.

- `log_level`  
  Logging verbosity, e.g. `INFO`, `DEBUG`, `WARNING`.

- `resume` *(if present in config)*  
  Whether to resume from an existing checkpoint and cached results.

---

### 4.2 `inputs`
Defines required input files.

- `ligand`  
  Path to the ligand file (`.sdf` or `.pdb`).  
  It is recommended to provide this explicitly.

  The pipeline first tries `inputs.ligand`.  
  If this is missing or invalid, some code paths may fall back to:
  - `data/ligand.sdf`
  - `data/ligand.pdb`

---

### 4.3 `ce`
Controls the Cross-Entropy (CE) optimization loop.

- `length`  
  RNA sequence length.

- `alphabet`  
  Sequence alphabet, typically `RNA`.

- `batch`  
  Number of sequences sampled and evaluated per CE iteration.

- `iters`  
  Maximum number of CE iterations.

- `elite_frac`  
  Fraction of the docking-evaluated batch used as CE elites for updating the sequence distribution.

- `alpha`  
  CE update smoothing factor.  
  Larger values make updates more aggressive.

- `min_prob`  
  Minimum probability floor for each nucleotide at each position.

- `temperature`  
  Sampling temperature for sequence generation.

- `topN`  
  Number of top candidates exported to `topN.csv`.

**Important:**  
`topN` is only used for exporting and inspection.  
It does **not** control how many sequences are used to update the CE distribution.

The CE distribution is updated using the **docking elite set**, controlled by `ce.elite_frac`.

---

### 4.4 `eval`
Controls structure generation and docking evaluation.

- `simrna_config`  
  Path to the SimRNA configuration file, if required by your environment.

- `omp_threads`  
  Number of OpenMP threads used in folding-related tools.  
  When `workers > 1`, this is typically kept at `1` to avoid CPU oversubscription.

- `simrna_iterations`  
  Number of SimRNA sampling steps.  
  Larger values may improve structure sampling but increase runtime.

- `simrna_frames`  
  Which exported frames are used for docking, e.g. `"1"` or `"1-3"`.

- `decoy_agg`  
  Aggregation method across multiple frames / decoys / ligand conformers.  
  Supported values include:
  - `min`
  - `mean`
  - `p10`
  - `boltz`

- `vina_exhaustiveness`  
  AutoDock Vina exhaustiveness parameter.

- `vina_cpu`  
  Number of CPU threads used by each Vina docking task.

- `grid_padding`  
  Docking box padding in angstroms.

- `ligand_conformers`  
  Number of ligand conformers to generate before docking.

- `ligand_seed`  
  Random seed for ligand conformer generation.

- `ligand_prune_rmsd`  
  RMSD threshold for pruning similar ligand conformers.

- `ligand_forcefield`  
  Force field used for conformer optimization, e.g. `MMFF`.

- `ligand_optimize`  
  Whether to optimize generated ligand conformers.

- `ligand_max_opt_iters`  
  Maximum optimization iterations for ligand conformer refinement.

---

### 4.5 `regularize`
Controls regularization and diversity preservation during CE updates.

- `entropy_uniform_mix`  
  Uniform-mixing regularization strength.  
  A small fraction of the distribution is mixed with a uniform distribution to prevent premature collapse.

- `elite_min_hamming`  
  Minimum normalized Hamming distance required between selected elite sequences.

- `struct_unique`  
  If enabled, prevents selecting multiple elites with the same dot-bracket structure.

- `struct_metric`  
  Structural distance metric used when enforcing structural diversity.  
  Supported values:
  - `bp`   : base-pair set distance
  - `edit` : normalized dot-bracket mismatch

- `struct_min_dist`  
  Minimum structural distance required between elites.

**Important:**  
The current code uses:
- `entropy_uniform_mix`
- `elite_min_hamming`
- `struct_unique`
- `struct_metric`
- `struct_min_dist`

Parameters such as `elite_diverse` or `diversity_weight` are **not used** unless you explicitly add code to support them.

---

### 4.6 `monitor`
Controls stopping criteria and stability monitoring.

#### Penalty handling
- `penalty_skip.enabled`  
  Whether to skip CE updates when too many sampled sequences fail.

- `penalty_skip.frac_thr`  
  If the fraction of penalty-scored sequences exceeds this threshold, the CE update is skipped.

- `penalty_skip.action`  
  Action taken when repeated penalty rounds occur.  
  Common options:
  - `skip`
  - `reset_uniform`

- `penalty_skip.reset_patience`  
  Number of consecutive high-penalty iterations before resetting to a uniform distribution (if enabled).

#### Internal CE stopping
- `early_stop.enabled`  
  Whether to stop the CE loop when improvement stalls.

- `early_stop.patience`  
  Number of consecutive iterations allowed without sufficient improvement.

- `early_stop.min_improve`  
  Minimum required improvement in the global best docking score.

#### Collapse detection
- `collapse.enabled`  
  Whether to monitor distribution collapse.

- `collapse.max_prob`  
  Probability threshold used to define a collapsed position.

- `collapse.frac_thr`  
  Fraction of positions allowed to exceed `max_prob` before collapse is considered significant.

- `collapse.patience`  
  Number of consecutive collapsed iterations before stopping.

#### External loop stopping
These parameters are used by the outer CE↔MMGBSA loop script (e.g. `scripts/run_ce_gbsa_loop.sh`).

- `external_stop.gbsa_patience`  
  Number of MMGBSA iterations checked for GBSA plateau detection.

- `external_stop.gbsa_min_improve`  
  Minimum MMGBSA improvement required to avoid GBSA plateau stop.

- `external_stop.dock_patience`  
  Number of iterations checked for docking plateau detection in the external loop.

- `external_stop.dock_min_improve`  
  Minimum docking improvement required to avoid external docking plateau stop.

**Note:**  
`early_stop` already monitors docking improvement inside `ce_design.py`.  
Therefore, `external_stop.dock_*` may overlap with `early_stop`.  
In many cases, it is sufficient to use:
- internal `early_stop` for docking
- external `gbsa_*` for MMGBSA

---

### 4.7 `rescore`
Controls MMGBSA rescoring and feedback selection.

- `enabled`  
  Whether rescoring is enabled.

- `include_penalty`  
  Whether penalty/failure candidates are allowed into rescoring.  
  Usually set to `false`.

- `top_frac` *(optional)*  
  Fraction of the CE batch selected for MMGBSA rescoring.

- `elite_frac` *(optional)*  
  Fraction of the MMGBSA candidate set used for feedback update.

**Typical logic:**  
1. CE evaluates a batch with docking  
2. Top docking candidates are rescored by MMGBSA  
3. Top MMGBSA candidates are used to guide the next feedback update

---

### 4.8 `md`
Controls optional post-run MD refinement.

- `enabled`  
  Whether MD postprocessing is enabled.

- `topX`  
  Number of top sequences selected for MD simulation.

- `gmx`  
  GROMACS executable name, e.g. `gmx`.

- `mdp_dir`  
  Directory containing the required `.mdp` files.

- `forcefield`  
  Force field used for topology generation.

- `water`  
  Water model used in GROMACS setup.

- `box_type`  
  Box shape, e.g. `cubic`.

- `box_distance_nm`  
  Minimum solute-to-box boundary distance in nm.

- `salt_conc_m`  
  Salt concentration in mol/L.

- `neutralize`  
  Whether to neutralize the system.

- `threads`  
  Number of CPU threads for MD.  
  `0` usually lets GROMACS decide automatically.

- `gpu`  
  Whether to enable GPU acceleration for MD runs.

- `skip_if_done`  
  Skip MD reruns if output files already exist.

- `strip_5prime_phosphate`  
  Whether to remove 5' phosphate atoms before `pdb2gmx`, which can improve RNA topology generation compatibility.

- `strip_3prime_phosphate`  
  Whether to remove 3' phosphate atoms before `pdb2gmx`.

---

## Practical Notes

- `topN.csv` stores the top docking candidates only.  
  It does not automatically contain MMGBSA results.

- MMGBSA results are written under:
  `run_dir/rescore/iter_*/mmgbsa_results.csv`

- If you use an outer CE↔MMGBSA loop, the effective rescoring size is typically controlled by:
  - `rescore.top_frac`
  - `rescore.elite_frac`

- If both `early_stop` and `external_stop.dock_*` are enabled, docking plateau may be monitored twice.

- Some YAML keys may look reasonable but are not actually consumed unless the code explicitly reads them. Always check the implementation if you add new parameters.

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

This is the important part.

Per CE iteration:
- Sample ce.batch sequences from current dist.json
- Evaluate each sequence by SimRNA + docking
- Rank by docking score
- Select docking elites using ce.elite_frac
- Update CE distribution once from docking elites
- Export top rescoring candidates (rescore.topM or rescore.top_frac)
- Run MMGBSA on those candidates
- Select GBSA elites (rescore.elite_n or rescore.elite_frac)
- Update dist.json again using GBSA feedback
- Generate the next batch from the updated distribution

Therefore:
- Docking elites drive the main CE update
- MMGBSA elites provide a second, smaller feedback update
- topN.csv is only an exported view of best candidates, not the CE update set


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

### 11.3 Sort by the best GBSA from small to large and output the top 50

```bash
python scripts/export_mmgbsa_best.py --run_dir runs/ce_run1
```

### 11.4 Molecular Dynamics : `/runs/ce_run1/md`
The final summary results are presented in this document
- `/runs/ce_run1/md/md_summary.csv`
Contains information such as ranking, sequence, score, etc

---

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

