# Beyond Validation Accuracy: Temporal Generalization and Risk-Aware Decision Reliability in Bitcoin Fraud Detection

**Research implementation and reproducibility documentation**

This repository contains the experimental notebook for evaluating Bitcoin transaction fraud detection under chronological distribution changes, probability miscalibration, selective-screening risk, and delayed availability of transaction labels. The implementation uses the **Elliptic++ transaction dataset** and combines conventional machine-learning baselines, graph-based architectures, historical-context controls, frozen-model audits, and retrospective retraining experiments.

The central research question is not simply whether a classifier achieves high validation accuracy. It is whether its scores, ranking performance, and operational decisions remain reliable when evaluated on later Bitcoin transactions.

> **Research status:** This is a research evaluation pipeline, **not** a production fraud-detection system. Its risk estimates, historical threshold experiments, and uncertainty intervals do **not** provide prospective safety guarantees.

## Contents

1. [Research objectives](#research-objectives)
2. [Repository contents](#repository-contents)
3. [Dataset and access requirements](#dataset-and-access-requirements)
4. [Experimental design](#experimental-design)
5. [Environment and dependencies](#environment-and-dependencies)
6. [Running the notebook on Kaggle](#running-the-notebook-on-kaggle)
7. [Implementation stages](#implementation-stages)
8. [Key experimental findings](#key-experimental-findings)
9. [Generated artifacts and result verification](#generated-artifacts-and-result-verification)
10. [Interpretation and reproducibility constraints](#interpretation-and-reproducibility-constraints)
11. [GitHub publication checklist](#github-publication-checklist)
12. [Citation, licensing, and contact](#citation-licensing-and-contact)

## Research objectives

The notebook investigates five related questions:

1. **Predictive generalization:** How do models trained on earlier Bitcoin transactions perform on chronologically later observations?
2. **Representation reliability:** Do graph connections, graph-feature fusion, or past-only historical context provide consistent improvements over transaction-level features?
3. **Probability reliability:** Does improved global probability calibration imply accurate risk estimates among transactions selected for automatic clearance?
4. **Operational screening:** How do clearance thresholds affect illicit transactions missed, manual-review workload, and illustrative decision costs?
5. **Temporal adaptation:** How do retraining frequency, historical window length, label delays, and sequential recalibration affect late-period performance?

The implementation deliberately distinguishes **ranking discrimination**, **probability calibration**, and **screening outcomes**. These are different evaluation targets and should not be treated as interchangeable.

## Repository contents

The supplied repository artifact is:

```text
.
├── README.md
└── Bitcoin_Fraud_Blockchain.ipynb
```

The notebook contains **145 cells**, including data checks, model development, evaluation, retrospective sensitivity analyses, audit procedures, and output packaging. It is a **stateful research notebook**: later cells depend on variables and files created earlier. It is not a modular Python package or a command-line application.

The Elliptic++ dataset, pretrained checkpoints, intermediate CSV files, and exported figures are **not assumed to be included** in this repository. Do not advertise them as available unless they are explicitly committed or linked.

## Dataset and access requirements

### Dataset

The notebook expects the Elliptic++ transaction files to be attached to a Kaggle notebook as an input dataset. The implementation explicitly loads:

| File | Purpose |
|---|---|
| `txs_features.csv` | Transaction attributes and chronological snapshot information |
| `txs_classes.csv` | Transaction labels |
| `txs_edgelist.csv` | Transaction graph edges |

The notebook currently uses the following **hard-coded Kaggle input directory**:

```python
DATA_DIR = Path("/kaggle/input/datasets/muhammadabrar78/elliptic-dataset")
```

The dataset must be attached under that path, **or this assignment must be changed** to match the actual mounted Kaggle input directory. Use the notebook's dataset-discovery cell to inspect available CSV paths before modifying the path.

The dataset should be obtained from its legitimate host, subject to its own licensing and attribution requirements. This repository does not imply ownership of the dataset or permission to redistribute it.

### Schema assumptions

The implementation validates the following fields:

- `txId` in the transaction features and class files;
- `class` in the class file;
- `txId1` and `txId2` in the edge file;
- `Time step` (including the space) as the chronological field;
- `Local_feature_1` through `Local_feature_93` as the primary transaction-level feature set.

The code maps source classes as follows:

| Source class | Research label | Meaning |
|---|---:|---|
| `1` | `1` | Illicit |
| `2` | `0` | Licit |
| `3` | `-1` | Unknown / excluded from supervised evaluation |

Unknown-label transactions may still be used where the notebook explicitly permits unlabeled graph representation construction; they must not be treated as labeled negative examples.

The notebook's recorded data checks report **203,769 transactions**, **49 time steps**, and **93 selected local features** for the principal classification experiments. It also inspects additional dataset attributes, but those should not be confused with the 93-feature primary model input.

### Chronological partitioning

The principal frozen-model protocol is:

| Partition | Time steps | Purpose |
|---|---|---|
| Training | 1–29 | Model fitting and training-only preprocessing |
| Validation | 30–34 | Development-period model comparison |
| Calibration | 35–37 | Platt fitting and historical threshold analysis |
| Assessment | 38–39 | Post-calibration assessment and specified operating-point diagnostics |
| Locked test | 40–49 | Chronological evaluation of the finalized frozen models |
| Late-test analysis | 43–49 | Retrospective adaptation and deterioration analysis |
| Sequential calibration audit | 44–49 | Historical out-of-sample calibration predictions |

**Do not randomize these chronological partitions** when reproducing the principal experiment. Randomized splits appearing later in the notebook are explicitly labeled **exploratory positive controls**, not substitutes for chronological testing.

## Experimental design

### Predictive models

The notebook evaluates or audits the following model families:

- Logistic Regression, Random Forest (RF), and XGBoost;
- Multilayer Perceptron (MLP);
- Graph Convolutional Network (GCN) and GraphSAGE;
- Separately trained no-edge GraphSAGE and inference-time edge removal;
- Adaptive and residual graph-feature fusion;
- Past-only temporal-conditioned MLP and zero-context control;
- Historical graph-memory conditioning and zero-memory control.

Graph connections are evaluated within temporal snapshots. The presence of a graph architecture should **not** be interpreted as proof of cross-snapshot temporal message passing.

The main RF configuration uses 300 trees, maximum depth 20, minimum split size 5, minimum leaf size 2, square-root feature sampling, and `balanced_subsample` class weights.

### Calibration and screening

The frozen RF's raw probabilities are compared with Platt-calibrated probabilities. Evaluation includes:

- Average precision (AP) and ROC-AUC;
- Brier score, logarithmic loss, and 15-bin expected calibration error (ECE-15);
- Clearance coverage, illicit auto-clearance rate, and manual-review share;
- Calibration in the automatically cleared low-risk population;
- Matched-coverage comparisons to separate score ordering from threshold effects;
- Illustrative cost sensitivity, historical threshold selection, and confidence-interval diagnostics.

The primary three-action policy is:

| Decision | Score condition |
|---|---|
| Automatically clear | `score < 0.02` |
| Manual review | `0.02 <= score <= 0.50` |
| Automatically flag | `score > 0.50` |

These are **research operating points**, not recommended real-world risk tolerances. In particular, the illicit auto-clearance fraction and the illicit prevalence **among cleared transactions** have different denominators.

### Label-delay-aware adaptation

For a prediction at snapshot `t` and delay `d`, retraining may use labeled data only through `t - 1 - d`. The explored delays are `d = 0, 1, 2, 3`.

The notebook compares:

- A single historical RF refit;
- Expanding-history RF updated every snapshot;
- Expanding-history RF updated every two or three snapshots;
- Rolling-history RF with a 10-snapshot window;
- Rolling-history sensitivity with `W = 5, 10, 15, 20`;
- A 2 × 2 comparison of training-history length and refitting frequency;
- Sequential, historically fitted Platt calibration;
- Matched-coverage screening and a retrospective prevalence-shift sensitivity analysis.

The principal adaptation comparisons use seeds **42, 123, and 2026**. Snapshot-block bootstrap analyses use **2,000 replicates** where specified in the notebook.

## Environment and dependencies

### Recommended execution platform

**Kaggle Notebooks** is the intended environment because the code uses absolute `/kaggle/input/` and `/kaggle/working/` paths and relies on a stateful notebook execution sequence.

The manuscript's recorded environment includes:

| Component | Recorded environment |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| PyTorch Geometric | 2.8.1 |
| GPU | NVIDIA Tesla T4, where available |

These are **recorded versions**, not a tested cross-platform compatibility matrix. Kaggle image versions may change.

Other imported packages include `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`, `xgboost`, `joblib`, and notebook display utilities. Some cells use additional standard-library modules.

The notebook includes a cell that installs `torch-geometric` using `pip` without explicitly upgrading PyTorch. For a fresh environment, verify that PyTorch, CUDA, and PyTorch Geometric versions are compatible before executing graph-model cells. Internet access or a preinstalled compatible package may be required.

A fully pinned `requirements.txt` or `environment.yml` has **not** been validated for this uploaded notebook; do not claim that dependency installation is guaranteed to reproduce every historical result.

## Running the notebook on Kaggle

### Step 1 — Create the notebook

1. Sign in to Kaggle and create a new notebook.
2. Upload or import `Bitcoin_Fraud_Blockchain.ipynb`.
3. Select a Python runtime. Enable a GPU when available for the neural and graph experiments.
4. Attach the Elliptic++ dataset using **Add Input**.

### Step 2 — Confirm the input path

Run the initial dataset-discovery cells. Confirm that the three required CSV files are present.

If Kaggle mounts the dataset at a different directory, update **only the path configuration** in the dataset-loading cell, for example:

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/input/<your-mounted-dataset-directory>")
```

The placeholder must be replaced with the actual directory discovered in the Kaggle runtime. Do not change column names to match an assumed schema: inspect the files first.

### Step 3 — Execute the data integrity checks

Run the loading, merge, label-mapping, chronological validation, and feature-audit cells. Check that:

- transaction IDs are unique where required;
- the class mapping is correct;
- the `Time step` column contains the expected snapshots;
- the 93 selected local features exist;
- no unsupported missing or infinite values remain in the model matrix;
- preprocessing is fitted only on the intended training partition.

Stop if an assertion fails. Do not bypass failed assertions merely to obtain a final metric.

### Step 4 — Execute model development

Run the conventional baselines, neural/graph baselines, and structural/historical controls in notebook order. GPU memory, package compatibility, and training time can affect whether the complete notebook finishes in a single Kaggle session.

The notebook saves several fitted-model checkpoints and intermediate artifacts under `/kaggle/working/`. Preserve these files if later audit cells require them.

### Step 5 — Execute frozen-model evaluation

Run the calibration, threshold-selection, saved-artifact checks, and chronological locked-test cells. **Preserve the separation between development, assessment, and test periods.**

Some later cells explicitly reload frozen checkpoints and validate artifact provenance. If a checkpoint is absent, determine which earlier training/export cell produces it. Do not silently replace a missing frozen checkpoint with a newly trained model and call the result an untouched test.

### Step 6 — Execute the retrospective extensions

The notebook's later numbered experiments are particularly important:

| Notebook step | Experiment / purpose |
|---|---|
| 97–110 | Snapshot discrimination, calibration, temporal risk, uncertainty, and reviewer-oriented audits |
| 111–114 | Label-delay-aware RF retraining, screening mitigation, and consolidated predictions |
| 115 | Multi-seed RF strategy comparison, including single-refit and rolling/expanding baselines |
| 116C–116D | Rolling-window sweep and performance aggregation |
| 117A–117B | History-length × update-frequency factorial comparison |
| 118A–118B | Paired snapshot-block bootstrap |
| 119 | Sequential Platt calibration and matched-coverage audit |
| 120 | Cross-snapshot ranking-mechanism audit |
| 121 | Retrospective prevalence-shift sensitivity and associated diagnostics |

**Dependency warning:** Later steps read files produced by preceding steps. For example, Step 119 reads `step114_rf_control_predictions.csv`; Step 120 reads Step 119 outputs; Step 121 reads Steps 119 and 120 outputs. Running these out of order without the required artifacts will fail.

The final cells package the Kaggle working directory into an archive. Inspect its contents and size before publishing or sharing it.

### Step 7 — Export results

After successful execution, open the Kaggle **Output** panel and download the relevant generated CSV files, figures, and logs. Save the notebook version and record its Kaggle runtime settings.

For GitHub, prefer small reproducibility tables, code, and documented figures. Large checkpoints, dataset copies, and full working-directory archives should be distributed separately when licensing and repository-size constraints permit.

## Key experimental findings

The following figures are **reported by the supplied notebook and accompanying analysis**. They are not claims of an independent fresh execution in the reader's environment.

### Validation versus chronological test

| Metric | Development validation | Chronological locked test |
|---|---:|---:|
| RF average precision | 0.9894 | 0.6450 |
| RF ROC-AUC | 0.9968 | 0.8363 |

These periods have different illicit prevalence, so AP changes should be interpreted with that distinction in mind. The ROC-AUC decline additionally indicates reduced ranking discrimination.

### Calibration versus screening

| Quantity | Raw RF | Platt RF |
|---|---:|---:|
| Locked-test Brier score | 0.0350 | 0.0273 |
| Locked-test ECE-15 | 0.0422 | 0.0206 |
| Fixed-threshold auto-clearance coverage | 46.38% | 80.73% |
| Illicit transactions auto-cleared | 94 / 636 | 202 / 636 |
| Illicit auto-clearance rate | 14.78% | 31.76% |

The fixed-threshold policies operate at **different coverage**. At matched coverage, the frozen raw and Platt-transformed RF scores yield the same illicit-clearance counts in the reported evaluation. Better aggregate calibration must therefore **not** be described as an independent improvement in matched-coverage fraud screening.

At the calibrated model's 80.73% clearance operating point, the reported mean predicted illicit probability among cleared transactions was **0.27%**, compared with **2.24% observed** illicit prevalence.

### Late-period RF adaptation

For the three-seed comparison on steps **43–49** with immediate historical labels (`d = 0`):

| Strategy | Mean late ROC-AUC | Mean late AP |
|---|---:|---:|
| Single historical refit | 0.7661 | 0.0546 |
| Expanding-history, every snapshot | 0.9011 | 0.1845 |
| Rolling-history, W = 10 | 0.9400 | 0.3849 |

These numbers belong to the **three-seed late-period strategy comparison**. Do not substitute the separately reported single-seed expanding-history AUC of **0.9004** or pooled full-test values into this table without identifying the different evaluation protocol.

### Rolling-window sensitivity: verified values

The actual Step 116 window-sweep summary reports the following mean late-period AP:

| Label delay | W = 5 | W = 10 | W = 15 | W = 20 |
|---|---:|---:|---:|---:|
| d = 0 | **0.4881** | 0.3849 | 0.3329 | 0.3001 |
| d = 1 | **0.2328** | 0.1997 | 0.1645 | 0.1565 |
| d = 2 | 0.1242 | **0.1509** | 0.1214 | 0.1147 |
| d = 3 | 0.1010 | **0.1253** | 0.1027 | 0.0924 |

**Critical figure-verification instruction:** Any GitHub or manuscript figure labeled `Average Precision vs. Rolling-Window Length` must use **this table**, not an earlier illustrative figure with different `d = 1, 2, 3` values. The notebook's Step 116 output CSV is the preferred source for generating the final chart.

The strongest AP window is **W = 5 for delays 0 and 1**, and **W = 10 for delays 2 and 3**. The results do not establish a universally optimal history length. Window comparisons reuse the observed late-test period and are retrospective.

## Generated artifacts and result verification

The notebook writes numerous artifacts under `/kaggle/working/`, including checkpoint files, figures, and CSV reports. The following are especially useful for independent review:

| File | Meaning |
|---|---|
| `step114_rf_control_predictions.csv` | Prediction-level records for RF adaptation policies |
| `step114_rf_fit_log.csv` | RF fitting history and training-window metadata |
| `step115_rf_per_seed.csv` | Per-seed strategy metrics |
| `step115_rf_3seed_summary.csv` | Three-seed adaptation comparison |
| `step116_window_sweep_predictions.csv` | Prediction-level rolling-window sweep results |
| `step116_window_sweep_per_seed.csv` | Window sweep metrics by seed |
| `step116_window_sweep_summary.csv` | Mean/SD window sweep metrics |
| `step117_factorial_summary.csv` | Training-history × refitting-frequency results |
| `step117_factorial_contrast_summary.csv` | Factorial contrasts |
| `step118_paired_block_bootstrap.csv` | Paired snapshot-block uncertainty summary |
| `step119_rolling_platt_summary.csv` | Sequential calibration summary |
| `step119_calibration_audit.csv` | Snapshot-level calibration parameters and prevalence audit |
| `step119_matched_coverage.csv` | Raw-versus-calibrated matched-coverage results |
| `step119_exact_predictions_audit.csv` | Prediction-level calibration audit |
| `step120_exact_mechanism_audit.csv` | Cross-snapshot ranking-mechanism audit |
| `step121_exact_prevalence_sensitivity.csv` | Retrospective prevalence sensitivity |
| `step121_exact_a1_summary.csv` | A1 diagnostic summary |

**Verification procedure:** Recompute tables from the exported CSV files, checking seed, delay, policy, evaluation snapshots, and metric denominators. Do not use rounded manuscript numbers as the input for a new statistical analysis.

For example, after downloading the Step 116 summary, inspect its schema before filtering:

```python
import pandas as pd

results = pd.read_csv("step116_window_sweep_summary.csv")
print(results.columns.tolist())
print(results.to_string(index=False))
```

This avoids assuming column names or silently misaligning the results.

### Artifact provenance

The notebook contains frozen-model and checksum-related audit steps. These are useful for distinguishing saved-model evaluation from newly trained retrospective variants. When publishing results, document:

- the exact notebook revision;
- the dataset source and mounted path;
- the seeds used;
- the evaluation snapshot range;
- whether a model was frozen or refitted;
- whether results were obtained before or after inspection of test outcomes;
- the source CSV behind each figure and table.

## Interpretation and reproducibility constraints

1. **No prospective guarantee.** Historical threshold bounds and observed illicit miss rates are not certified future risk limits.
2. **Exploratory test reuse.** Later architecture audits, window sweeps, adaptation studies, and calibration diagnostics reuse an already examined chronological test period.
3. **Limited temporal uncertainty.** The paired late-period bootstrap resamples only seven snapshots; intervals describe sensitivity to the observed blocks, not independent future performance.
4. **Calibration is not ranking improvement.** A monotonic global Platt transform preserves within-model ranking, while sequential snapshot-specific calibrators can alter pooled cross-snapshot ordering.
5. **Label availability matters.** `d = 0` assumes labels from the immediately preceding snapshot are available before prediction. Longer delays reduce the eligible history.
6. **Operational costs are hypothetical.** Cost weights are illustrative and do not represent verified financial losses or investigator outcomes.
7. **No simulated investigation outcomes.** The manual-review action is a workload category; the notebook does not establish how investigators would resolve reviewed transactions.
8. **Snapshot class imbalance.** Some late snapshots contain very few illicit examples, making individual-snapshot AP and related metrics unstable.
9. **Not a one-command package.** The notebook uses in-memory variables, absolute Kaggle paths, saved artifacts, and sequential experiment dependencies.
10. **Environment reproducibility is conditional.** A matching dataset, compatible packages, sufficient resources, and all required artifacts are necessary; identical results across every environment are not guaranteed.

## Citation, licensing, and contact

**Study title:** *Beyond Validation Accuracy: Temporal Generalization and Risk-Aware Decision Reliability in Bitcoin Fraud Detection*

**Research implementation:** `Bitcoin_Fraud_Blockchain.ipynb`

If this implementation contributes to your research, cite the associated paper **when its verified bibliographic details are available**, and cite the original Elliptic++ dataset according to its official attribution instructions. No DOI, journal acceptance, or publication status is asserted here.

**Software license:** Not specified in the supplied repository materials. Until a license is selected and added, do not assume permission for unrestricted reuse.

---

*This README documents the uploaded implementation and its recorded experiments. It does not represent an independently rerun or externally validated reproduction of every notebook cell.*
