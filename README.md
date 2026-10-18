# Beyond Validation Accuracy: Temporal Generalization and Risk-Aware Decision Reliability in Bitcoin Fraud Detection

Research implementation and experimental notebook for evaluating **temporal generalization, probability calibration, and operational screening reliability** on the **Elliptic++ Bitcoin transaction dataset**.

The central question is whether strong validation-set fraud detection metrics translate into dependable decisions on later, unseen transaction periods. This project evaluates conventional classifiers, graph neural networks, graph fusion, past-only temporal representations, feature dependence, and three-action fraud screening. It is an **empirical evaluation framework**, not a claim of certified fraud-risk control or a newly invented fraud classifier.

## Repository contents

```text
.
├── README.md
└── Fraud_detection_kaggle_notbook.ipynb
```

The notebook contains the complete sequential experimental workflow, including its recorded outputs. Additional generated artifacts (frozen models, protocol JSON, CSV results) are written to Kaggle's `/kaggle/working/` directories during execution and are **not assumed to be included** in this repository.

## Dataset

The notebook uses the Elliptic++ transaction files attached in Kaggle under:

```text
/kaggle/input/datasets/muhammadabrar78/elliptic-dataset/
```

Key input files:

- `txs_features.csv` — transaction features and time steps
- `txs_classes.csv` — transaction labels
- `txs_edgelist.csv` — transaction-to-transaction edges

The Kaggle input also includes wallet and address-related files; the primary reported transaction classification experiments use the transaction data above.

| Dataset characteristic | Value |
|---|---:|
| Transactions | 203,769 |
| Transaction edges | 234,355 |
| Temporal snapshots | 49 |
| Local input features | 93 |
| Labeled licit transactions | 42,019 |
| Labeled illicit transactions | 4,545 |
| Unknown-label transactions | 157,205 |

Unknown-label nodes can participate in graph construction but are excluded from supervised loss and labeled evaluation metrics. The 93 local features are standardized with a `StandardScaler` fitted **only on transactions from training snapshots 1–29**; the scaler is then applied unchanged to later periods.

### Chronological evaluation protocol

| Split | Snapshots | Purpose |
|---|---|---|
| Training | 1–29 | Fit predictive models and training-only preprocessing |
| Validation | 30–34 | Compare models and select development configurations |
| Calibration | 35–37 | Fit Platt calibrator and select historical thresholds |
| Exploratory assessment | 38–39 | Examine post-calibration behavior before the final freeze |
| Locked test | 40–49 | Independent evaluation of frozen models and policies |

The locked test is not used to refit models, recalibrate scores, or modify decision thresholds.

## Experimental workflow

1. Load and audit transaction data, labels, features, and temporal graph structure.
2. Establish chronological partitions and fit training-only feature standardization.
3. Train and compare logistic regression, Random Forest, XGBoost, multilayer perceptron (MLP), graph convolutional network (GCN), and GraphSAGE.
4. Evaluate graph-edge removal, adaptive and residual graph fusion, past-only temporal conditioning, and historical graph-memory controls.
5. Audit feature-group dependence (including local features 41–60), redundancy, and temporal covariate variation.
6. Fit post-hoc Platt scaling to the selected Random Forest using the designated calibration period.
7. Compare fixed and exploratory historical-threshold three-action policies: **auto-clear**, **manual review**, and **auto-flag**.
8. Freeze artifacts and evaluation rules, verify SHA-256 hashes, and run the independent locked-test evaluation.

The experiments emphasize **average precision (PR-AUC)**, ROC-AUC, F1, calibration error, fraud auto-clear fraction, false flags, review workload, and illustrative decision cost.

## Implementation environment

The saved notebook's execution outputs report:

| Component | Recorded environment |
|---|---|
| Platform | Kaggle Notebook |
| Python | 3.13.15 |
| GPU | NVIDIA Tesla T4 |
| GPU memory reported | 14.56 GiB |
| CUDA reported by PyTorch | 12.8 |
| PyTorch | 2.11.0+cu128 |
| PyTorch Geometric | 2.8.1 (installed within the notebook) |
| Primary seed | 42 |
| Additional repeated-experiment seeds | 123, 2026 |

The notebook also uses NumPy, pandas, scikit-learn, XGBoost, SciPy, statsmodels, and joblib. Exact versions of all auxiliary packages, CPU model, and host RAM were not recorded comprehensively. **Note:** the saved Kaggle metadata lists `accelerator: none`, but the recorded execution output from the environment check explicitly reports `CUDA available: True`, `GPU name: Tesla T4`, and the graph-construction stage reports `Device: cuda`. The table above describes the **recorded execution**, not the saved metadata setting.

## Selected results

### Model development

| Configuration | Validation PR-AUC |
|---|---:|
| Random Forest (all 93 local features) | 0.9894 |
| Random Forest (excluding local features 41–60) | 0.9800 |
| Random Forest (local features 41–60 only) | 0.9680 |
| MLP | 0.9491 |
| GraphSAGE with original edges | 0.8657 |
| GraphSAGE without original edges | 0.9534 |
| Adaptive graph fusion | 0.9333 |
| Residual graph fusion | 0.9416 |
| Temporal-conditioned MLP | 0.9152 |
| Separately trained zero-context MLP | 0.9296 |
| Historical graph-memory MLP (3-seed mean) | 0.9366 ± 0.0029 |
| Zero-memory MLP (3-seed mean) | 0.9451 ± 0.0033 |

These results belong to different controlled experiments; the rows should **not** be interpreted as one uniformly matched leaderboard. The graph-memory entries report three-seed descriptive means and standard deviations.

### Independent locked test (snapshots 40–49)

The locked test contains **11,184 labeled transactions**, including **636 illicit transactions**.

| Metric | Raw Random Forest | Platt-calibrated Random Forest |
|---|---:|---:|
| PR-AUC | 0.645028 | 0.645028 |
| ROC-AUC | 0.836261 | 0.836261 |
| Brier score (lower is better) | 0.035016 | 0.027316 |
| Precision at reported classification operating point | 0.768750 | 0.920765 |
| Recall at reported classification operating point | 0.580189 | 0.529874 |
| F1 | 0.661290 | 0.672655 |

The Random Forest's validation PR-AUC of **0.9894** falls to **0.6450** on the later locked test. Platt scaling improves the Brier score but does not improve the score ranking (PR-AUC/ROC-AUC).

### Frozen operational screening policies

| Policy | Manual review rate | Illicit transactions auto-cleared | Fraud auto-clear fraction | Illustrative total cost |
|---|---:|---:|---:|---:|
| Fixed thresholds, raw RF | 49.33% | 94 / 636 | 14.78% | 5,219.5 |
| Fixed thresholds, Platt RF | 16.00% | 202 / 636 | 31.76% | 5,973.5 |
| Historical order-statistic threshold, raw RF | 30.03% | 144 / 636 | 22.64% | 5,390.5 |
| Historical order-statistic threshold, Platt RF | 31.05% | 144 / 636 | 22.64% | 5,365.5 |

For the fixed policies, scores below **0.02** are auto-cleared, scores above **0.50** are auto-flagged, and the rest go to manual review. The historical order-statistic comparators use different lower thresholds selected before the locked test; their upper flag threshold remains 0.50.

The illustrative cost function is:

```text
Cost = 25 × (illicit transactions auto-cleared)
     +  1 × (licit transactions auto-flagged)
     + 0.5 × (transactions sent for manual review)
```

These coefficients are **assumed relative penalties**, not measured financial losses. Manual review outcomes are not simulated.

**Risk-control caveat:** Historical Wilson-bound and order-statistic procedures were explored during development. The implemented selection/indexing, reuse of calibration data, score ties, and temporal shift prevent interpreting their nominal bounds as valid prospective risk guarantees. In particular, the historical order-statistic policies auto-clear **22.64%** of illicit locked-test transactions despite a nominal **5%** risk objective. These are **exploratory comparators**, not certified safety methods.

## Reproducing the notebook

1. Open `Fraud_detection_kaggle_notbook.ipynb` in **Kaggle Notebooks**.
2. Attach the Elliptic++ dataset containing `txs_features.csv`, `txs_classes.csv`, and `txs_edgelist.csv`. The notebook's original absolute input path is shown above; if Kaggle mounts the dataset elsewhere, update the input path references.
3. Enable a GPU runtime to match the recorded Tesla T4 execution where available. Actual availability depends on Kaggle account and session limits.
4. Run cells **in their original order**. The notebook includes installation and verification of PyTorch Geometric; do not skip its environment/setup cells.
5. Allow the notebook to create intermediate artifacts in `/kaggle/working/`. Do not start directly from the final evaluation cells unless the required frozen models and protocol files already exist.
6. Inspect the notebook's **Step 79** (protocol freeze and SHA-256 verification) and **Step 80** (independent locked-test evaluation). Preserve the chronological splits and do not tune on test results.

Key artifact locations used by the notebook include:

```text
/kaggle/working/uc_tgnn_data/
/kaggle/working/uc_tgnn_policy/
/kaggle/working/final_paper_protocol/
```

Step 80 writes, among other outputs:

```text
/kaggle/working/final_paper_protocol/final_test_predictive_metrics.csv
/kaggle/working/final_paper_protocol/final_test_operational_metrics.csv
```

The recorded SHA-256 of `final_evaluation_protocol.json` at the original freeze was:

```text
2345c46d878b5cbaa90e961533eb6225c55b804c16ec147e441deeafd6562471
```

This checksum is **specific to the original frozen protocol file**. A newly generated protocol may have a different checksum even if the experimental procedure is equivalent; do not manually force it to match.

## Reproducibility and interpretation notes

- The original notebook is the authoritative record of exact implementation details and experimental outputs.
- Kaggle-hosted dependencies and hardware availability can change; exact bit-for-bit replication is not guaranteed without the original environment and saved artifacts.
- The locked test was reserved for final evaluation; its results must not be used for additional hyperparameter or threshold selection.
- Validation performance, probability calibration, and operational decision safety are **distinct evaluation dimensions**.
- Results on this historical benchmark do not establish performance in live financial compliance systems.
- The analysis does **not** claim distribution-free guarantees under arbitrary temporal distribution shifts.

## Citation

If you use this repository, please cite the associated manuscript when its bibliographic details are available:

> *Beyond Validation Accuracy: Temporal Generalization and Risk-Aware Decision Reliability in Bitcoin Fraud Detection.*

Publication venue, DOI, and final bibliographic metadata are intentionally omitted until verified.

## License

No license is declared in this README. Add an appropriate `LICENSE` file if you intend to grant reuse permissions for the repository code. Dataset licensing and redistribution terms must be checked separately with the original dataset provider.
