# Final Bitcoin Fraud Detection — Comprehensive Reproducibility README

**Source notebook:** `Final_Bitcoin_Fraud.ipynb` (130 notebook cells, including code, markdown, and blank cells).  
**Paper:** *Beyond Validation Accuracy: Temporal Generalization and Risk-Aware Decision Reliability in Bitcoin Fraud Detection*.  
**Study:** Elliptic++ Bitcoin transaction-level illicit/licit classification, chronological generalization, calibration, graph and temporal controls, and three-action operational screening.  
**README scope:** Documents the uploaded notebook **as saved**, including original experiments, later post-hoc analyses, Step 79 artifact-hash troubleshooting, results, outputs, and limitations. **This README does not imply that all cells have been freshly rerun in this environment.** Numerical values below come from stored notebook outputs.

## 1. Quick start

1. Open `Final_Bitcoin_Fraud.ipynb` in **Kaggle Notebooks** and attach the Elliptic++ dataset used by the notebook. Inspect the input-path discovery and `DATA_DIR` assignment in the opening cells; change paths only if your attached dataset is mounted differently.
2. Enable a suitable GPU for the neural/GNN sections. The notebook checks and may install PyTorch Geometric; after environment changes a kernel restart can be necessary. Original reported runtime was Kaggle with an NVIDIA Tesla T4; recorded software versions should be treated as run-specific, not universal requirements.
3. Run cells **in displayed notebook order** (not strictly numerical Step order: some notebook cells are intentionally out of numerical sequence). The notebook is stateful; many later cells depend on variables and files created by earlier cells.
4. **Immediately before Step 79**, run the checksum-diagnostic cell (existing notebook code cell index 88), or the more complete four-artifact verification cell in Section 8 below. If the current frozen files differ from the historical hashes embedded in Step 79, investigate why; only after verifying their identity and behavior should you update the `expected_hashes` literals.
5. Run Step 79 to save and hash `final_evaluation_protocol.json`; then Step 80 checks the protocol and artifacts before locked-test inference. Run the later post-hoc Steps 81–110 for the additional experiments. Do not describe later test-informed work as independent confirmatory testing.
6. The final ZIP cells package `/kaggle/working`. **Warning:** the last copy operation places a ZIP inside `/kaggle/working`; avoid rerunning the archive process repeatedly without cleaning up the copied archive, otherwise ZIP size can grow unnecessarily.

### Input files and essential data objects

The opening notebook cells load Elliptic++ transaction features, transaction classes, and transaction-to-transaction edges, merge them on transaction identifiers, map original classes to `label` (licit `0`, illicit `1`, unknown `-1`), and establish `Time step`. The predictive features are `Local_feature_1` through `Local_feature_93`. The graph uses same-snapshot edges, with unknown-label nodes available for context but excluded from supervised metrics. Common stateful variables include `df`, `local_features`, `X_all`, `labels_all`, `times_all`, `scaler`, `rf_model`, `platt_calibrator`, `rf_final`, `platt_final`, `scaler_final`, `p_raw`, `p_cal`, `y_test`, `test_times`, `y_fit`, `p_fit_raw`, `p_fit_cal`, `y_assess`, `p_assess_raw`, and `p_assess_cal`. Variables may be overwritten by later cells: **do not execute individual late cells in a fresh kernel and assume they are standalone**.

### Chronological data split

| Partition | Time steps | Labeled samples | Illicit samples | Purpose |
|---|---:|---:|---:|---|
| Training | 1–29 | 26,381 | 2,871 | Fit predictive models |
| Validation | 30–34 | 3,513 | 591 | Development-period model comparison |
| Calibration fit | 35–37 | 3,547 | 255 | Fit Platt mapping; historical threshold selection |
| Assessment | 38–39 | 1,939 | 192 | Operational assessment and threshold exploration |
| Locked test | 40–49 | 11,184 | 636 | Original frozen-policy evaluation; reused in clearly labeled post-hoc analyses |

The dataset is described as 203,769 transaction records, with 4,545 illicit, 42,019 licit, and 157,205 unknown class labels. The training-only `StandardScaler` is reused where standardized features are required. The notebook's graph operations restrict edges to each time step, rather than using direct cross-snapshot message passing.

## 2. Research questions and workflow

- **RQ1, temporal generalization:** Conventional models (logistic regression, RF, XGBoost), MLP, GCN, GraphSAGE, fusion variants, temporal context, graph memory, and their controls, using chronological validation and later test-period checks.
- **RQ2, feature reliability:** Within-snapshot permutation, retrained exclusion, feature redundancy/reconstruction, source-to-processed integrity, and temporal KS/score-shift diagnostics.
- **RQ3, probability reliability:** Raw versus Platt-calibrated RF, Brier score, log loss, ECE, matched coverage, low-probability tails, supplementary isotonic calibration.
- **RQ4, operational decisions:** Auto-clear / manual-review / auto-flag rules, cost assumptions, Wilson and order-statistic historical thresholds, corrected CP selection, fixed-FPR recall, and randomized controls.

**Core interpretation:** Strong validation AP, aggregate calibration improvement, and low historical miss rates are distinct from future risk guarantees. All later retrospective or test-informed extensions must be reported as exploratory.

## 3. End-to-end notebook implementation (all numbered steps)

The following inventory is generated from the **actual source of the uploaded notebook**. “Cell” is its zero-based notebook index; it is **not** the same as the Step number. Notebook cells with nonsequential or repeated step labels are listed separately so that no implementation is silently omitted.

| Notebook cell | Recorded step / implementation | Reproducibility role |
|---:|---|---|
| 0 | # This Python 3 environment comes with many helpful analytics libraries installed | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 1 | import os | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 2 | from pathlib import Path | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 3 | import pandas as pd | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 4 | assert "txId" in features.columns | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 5 | print("=" * 55) | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 6 | import pandas as pd | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 7 | import numpy as np | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 8 | import numpy as np | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 9 | STEP 1: DATASET STRUCTURE AND LABEL VALIDATION") | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 10 | import numpy as np | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 11 | # Verify class-label consistency | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 12 | STEP 4: FEATURE SELECTION AND MISSING VALUE ANALYSIS") | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 13 | STEP 5: LEAKAGE-FREE FEATURE STANDARDIZATION") | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 14 | STEP 6: PREPARING CLASSIFICATION DATASETS") | Dataset loading, labels, splits, feature preparation and preprocessing checks |
| 15 | STEP 7: LOGISTIC REGRESSION BASELINE TRAINING") | Conventional model fitting, metrics, and temporal ID checks |
| 16 | STEP 8: RANDOM FOREST BASELINE TRAINING") | Conventional model fitting, metrics, and temporal ID checks |
| 17 | STEP 9: XGBOOST BASELINE TRAINING") | Conventional model fitting, metrics, and temporal ID checks |
| 18 | STEP 10: TEMPORAL FEATURE AND ID OVERLAP AUDIT") | Conventional model fitting, metrics, and temporal ID checks |
| 19 | STEP 11: GRAPH EDGE DATASET INSPECTION") | Transaction graph inspection, snapshot construction and PyG preparation |
| 20 | STEP 12: TRANSACTION GRAPH AND TEMPORAL EDGE AUDIT") | Transaction graph inspection, snapshot construction and PyG preparation |
| 21 | STEP 13: TEMPORAL GRAPH SNAPSHOT CONSTRUCTION") | Transaction graph inspection, snapshot construction and PyG preparation |
| 22 | STEP 14: TEMPORAL GRAPH CONNECTIVITY AUDIT") | Transaction graph inspection, snapshot construction and PyG preparation |
| 23 | STEP 15: PYTORCH GEOMETRIC ENVIRONMENT CHECK") | Transaction graph inspection, snapshot construction and PyG preparation |
| 24 | STEP 16: INSTALLING PYTORCH GEOMETRIC") | Transaction graph inspection, snapshot construction and PyG preparation |
| 25 | STEP 17: CONVERT SNAPSHOTS TO PYTORCH GEOMETRIC") | Transaction graph inspection, snapshot construction and PyG preparation |
| 26 | STEP 18: DEFINE GCN BASELINE ARCHITECTURE") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 27 | STEP 19: GCN BASELINE TRAINING") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 28 | STEP 20: BEST GCN CHECKPOINT EVALUATION") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 29 | STEP 21: DEFINE GRAPHSAGE BASELINE ARCHITECTURE") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 30 | STEP 22: GRAPHSAGE BASELINE TRAINING") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 31 | STEP 23: BEST GRAPHSAGE CHECKPOINT EVALUATION") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 32 | STEP 24: DEFINE NODE-ONLY MLP BASELINE") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 33 | STEP 25: NODE-ONLY MLP BASELINE TRAINING") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 34 | STEP 26: BEST MLP CHECKPOINT EVALUATION") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 35 | STEP 27: GRAPHSAGE EDGE-ABLATION DIAGNOSTIC") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 36 | STEP 28: CONTROLLED GRAPHSAGE EDGE ABLATION SETUP") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 37 | STEP 29: TRAIN CONTROLLED NO-EDGE GRAPHSAGE") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 38 | STEP 30: CONTROLLED NO-EDGE GRAPHSAGE EVALUATION") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 39 | STEP 31: DEFINE ADAPTIVE GRAPH FUSION NETWORK") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 40 | STEP 32: TRAIN ADAPTIVE GRAPH FUSION NETWORK") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 41 | STEP 33: FUSION EVALUATION AND GATE ANALYSIS") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 42 | STEP 34: DEFINE RESIDUAL GRAPH FUSION NETWORK") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 43 | STEP 35: TRAIN RESIDUAL GRAPH FUSION NETWORK") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 44 | STEP 36: RESIDUAL GRAPH FUSION EVALUATION") | Neural/GNN baselines, graph edge controls, and adaptive/residual fusion |
| 45 | STEP 37: TEMPORAL DISTRIBUTION SHIFT ANALYSIS") | Feature distribution shift, integrity, provenance and redundancy |
| 46 | STEP 38: TEMPORAL PERFORMANCE AND LABEL AUDIT") | Feature distribution shift, integrity, provenance and redundancy |
| 47 | STEP 39: CLASS-CONDITIONAL TEMPORAL DRIFT") | Feature distribution shift, integrity, provenance and redundancy |
| 48 | STEP 40: SAMPLE-SIZE-CONTROLLED DRIFT CHECK") | Feature distribution shift, integrity, provenance and redundancy |
| 49 | STEP 41: FEATURE LEAKAGE AND DATA INTEGRITY AUDIT") | Feature distribution shift, integrity, provenance and redundancy |
| 50 | STEP 42: FEATURE PROVENANCE AND REDUNDANCY AUDIT") | Feature distribution shift, integrity, provenance and redundancy |
| 51 | STEP 43: EXACT FEATURE MAPPING AND PROVENANCE CHECK") | Feature distribution shift, integrity, provenance and redundancy |
| 52 | STEP 44: BUILD PAST-ONLY TEMPORAL CONTEXT") | Past-only temporal conditioning, ablations and control comparisons |
| 53 | STEP 45: DEFINE TEMPORAL-CONDITIONED MLP") | Past-only temporal conditioning, ablations and control comparisons |
| 54 | STEP 46: TRAIN TEMPORAL-CONDITIONED MLP") | Past-only temporal conditioning, ablations and control comparisons |
| 55 | STEP 47: TEMPORAL CONTEXT SENSITIVITY ABLATION") | Past-only temporal conditioning, ablations and control comparisons |
| 56 | STEP 48: TEMPORAL MODULATION DIAGNOSTICS") | Past-only temporal conditioning, ablations and control comparisons |
| 57 | STEP 49: PARAMETER-MATCHED ZERO-CONTEXT CONTROL") | Past-only temporal conditioning, ablations and control comparisons |
| 58 | STEP 50: PER-SNAPSHOT TEMPORAL ROBUSTNESS COMPARISON") | Past-only temporal conditioning, ablations and control comparisons |
| 59 | **Calibration-set diagnostic for Random Forest** | RF preprocessing verification and Platt calibration |
| 60 | STEP 53: CORRECTED RANDOM FOREST CALIBRATION AUDIT") | RF preprocessing verification and Platt calibration |
| 61 | **Random Forest preprocessing verification** | RF preprocessing verification and Platt calibration |
| 62 | STEP 52: VERIFY RANDOM FOREST INPUT PREPROCESSING") | RF preprocessing verification and Platt calibration |
| 63 | STEP 54: CHRONOLOGICAL SIGMOID CALIBRATION") | RF preprocessing verification and Platt calibration |
| 64 | STEP 55: SELECTIVE PREDICTION AND RISK-COVERAGE AUDIT") | RF preprocessing verification and Platt calibration |
| 65 | STEP 56: OPERATIONAL FRAUD DETECTION WITH ABSTENTION") | Selective screening, illustrative costs, frozen artifacts and integrity |
| 66 | STEP 57: ASYMMETRIC OPERATIONAL COST AUDIT") | Selective screening, illustrative costs, frozen artifacts and integrity |
| 67 | STEP 58: COST-SENSITIVE THREE-ACTION DECISION AUDIT") | Selective screening, illustrative costs, frozen artifacts and integrity |
| 68 | STEP 59: FREEZE COST-SENSITIVE DECISION POLICY") | Selective screening, illustrative costs, frozen artifacts and integrity |
| 69 | STEP 60: REPRODUCIBILITY AND ARTIFACT INTEGRITY AUDIT") | Selective screening, illustrative costs, frozen artifacts and integrity |
| 70 | STEP 61: TEMPORAL GRAPH FEASIBILITY AUDIT") | Historical graph memory and three-seed controls |
| 71 | STEP 62: STRICTLY PAST-ONLY GRAPH MEMORY") | Historical graph memory and three-seed controls |
| 72 | STEP 63: HISTORICAL GRAPH-MEMORY CONTROLLED EXPERIMENT") | Historical graph memory and three-seed controls |
| 73 | STEP 64: MULTI-SEED GRAPH-MEMORY ROBUSTNESS") | Historical graph memory and three-seed controls |
| 74 | STEP 65: RF FEATURE RELIANCE AND TEMPORAL STABILITY") | Historical graph memory and three-seed controls |
| 75 | STEP 66: FEATURE RELIANCE AND WITHIN-SNAPSHOT AUDIT") | Feature-group audits, source provenance and reliability checks |
| 76 | STEP 67: RF FEATURE-GROUP SUFFICIENCY AND NECESSITY") | Feature-group audits, source provenance and reliability checks |
| 77 | STEP 68: CROSS-GROUP FEATURE REDUNDANCY AUDIT") | Feature-group audits, source provenance and reliability checks |
| 78 | STEP 69: DATASET PROVENANCE AND FEATURE DOCUMENTATION AUDIT") | Feature-group audits, source provenance and reliability checks |
| 79 | STEP 70: ORIGINAL FEATURE INDEXING AND PROVENANCE CHECK") | Feature-group audits, source provenance and reliability checks |
| 80 | STEP 71: HEADER-CORRECTED DATASET INTEGRITY AUDIT") | Feature-group audits, source provenance and reliability checks |
| 81 | STEP 72: FEATURE-FAMILY EXCLUSION SENSITIVITY") | Feature-group audits, source provenance and reliability checks |
| 82 | STEP 73: TEMPORAL DRIFT AND PREDICTIVE RELIABILITY") | Feature-group audits, source provenance and reliability checks |
| 83 | STEP 74: TEMPORAL UNCERTAINTY AND CALIBRATION DIAGNOSTIC") | Feature-group audits, source provenance and reliability checks |
| 84 | STEP 75: FROZEN POLICY FRAUD-MISS AND REVIEW AUDIT") | Feature-group audits, source provenance and reliability checks |
| 85 | STEP 76: HISTORICAL RISK-CONSTRAINED AUTO-CLEAR THRESHOLD") | Historical risk threshold selection and tie-safe sensitivity |
| 86 | STEP 77: ORDER-STATISTIC FRAUD-RISK CONTROL") | Historical risk threshold selection and tie-safe sensitivity |
| 87 | STEP 78: TIE-SAFE THRESHOLD AND TEMPORAL RISK AUDIT") | Historical risk threshold selection and tie-safe sensitivity |
| 88 | Pre-Step-79 SHA-256 diagnostic for frozen Platt calibrator | Artifact hash diagnostic, protocol freeze and locked-test evaluation |
| 89 | Markdown troubleshooting note: frozen_platt_calibrator.joblib | Artifact hash diagnostic, protocol freeze and locked-test evaluation |
| 90 | STEP 79: FINAL EXPERIMENTAL PROTOCOL FREEZE | Artifact hash diagnostic, protocol freeze and locked-test evaluation |
| 91 | STEP 80: FINAL INDEPENDENT LOCKED-TEST EVALUATION | Artifact hash diagnostic, protocol freeze and locked-test evaluation |
| 92 | STEP 81: M1 FROZEN MODEL AND ARTIFACT AVAILABILITY AUDIT | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 93 | STEP 82: M1 CHECKPOINT STRUCTURE VERIFICATION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 94 | STEP 83: M1 ORIGINAL IMPLEMENTATION AUDIT | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 95 | STEP 84: M1 DATA AND INFERENCE COMPATIBILITY AUDIT | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 96 | STEP 85: M1 FROZEN-MODEL VALIDATION REPRODUCTION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 97 | STEP 86: M1 FROZEN-MODEL CHRONOLOGICAL TEST EVALUATION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 98 | STEP 87: M1 REMAINING MODELS - CONSOLIDATED READINESS AUDIT | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 99 | STEP 88: M1 FINAL INPUT-PROTOCOL VERIFICATION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 100 | STEP 89: M1 COMBINED REMAINING-MODEL EVALUATION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 101 | STEP 90: GRAPH-MEMORY RECONSTRUCTION AUDIT | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 102 | STEP 91: GRAPH-MEMORY RECONSTRUCTION VERIFICATION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 103 | STEP 92: EXACT GRAPH-MEMORY RECONSTRUCTION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 104 | Step 94: Recovery of Original Graph-Structural Feature Construction | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 105 | STEP 94: RECOVER ORIGINAL GRAPH-STATISTICS SOURCE | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 106 | STEP 95: POST-HOC GRAPH-MEMORY LOCKED-PERIOD EVALUATION | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 107 | STEP 96: FINAL M1 PROVENANCE AUDIT | Post-hoc frozen-checkpoint evaluation and graph-memory provenance |
| 108 | Step 43 prevalence:", prev.get(43, np.nan)) | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 109 | STEP 97 — RF + XGBOOST SNAPSHOT RESULTS | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 110 | STEP 97 (second cell): matched-coverage calibration | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 111 | STEP 98 — M6: STATISTICAL UNCERTAINTY (95% CI) | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 112 | STEP 99 — Expanding-window RF (M4) | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 113 | STEP 100 — M3/M7: TEMPORAL PREVALENCE AND FEATURE DRIFT | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 114 | STEP 101 — M5/M10: RISK CONTROL + COST SENSITIVITY | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 115 | STEP 102 — M2: RISK–COVERAGE + RELIABILITY DIAGRAMS | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 116 | STEP 103 — M3: FIXED-FPR RECALL + TEMPORAL LABEL SHIFT | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 117 | STEP 104 — M2/R3: CALIBRATION-TAIL AUDIT | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 118 | STEPS 105 + 106: cost sensitivity and paired bootstrap | Post-hoc calibration, uncertainty, temporal metrics, plots and cost analyses |
| 119 | STEP 107: DATASET AND VARIABLE CHECK") | Post-hoc reviewer diagnostics and corrected risk controls |
| 120 | STEP 107 — FIND EXACT CALIBRATION / ASSESSMENT VARIABLES") | Post-hoc reviewer diagnostics and corrected risk controls |
| 121 | STEP 107 (main analysis): historical calibration-tail audit | Post-hoc reviewer diagnostics and corrected risk controls |
| 122 | STEP 108 — REVIEWER POINT 2 | Post-hoc reviewer diagnostics and corrected risk controls |
| 123 | STEP 109: class-conditional score-shift audit | Post-hoc reviewer diagnostics and corrected risk controls |
| 124 | STEP 110: matched-rule randomized control | Post-hoc reviewer diagnostics and corrected risk controls |
| 126 | Export: create complete /kaggle/working ZIP | Export / packaging |
| 128 | Export: copy ZIP into /kaggle/working | Export / packaging |

### Implementation detail by experimental phase

**Opening cells and Steps 1–14:** Inspect source CSVs and schemas, join transaction attributes to class labels, validate transaction IDs and time steps, quantify missing/nonfinite values, define the 93 local features, check label mappings, fit training-only scaling, and build chronological labeled subsets. The first Kaggle-generated import cell and unnumbered inspection cells are part of the executable setup.

**Steps 15–30:** Audit transaction edges and time-step consistency; create snapshot-specific graph structures; check PyTorch Geometric installation; construct graph `Data` objects; train/evaluate GCN, GraphSAGE, and node-only MLP using weighted loss and early stopping; compare inference-time edge removal with a separately trained no-edge GraphSAGE. These two edge-ablation procedures are **not** equivalent.

**Steps 31–36:** Define, train and evaluate adaptive graph fusion and residual graph fusion. Inspect learned gates and note that large gates do not by themselves establish that graph information improves prediction. Saved best-model checkpoints are later reused for post-hoc comparisons.

**Steps 37–43:** Evaluate temporal distributional variation, per-step model performance and labels, class-conditional drift, sample-size controls, data integrity, original feature provenance and reconstruction-based redundancy. Index-defined features 41–60 are analyzed by permutation and retraining; feature indices are not verified semantic names.

**Steps 44–50:** Construct strictly past-only snapshot summary context, train a temporal-conditioned MLP, perform context/lag sensitivity tests and modulation diagnostics, train a zero-context control, and report snapshot-level comparisons. Historical conditioning does not make a model prospectively validated.

**Steps 52–60:** Check the RF preprocessing path, fit chronological Platt logistic calibration on steps 35–37 using clipped RF log-odds, measure calibration, evaluate risk–coverage and three-action decisions, audit illustrative costs, save RF/calibrator/policy artifacts, and verify that a reloaded scaler reproduces the earlier transformed data. **Step 53 appears before Step 52 in the notebook's displayed order**; follow cell order. Step 59 derives cost-based action regions from the three assumed losses, whereas the fixed screening policy elsewhere uses explicit `tau_L=0.02` and `tau_H=0.50`.

**Steps 61–64:** Inspect temporal graph feasibility; construct historical graph-memory statistics using strictly prior snapshots; compare graph-memory MLP against a zero-memory control; repeat selected comparisons with seeds 42, 123, 2026. Multi-seed validation summaries and later single-checkpoint test values must not be conflated.

**Steps 65–74:** Analyze RF feature reliance and temporal stability, perform within-snapshot permutation, test feature-group sufficiency/necessity, reconstruct feature groups, audit source dataset provenance and corrected headers, assess temporal drift and calibration uncertainty. Distinguish a drop caused by perturbing a trained model from retraining a new model without those features.

**Steps 75–78:** Report fraud auto-clearances, review workload and risk-constrained threshold searches. Step 77 uses an original order-statistic-inspired selection with a known indexing inconsistency. Step 78 examines tie-safe behavior. These analyses are historical and exploratory; their nominal CP/Wilson bounds are **not** validated post-selection or future-temporal guarantees.

**Pre-Step-79 diagnostic and Steps 79–80:** Verify frozen files, save a protocol manifest and its SHA-256 digest, and perform original frozen locked-test inference. Step 79 stores the original fixed and historical policies; Step 80 reloads RF, Platt and scaler, verifies protocol and artifact hashes, evaluates pooled and snapshot-level classification and operational metrics, and exports CSVs. The post-hoc corrected historical thresholds introduced later **do not retroactively replace the original frozen protocol**.

**Steps 81–96:** Read-only inventory of frozen model checkpoints, model-class/shape compatibility checks, reconstruction of exact input and historical graph-memory features, reproduction of validation behavior, chronological test evaluation of frozen models, and provenance audits. Where original checkpoint metadata is absent or validation re-evaluation differs from earlier summaries, the notebook documents the discrepancy rather than assuming a cause. These are post-hoc evaluations, not new untouched tests.

**Steps 97–103:** RF/XGBoost per-snapshot evaluation; matched-coverage calibration; two paired bootstrap procedures (class-stratified transaction and temporal snapshot-block resampling); retrospective expanding-window RF; prevalence/feature-shift analysis; risk/cost sensitivity; calibration and risk–coverage figures; and fixed-FPR recall plus temporal prevalence plots. **There are two cells labeled Step 97.** Step 103 chooses a threshold via `roc_curve(..., drop_intermediate=False)` from assessment scores and applies `score >= threshold`; see Section 7 for the separate Step 109 convention.

**Steps 104–106:** Low-risk calibration-tail audit, pooled and per-snapshot risk plots, illustrative cost-derived threshold sweeps, and RF-minus-XGBoost paired uncertainty. The calibration-tail audit is separate from aggregate Brier/ECE, and cost weights are illustrative, not measured business costs.

**Steps 107–108:** Inspect available variables, audit historical calibration-tail underprediction across calibration/assessment/test and supplementary isotonic calibration, then compute a corrected historical order-statistic/CP threshold under strict clearance. Historical thresholds are selected on steps 35–37; later periods are evaluation only. The correction changes the miss count from the original historical policy but does not yield a prospective guarantee.

**Step 109:** Frozen RF class-conditional score quantiles (10th, median, 90th) for assessment 38–39, early test 40–42 and late test 43–49; a separately selected assessment 5%-FPR operating point; AP, ROC-AUC, per-snapshot recall/FPR; and fixed-0.02 auto-clearance rates relative to prevalence. Results are descriptive and do not prove the causal source of drift.

**Step 110:** Apply the same corrected nominal CP threshold rule to historical calibration and a 300-repetition randomized test-period illicit-score split (255 calibration / 381 evaluation, seed 2026). Report the empirical held-out miss rate, fraction meeting the empirical 5% target, and fraction whose **evaluation-set** one-sided CP upper bound meets 5%. The within-period control reuses inspected test scores, removes temporal order, and is not prospective validation.

**Final export cells:** Package the contents of `/kaggle/working` as `/kaggle/working_complete_final.zip`, then copy it into `/kaggle/working/working_complete_final.zip`. These are archive operations, not new experiments.

## 4. Verified numerical outputs preserved in the notebook

### Predictive discrimination

| Comparison | Result |
|---|---|
| RF validation AP | 0.9894 |
| RF locked-test AP / ROC-AUC | 0.64503 / 0.83626 |
| RF expanding-window retrospective test AP / ROC-AUC | 0.7612 / 0.9489 |
| RF vs XGBoost pooled test AP difference (RF − XGBoost) | +0.002503; both paired bootstrap 95% intervals include zero |
| RF vs XGBoost pooled test ROC-AUC difference | −0.019292; both paired intervals exclude zero |
| RF validation ROC-AUC | approximately 0.9968 |

### Calibration and original frozen screening

| Item | Stored result |
|---|---|
| Test RF Brier, raw → Platt | 0.03502 → 0.02732 |
| Raw fixed `p < 0.02` illicit cleared | 94/636 = 14.78%; 5,187 transactions cleared |
| Platt fixed `p < 0.02` illicit cleared | 202/636 = 31.76%; 9,029 transactions cleared |
| Platt matched 80.73% clearance coverage, test low-risk tail | mean predicted illicit 0.27%, observed illicit 2.24% |
| Original historical order-statistic, test | 144/636 = 22.64%; 7,345 cleared |
| Corrected historical threshold, test | 134/636 = 21.07%; 6,991 cleared |
| Original randomized within-period control | mean empirical held-out illicit miss ~3.05% (318/318 split) |

### Step 107 historical calibration-tail extension

- Platt low-risk tail on **assessment 38–39** at matched coverage: approximately **0.18% predicted versus 0.78% observed** illicit proportion.
- Platt low-risk tail on **test 40–49** at matched coverage: approximately **0.27% predicted versus 2.24% observed**.
- Supplementary isotonic mapping was fitted on **steps 35–37** and inspected on later periods; it did not eliminate test-tail underprediction. Its calibration-fit metrics are in-sample for the isotonic mapping.
- Test log loss (raw / Platt / isotonic): approximately **0.15955 / 0.16068 / 0.26097**. Improved Brier score must not be interpreted as uniformly improved calibration under all metrics.

### Step 108 corrected historical threshold

- Historical illicit calibration sample: **255**, strict-clearance allowed misses **6**, nominal one-sided CP upper bound **0.045910**.
- Corrected **raw RF** clearance threshold: **0.04176944361297847**; corrected **Platt** threshold approximately **0.003745**.
- Assessment 38–39: **6/192 = 3.125%** illicit cleared.
- Test 40–49: **134/636 = 21.069%** illicit cleared; **6,991/11,184 = 62.51%** of all labeled test transactions cleared.
- The test miss-fraction two-sided 95% CP interval reported in Step 108 is approximately **17.96%–24.45%**; Step 110 separately reports a **one-sided** upper bound of **23.91%**. Do not confuse these distinct intervals.

### Step 109 class-conditional score shift

| Period | Illicit n | Illicit score median | Illicit recall at Step-109 threshold | Licit FPR |
|---|---:|---:|---:|---:|
| Assessment 38–39 | 192 | 0.973641 | 93.75% | 4.98% |
| Early test 40–42 | 467 | 0.970402 | 82.66% | 5.01% |
| Late test 43–49 | 169 | 0.023649 | 2.96% | 5.23% |
| Full test 40–49 | 636 | 0.756712 | 61.48% | 5.15% |

- Assessment-selected Step-109 threshold: **0.27624742689746656**; predictions use `score > threshold`.
- Illicit score 10th percentiles (assessment / early / late): **0.583423 / 0.076379 / 0.003230**.
- Licit median scores (assessment / early / late): **0.008929 / 0.023114 / 0.020700**.
- At raw RF fixed `p < 0.02`, late-test cleared-pool fraud rate was **2.37%**, near late-test illicit prevalence **2.53%**. This is a descriptive operational warning, not a causal finding.

### Step 110 matched-rule randomized sensitivity

| Metric | Stored output |
|---|---:|
| Historical calibration | 255 illicit scores; 6 misses |
| Historical test | 134/636 = 21.07% missed |
| Randomized splits | 300 repetitions; 255 calibration / 381 evaluation |
| Mean / median held-out miss rate | 2.6395% / 2.6247% |
| Held-out miss-rate 2.5th / 97.5th percentiles | 0.5249% / 5.1247% |
| Fraction with empirical miss rate ≤5% | 97.3333% |
| Fraction with evaluation one-sided CP upper ≤5% | 65.3333% |

These percentages are descriptive of repeated splits of **the previously examined test pool**. The empirical success frequency is not a formal coverage guarantee, and matching the calibration sample size does not make chronological and randomized samples exchangeable.

## 5. Outputs and files

All paths below are relative to Kaggle's writable `/kaggle/working/`. Files are produced when their corresponding cells execute successfully; some files may be absent if only part of the notebook was rerun.

| Folder / file | Contents and producer |
|---|---|
| `uc_tgnn_policy/` | Frozen RF, Platt calibrator, scaler, policy config, and related metadata (Steps 59–60) |
| `final_paper_protocol/final_evaluation_protocol.json` | Original Step-79 protocol, original policy thresholds, artifact SHA-256 mapping |
| `final_paper_protocol/final_evaluation_protocol.sha256` | Hash of the saved protocol JSON |
| `final_paper_protocol/final_test_predictive_metrics.csv` | Step-80 pooled and per-snapshot predictive metrics |
| `final_paper_protocol/final_test_operational_metrics.csv` | Step-80 four-policy operational metrics |
| `m1_posthoc_extension/` | Frozen-model and graph-memory post-hoc audit outputs |
| `step103_m3/` | Fixed-FPR pooled/snapshot CSVs, temporal-label-shift CSV and figures |
| `step104_calibration_tail/` | Low-risk tail CSVs and PDF/PNG plots |
| `step105_106_r3_r6/` | Cost-dependent thresholds and paired-bootstrap CSVs |
| `step107_calibration_comparison/` | Calibration-tail audit, period metrics, probability bins, comparison figure |
| `step108_corrected_cp/` | Corrected historical CP metrics, original-vs-corrected comparison, protocol CSV |
| `step109_score_quantiles.csv` | Class-conditional quantiles |
| `step109_temporal_performance.csv` | Period-level Step-109 metrics |
| `step109_snapshot_metrics.csv` | Per-test-step recall/FPR at Step-109 threshold |
| `step110_historical.csv` | Corrected historical threshold results |
| `step110_randomized_repetitions.csv` | All 300 matched-rule randomized repetitions |
| `step110_summary.csv` | Aggregated randomized diagnostic |
| `working_complete_final.zip` | Copy of exported `/kaggle/working` archive |

The notebook also creates model `.pt` checkpoints for GCN, GraphSAGE, no-edge GraphSAGE, MLP, adaptive/residual fusion, temporal context and controls, and graph-memory models and controls. Their exact presence depends on whether training cells were executed and whether files were retained in the Kaggle working directory.

## 6. Step 103 vs Step 109: why fixed-FPR values differ

**Do not overwrite Step 103's Figure 4 data with Step 109 data.** The two implementations select/interpret their thresholds differently:

- **Step 103:** computes a full assessment ROC curve with `roc_curve(..., drop_intermediate=False)`, chooses the feasible assessment point with highest TPR subject to `FPR <= 0.05`, and flags transactions using `score >= selected_threshold`. Its stored figure analysis reports assessment recall **93.75%**, assessment FPR **4.58%**, full-test recall **61.32%**, and full-test FPR **4.88%**.
- **Step 109:** derives a permissive cutoff from the assessment licit-score distribution and flags using `score > selected_threshold` (threshold **0.27624742689746656**). Its output reports assessment recall **93.75%**, assessment FPR **4.98%**, full-test recall **61.48%**, and full-test FPR **5.15%**.

Both results can be reported **only when their different selection/boundary conventions are disclosed**. This code-level difference is confirmed, but its precise numerical contribution has not been isolated by a side-by-side rerun of both selected thresholds and confusion counts. Step 103 remains the original figure analysis; Step 109 is a separate post-hoc score-shift diagnostic.

## 7. Three-action policy, metrics, and statistical safeguards

- Action convention: **clear** if `score < tau_L`; **manual review** for scores between the thresholds; **flag** if `score > tau_H`. The primary fixed policy uses `tau_L=0.02`, `tau_H=0.50` for raw and Platt RF. Boundary ties are handled by the implemented inequalities, not by assumed continuity.
- Operational illicit miss fraction: `illicit_auto_cleared / all_labeled_illicit` (not the fraud rate within cleared transactions). Cleared-pool fraud rate instead equals `illicit_auto_cleared / all_cleared`.
- Workload: `manual_review_count / all_labeled_transactions`. Illustrative cost: `25 * illicit_cleared + 1 * licit_flagged + 0.5 * manual_review_count`. These are hypothetical penalties, **not** real monetary losses.
- AP is sensitive to prevalence, while ROC-AUC provides a different discrimination view; neither alone guarantees low auto-clearance risk. Brier, ECE and log loss measure different aspects of probability reliability.
- Paired bootstrap comparisons are exploratory; transaction-level resampling assumes stronger independence than may hold, and temporal-block resampling uses only ten test steps.
- Historical Wilson/CP calculations are not valid simultaneous post-selection guarantees. Temporal exchangeability, label delay, prospective feature availability, and independent future deployment validation remain unverified.
- Post-hoc analyses reuse the inspected test period and cannot be presented as an independent confirmatory replication. Do not claim that randomized within-period controls prove the cause of temporal deterioration.

## 8. CRITICAL: Step 79 checksum mismatch and the preceding cell

### 8.1 What the uploaded notebook actually does

- **Notebook code cell index 88** (immediately after Step 78 and before the markdown note / Step 79) reads `/kaggle/working/uc_tgnn_policy/frozen_platt_calibrator.joblib`, calculates its SHA-256, compares it to a **separate older literal** `f5330025a15e9f609a71796e72091ccfad3205bccaa31d01ec03d6ee2ebd7559`, and tries to load/inspect the fitted calibrator. It does **not** itself modify Step 79's dictionary.
- **Notebook code cell index 90, Step 79**, computes hashes of four current files and checks them against its own `expected_hashes` dictionary. The Step-79 stored Platt hash is **`bda0b6cd2cc985c0f11a7c83acd77fd1fb242ac91655638b598bc3325e189a3c`**, which is **not the same** as the diagnostic cell's older `f533...` value. Thus the diagnostic cell may report `CHECKSUM MISMATCH` even if Step 79 passes; the two cells are comparing against different historical manifests.
- Step 79's stored notebook output says **“Frozen artifact checksums verified.”** This proves that the saved run passed at that time; it does **not** establish that a fresh Kaggle session has the same file bytes.
- **Step 80** then verifies the saved protocol JSON digest and each artifact digest against the Step-79 manifest. If you alter the frozen artifacts after Step 79, Step 80 should fail. Do **not** bypass these checks or edit a protocol merely to make a mismatch disappear.

### 8.2 Recommended NEW code cell immediately before Step 79

Insert or run the following **after Step 78 and after all frozen artifacts have been saved**, but **before Step 79**. It prints current SHA-256 values in the exact dictionary format needed by Step 79, and verifies that files can be loaded. It deliberately **does not rewrite artifacts or automatically approve new hashes**.

```python
from pathlib import Path
import hashlib
import json
import joblib

FROZEN_DIR = Path('/kaggle/working/uc_tgnn_policy')
FILES = [
    'frozen_policy_config.json',
    'frozen_rf_model.joblib',
    'frozen_platt_calibrator.joblib',
    'frozen_train_scaler.joblib',
]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

current_hashes = {}
for name in FILES:
    path = FROZEN_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen artifact: {path}')
    current_hashes[name] = sha256_file(path)
    print(f'{name}: {path.stat().st_size:,} bytes; SHA256={current_hashes[name]}')

# Load and inspect without changing any artifact.
with (FROZEN_DIR / 'frozen_policy_config.json').open() as handle:
    frozen_config = json.load(handle)
rf_check = joblib.load(FROZEN_DIR / 'frozen_rf_model.joblib')
platt_check = joblib.load(FROZEN_DIR / 'frozen_platt_calibrator.joblib')
scaler_check = joblib.load(FROZEN_DIR / 'frozen_train_scaler.joblib')
assert hasattr(rf_check, 'predict_proba')
assert hasattr(platt_check, 'predict_proba')
assert hasattr(scaler_check, 'transform')

print('
COPY THE FOLLOWING INTO STEP 79 *ONLY AFTER* VERIFYING ARTIFACT IDENTITY:')
print('expected_hashes = ' + json.dumps(current_hashes, indent=4))
print('
READ-ONLY AUDIT COMPLETE. No frozen files were modified.')
```

### 8.3 Exactly what to replace in Step 79

In the Step-79 cell, locate:

```python
# Verify against previously recorded hashes.
expected_hashes = {
    'frozen_policy_config.json': '...',
    'frozen_rf_model.joblib': '...',
    'frozen_platt_calibrator.joblib': '...',
    'frozen_train_scaler.joblib': '...'
}
```

**Procedure:**

1. Compare the four printed `current_hashes` values with the **existing** `expected_hashes` values. If they match, change **nothing**; rerun Step 79.
2. If **only the Platt file** differs, first compare the loaded Platt coefficients, intercept, classes, feature preprocessing and predictions against the originally verified calibration run. Serialization changes can change file bytes, but this possibility must be **verified**, not assumed. The notebook's cell index 88 can help inspect the Platt file.
3. If the model or its predictions are different unexpectedly, **restore the intended original frozen file** or investigate why it changed. Do not replace hashes to conceal a changed model.
4. **Only after verifying the current artifacts are the intended, unchanged scientific objects**, copy the new four-entry `expected_hashes = {...}` dictionary from the diagnostic output into the Step-79 cell, replacing its old literal dictionary. It is acceptable to update just the one verified changed entry; preserve all matching entries.
5. Run Step 79 again. It should create a new `final_evaluation_protocol.json` and matching `.sha256` based on the verified current artifacts. Then run Step 80, which independently checks that the protocol and artifacts still agree.
6. Record **why** the historical hash was updated, the date/session, and whether model predictions remained identical. A new digest is **not** proof that an altered model is equivalent to the original one.

**Important:** A SHA-256 digest is a fingerprint of file **bytes**, not a guarantee of statistical correctness. Never claim an artifact was frozen before the test if it was regenerated after examining test results. The Step-79 protocol is a record of the original evaluation procedure; a fresh post-hoc reconstruction must be described accurately.

### 8.4 Historical expected hashes stored in Step 79

```text
frozen_policy_config.json:
36cd1790a87637fa6f930f41dea76119502700dbcde7c39de7e6a94172482ed5

frozen_rf_model.joblib:
e3b9a540ec0d4606deb34681407c4872c0d44ef6877a2ac252ef37d3408bd34d

frozen_platt_calibrator.joblib:
bda0b6cd2cc985c0f11a7c83acd77fd1fb242ac91655638b598bc3325e189a3c

frozen_train_scaler.joblib:
f9a2293bcb151a0a2643a279a3c9b8324e9b22e44ceea33a2a5ce3f50142f47e
```

### 8.5 Other common failure modes

- `FileNotFoundError`: check that Steps 59–60 created the files in `/kaggle/working/uc_tgnn_policy/`; don't substitute a similarly named file without checking it.
- `AssertionError: Artifact checksum mismatch`: file bytes differ from the stored Step-79 manifest. Diagnose the artifact; do not disable the assertion.
- Step 80 protocol SHA mismatch: the protocol JSON changed without regenerating its `.sha256`, or a stale protocol file was copied into the session. Rebuild the protocol **from verified artifacts** through Step 79, not by editing its hash alone.
- Different `joblib`/`scikit-learn` versions: loading or reserializing may fail or yield different bytes. Record environment versions; avoid silent conversion.
- Missing stateful variables: restart and execute preceding cells in notebook order, rather than manufacturing missing arrays or fitting on test labels.
- Missing PyG: run the environment-check/install cells in a compatible Kaggle runtime and restart if necessary.
- Checkpoint re-evaluation mismatch: inspect feature order, scaler, graph edge construction, model class, and saved state; do not silently substitute a newly trained model.

## 9. Reproduction checklist

- [ ] Correct Elliptic++ transaction, class and edge CSVs attached.
- [ ] Class mapping, labeled counts and 93-feature ordering match the recorded protocol.
- [ ] Chronological split remains 1–29 / 30–34 / 35–37 / 38–39 / 40–49.
- [ ] Scaler fitted on training period only and later reused without refitting.
- [ ] Snapshot graphs contain only within-snapshot edges; unknown labels excluded from supervised metrics.
- [ ] RF/Platt files load and reproduce expected validation/assessment behavior.
- [ ] Step-79 hashes match the **intended** artifacts; reasons for any digest changes documented.
- [ ] Step 80 independently verifies saved protocol digest and artifact hashes.
- [ ] Original four frozen screening policies remain distinct from corrected post-hoc policies.
- [ ] Figure 4 / Step 103 uses its original ROC-curve `>=` threshold convention.
- [ ] Step 109 is identified as a separate exploratory `>` threshold and score-shift diagnostic.
- [ ] Step 107/108/110 use appropriate one-sided vs two-sided CP interpretations.
- [ ] No randomized positive control is presented as prospective temporal validation.
- [ ] Saved CSVs, plots, model checkpoints, protocol files, and ZIP are included where available.
- [ ] Manuscript Abstract, Methodology, Results, Discussion and Conclusion use consistent values and explicitly label post-hoc analyses.

## 10. Scope and known limitations

The notebook is a **research notebook**, not a packaged inference service. Its cells rely on shared kernel state, Kaggle-specific directories, attached data and checkpoint availability. Some experiments were added after the original locked-test results had been inspected. Feature availability at the moment of a real-world decision is not independently established, retrospective labels may be delayed operationally, and no external blockchain deployment dataset or prospective risk guarantee is supplied. Neural checkpoint provenance may be incomplete; the original development-stage validation summary can differ slightly from reloaded-checkpoint metrics. All claims should be tied to the corresponding saved outputs and evaluated under these limitations.

**README verification basis:** source code, cell ordering, and stored text outputs of the user-supplied `Final_Bitcoin_Fraud.ipynb`. No unseen Kaggle filesystem, new model training, or external result was assumed.
