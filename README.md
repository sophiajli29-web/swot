---
language:
- en
license: mit
tags:
- text-classification
- toxicity-detection
- algorithmic-fairness
- calibration
- jigsaw-civil-comments
- interpretability
datasets:
- google/jigsaw_toxicity_pred
- civil_comments
metrics:
- pr_auc
- roc_auc
- f1
- ece
---

# Subgroup Error & Calibration Audit of Toxicity Classifiers

## Overview
This repository contains the models, evaluation scripts, and subgroup fairness artifacts for our CS 74 project auditing identity-subgroup disparities in automated toxicity detection. Rather than focusing solely on aggregate accuracy, we evaluate false positive rates (FPR), false negative rates (FNR), probability calibration (Expected Calibration Error, ECE), and counterfactual identity-term sensitivity across demographic and identity-related comment groups.

## Models & Feature Representations
We compare three lightweight classification pipelines trained on deduplicated Civil Comments / Jigsaw data (~8% positive toxicity prevalence):

1. **TF-IDF + Logistic Regression (`tfidf_lr`)**: Word/character n-gram TF-IDF features paired with L2-regularized Logistic Regression.
2. **TF-IDF + LinearSVC + Platt Scaling (`tfidf_svm_calibrated`)**: Linear Support Vector Classifier calibrated via sigmoid (Platt) scaling on a dedicated held-out calibration split (`platt_set`) to produce probability estimates.
3. **Frozen Sentence Embeddings + Logistic Regression (`minilm_lr`)**: L2-normalized sentence embeddings extracted from `sentence-transformers/all-MiniLM-L6-v2` (`max_seq_length=128`) paired with Logistic Regression.

## Data Splits & Leakage Prevention
All text splits are strictly deduplicated and checked for zero exact-text overlap across partitions:
- **`train_sample_300k`**: 300,000 training comments used to fit feature vectorizers and base classifiers.
- **`platt_set` (50k)**: 50,000 held-out comments used exclusively to fit Platt probability calibration.
- **`thresh_set` (50k)**: 50,000 held-out comments used exclusively to select $F_1$-optimal decision thresholds.
- **`val`**: Validation split used for hyperparameter selection (regularization strength $C$ selected via validation PR-AUC).
- **`test`**: Single-pass held-out evaluation split for final performance, bootstrap confidence intervals, and subgroup calibration metrics.

## Key Audit Components
- **Aggregate & Subgroup Error Rates**: Overall PR-AUC, ROC-AUC, $F_1$, and per-group FPR/FNR with bootstrap confidence intervals across identity subgroups (e.g., `black`, `white`, `muslim`, `christian`, `jewish`, `lgbtq`, `female`, `male`, `immigration`).
- **Probability Calibration**: Subgroup Expected Calibration Error (ECE) and reliability curves (`reliability_by_group.png`, `calibration_by_group.csv`) assessing whether predicted toxicity probabilities are systematically over- or under-confident for specific groups.
- **Counterfactual Identity-Swap Analysis**: Paired evaluation (`swap_test.csv`) measuring prediction shifts when identity terms are swapped in non-toxic templates, alongside linear feature-weight inspection (`identity_weights_by_group.csv`).
- **Manual Error & Term-Matching Audit**: Human verification (`manual_check.csv`, `term_precision.csv`, `error_analysis.csv`) quantifying regex/term-matching precision and categorizing false positives and false negatives (e.g., counterspeech/quoting, neutral identity mentions, sarcasm, and borderline annotations).

## Repository Structure
```text
├── README.md
├── notebooks/                     # End-to-end training, calibration, and audit notebook
├── results/
│   ├── calibration_by_group.csv   # Subgroup ECE and calibration bin statistics
│   ├── reliability_by_group.png   # Reliability diagrams across identity groups
│   ├── fpr_ratio_by_threshold.csv # Subgroup FPR disparities across decision thresholds
│   ├── recall_by_threshold.csv    # Subgroup recall / FNR sensitivity across thresholds
│   ├── swap_test.csv              # Counterfactual identity-swap score shifts
│   ├── identity_weights_by_group.csv
│   ├── term_precision.csv         # Precision of identity regex/phrase matching
│   └── error_analysis.csv         # Categorized FP/FN qualitative audit
└── metadata/
    ├── train_sample_300k_metadata.json
    ├── thresh_50k_metadata.json
    └── val_metadata.json
```

## Quickstart & Reproduction
1. **Install dependencies:**
   ```bash
   pip install numpy pandas scikit-learn sentence-transformers matplotlib scipy
   ```
2. **Verify dataset splits & metadata:** Ensure your local or Drive cache directory matches the row counts recorded in `metadata/*_metadata.json` (delete stale `.npy` embedding caches if split deduplication parameters are modified).
3. **Run the audit pipeline:** Execute the notebook/scripts in order using fixed random seeds to reproduce feature extraction, hyperparameter tuning, Platt calibration, threshold selection, and test-set evaluation.

## Limitations & Ethical Considerations
- **Lexical Matching Noise**: Keyword and phrase rules used to identify demographic mentions can still capture non-identity contexts or miss implicit/coded references; term precision is explicitly documented in `term_precision.csv`.
- **Label Subjectivity**: Crowd-annotated toxicity labels reflect annotator norms and disagreement, particularly on reclaimed language, counterspeech, and political discourse.
- **Intended Use**: These models and audit artifacts are released strictly for **research, educational benchmarking, and fairness analysis**. They are not intended for autonomous production content moderation without human oversight.

## Citation & Course Context
Developed as part of **CS 74 (Machine Learning and Statistical Data Analysis)** at Dartmouth College. If you reference this repository, please also cite the underlying Jigsaw / Civil Comments dataset (Borkan et al., 2019).