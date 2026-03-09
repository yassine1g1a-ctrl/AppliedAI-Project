# ML Benchmark: Supervised & Unsupervised Learning on 22 Datasets

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)
![License](https://img.shields.io/badge/license-MIT-green)

> A comprehensive machine learning benchmark comparing **6 models** (3 supervised, 3 unsupervised) across **22 real-world datasets**, with rigorous cross-validation, hyperparameter tuning, and comparison against published baselines.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)

---

## Overview

This project systematically evaluates and compares classic machine learning algorithms — both supervised and unsupervised — across a diverse collection of UCI datasets. It includes two **handmade implementations** (MLP and SOM) built from scratch using only NumPy, alongside scikit-learn models.

Results are benchmarked against the reference paper:
> *"Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?"* — Fernández-Delgado et al.

---

## Models

### Supervised
| Model | Library | Tuned Hyperparameters |
|---|---|---|
| Logistic Regression | scikit-learn | `C` ∈ {0.01, 0.1, 1, 10} |
| Random Forest | scikit-learn | `n_estimators`, `max_depth` |
| MLP (from scratch) | NumPy only | `hidden_size`, `learning_rate` |

### Unsupervised
| Model | Library | Tuned Hyperparameters |
|---|---|---|
| K-Means | scikit-learn | `n_clusters` ∈ {2, 3, 4} |
| Gaussian Mixture Model | scikit-learn | `n_components` ∈ {2, 3, 4, 5} |
| SOM (from scratch) | NumPy only | `grid_size`, `learning_rate` |

---

## Datasets

22 datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/):

| Dataset | Samples | Features | Classes |
|---|---|---|---|
| Iris | 150 | 4 | 3 |
| Wine | 178 | 13 | 3 |
| Waveform | 5,000 | 21 | 3 |
| Car Evaluation | 1,728 | 6 | 4 |
| Titanic | 2,201 | 3–4 | 2 |
| Ionosphere | 351 | 34 | 2 |
| Dermatology | 366 | 34 | 6 |
| Ecoli | 336 | 7 | 8 |
| Zoo | 101 | 16 | 7 |
| Page Blocks | 5,473 | 10 | 5 |
| Spambase | 4,601 | 57 | 2 |
| Acute Inflammation | 120 | 6 | 2 |
| Vertebral Column (2 & 3 classes) | 310 | 6 | 2/3 |
| Tic-Tac-Toe | 958 | 9 | 2 |
| Synthetic Control | 600 | 60 | 6 |
| Statlog Vehicle | 846 | 18 | 4 |
| Statlog Heart | 270 | 13 | 2 |
| Seeds | 210 | 7 | 3 |
| Balance Scale | 625 | 4 | 3 |
| Acute Nephritis | 120 | 6 | 2 |
| Heart (Hungarian) | 294 | 13 | 2 |

---

## Methodology

### Preprocessing (leak-free pipeline)
- Missing/infinite values → imputed with **column mean** (fit on train only)
- Categorical features → **LabelEncoder**
- Feature scaling → **StandardScaler** (fit on train, applied to val/test)
- Constant columns and ID-like columns are automatically removed

### Validation Strategy
- **80/20 train-test split** (stratified)
- **5-fold Stratified Cross-Validation** for hyperparameter tuning
- Fallback strategy for rare classes (< 5 samples per class)
- All preprocessing is re-fit inside each CV fold to prevent data leakage

### Metrics
- **Supervised**: Accuracy, F1-macro, Cohen's Kappa
- **Unsupervised**: Accuracy (via majority-vote cluster mapping), F1-macro, Cohen's Kappa, Silhouette Score, Davies-Bouldin Index
- For imbalanced datasets (ratio > 2): F1-macro is used as the selection criterion

---

## Results

### Average performance across all 22 datasets

| Model | Accuracy | F1-macro | Cohen's κ |
|---|---|---|---|
| **Logistic Regression** | **90.7%** | **87.6%** | **81.8%** |
| **Random Forest** | **90.4%** | **87.9%** | **81.7%** |
| MLP (from scratch) | 87.0% | 80.0% | 76.4% |
| GMM | 78.2% | 65.4% | 56.8% |
| K-Means | 75.4% | 62.7% | 53.6% |
| SOM (from scratch) | 71.8% | 59.1% | 46.4% |

### Comparison with published baseline (Fernández-Delgado et al.)

| Model | Paper Accuracy | Our Accuracy | Paper κ | Our κ |
|---|---|---|---|---|
| Logistic Regression | 76.5% | **90.7%** | 53.7% | **81.8%** |
| Random Forest | 80.0% | **90.4%** | 57.4% | **81.7%** |
| MLP | 80.3% | **87.0%** | 62.0% | **76.4%** |

> Note: The paper averages over 121 datasets. Our subset of 22 was curated, which partially explains the performance gap.

---

## Project Structure

```
.
├── data/                          # Dataset folders (one per dataset)
│   ├── iris/
│   │   └── iris.arff
│   ├── wine/
│   │   └── wine.data
│   └── ...
├── mini_project_notebook.ipynb    # Main notebook
├── final_results_with_tuning.csv  # Full results table
├── best_hyperparameters.csv       # Best hyperparams per dataset/model
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ml-benchmark
cd ml-benchmark
pip install numpy pandas scikit-learn scipy
```

> Python 3.10+ recommended. No deep learning framework required — the MLP and SOM are implemented from scratch.

---

## Usage

Open and run the notebook:

```bash
jupyter notebook mini_project_notebook.ipynb
```

The notebook is self-contained and will:
1. Load all 22 datasets
2. Run cross-validated hyperparameter tuning for all 6 models
3. Evaluate on held-out test sets
4. Save results to CSV

---

## Key Findings

- **Random Forest and Logistic Regression** are the strongest overall performers, exceeding published baselines by ~10 percentage points in accuracy.
- **MLP from scratch** is competitive with scikit-learn baselines despite being a pure NumPy implementation.
- **Clustering models** (KMeans, GMM) perform well only on datasets with clear geometric structure (Wine, Iris, Acute datasets). They fail on datasets with overlapping classes.
- **SOM** is the weakest model overall, especially on high-dimensional or imbalanced datasets (negative silhouette scores on many datasets).
- Dataset characteristics — number of classes, dimensionality, class balance — are the dominant factors in model performance differences.

---

## Author

**Yassine TAHARASTE**  
INSA Lyon Student

---

## License

MIT
