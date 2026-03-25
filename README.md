# Fraud Detection — Ravelin Data Science Code Test

A full end-to-end fraud detection analysis completed as part of a Stage 2 Data Scientist interview at Ravelin. The project covers exploratory data analysis, feature importance, rule-based fraud detection, and ML model comparison — with a focus on actionable, client-ready outputs.

---

## Project Structure

```
├── Submission_Notebook.ipynb          # Full analysis notebook
├── Hamza Ahmed - Fraud Detection Report.pdf  # Written report
├── requirements.txt                   # Dependencies
└── diagrams/                          # Supporting visuals
```

---

## Dataset

- **10,800 transactions** across 38 features
- **7.41% fraud rate** — class imbalance present
- Features include: payment method, market/country, network signals, account age, order value/volume, anonymised numeric features

---

## Analysis Overview

### 1. Exploratory Data Analysis
- Identified **Google Pay as the highest-risk payment method** — 62% fraud rate vs 4.2% for standard card payments, accounting for nearly half of all fraud in the dataset
- **High-risk markets:** MARKET_ID24 (37%), MARKET_ID14 (18%), MARKET_ID9 (18%), MARKET_ID34 (17%) — filtered to markets with ≥50 transactions for statistical reliability
- **Temporal pattern:** Fraud peaked mid-September at 17.2% before declining steadily to 0.6% by November
- **Missing data:** `skuPopularity` features missing 67% of values; `isEWallet` effectively missing 94.5% (non-ewallet transactions stored as sentinel string)

### 2. Feature Importance
Compared two approaches to surface the most informative features:

| Method | Top Features |
|---|---|
| RF Gini Importance | networkGeneralSizeFeature25, emailFeature15, accountAgeFeature12 |
| Permutation Importance | latestOrderPriceFeature15, weekOfTransaction, networkGrowthFeature15 |

Key insight: RF Gini overstates features appearing near tree roots (correlated network features). Permutation importance more honestly tests actual model reliance. Combined story: fraudsters are identifiable by **recent order behaviour** and **network signals**.

### 3. Rule-Based Fraud Detection

Four rules were built and evaluated, with results compared in a summary table:

| Rule | Precision | Recall | Block Rate |
|---|---|---|---|
| Block Google Pay | 62.1% | 46.5% | 5.55% |
| Block High-Risk Markets (>30%) | 37.1% | 2.88% | 0.57% |
| New Model Score ≥ 0.3 | 71.1% | 30.75% | 3.20% |
| Combined (Rule 1 OR Rule 3) | 58.1% | 54.0% | 6.89% |
| **Overlap (Rule 1 AND Rule 3)** | **92.5%** | 23.25% | 1.86% |

**Recommended tiered approach:**
- **Auto-block** when both rules fire (92.5% precision, minimal legitimate disruption)
- **Manual review queue** when only one rule fires

### 4. Model Comparison — Old vs New Model

| Metric | Old Model | New Model |
|---|---|---|
| ROC-AUC | 0.83 | 0.89 |
| PR-AUC | lower | higher |
| Score range | 0–0.65 | 0–1.0 |

The new model scores higher on both ROC-AUC and PR-AUC. Crucially, the higher block rate at threshold 0.3 concentrates on fraudulent transactions rather than legitimate customers — the right trade-off. PR-AUC is used as the primary metric given the class imbalance.

---

## Tech Stack

| Tool | Use |
|---|---|
| Python | Core analysis |
| pandas | Data manipulation |
| scikit-learn | RandomForestClassifier, permutation importance, ROC/PR metrics |
| XGBoost | Model evaluation |
| matplotlib / seaborn | Visualisations |
| Jupyter Notebook | Development environment |

---

## Key Takeaways

1. **A single categorical signal (Google Pay) drives nearly half of all fraud** — simple, explainable, and highly effective as a standalone rule
2. **Combining rules beats individual rules on recall (54%)** while the overlap approach achieves 92.5% precision — precision and recall serve different business priorities
3. **The new model is clearly superior** — better AUC, better score separation, and higher-quality blocks
4. **PR-AUC is the honest metric for imbalanced fraud data** — ROC-AUC flatters models on imbalanced datasets; PR-AUC reflects real-world performance

---

## Author

**Hamza Ahmed** — Data Scientist
[LinkedIn](www.linkedin.com/in/hamzaahmed0196) | [GitHub](https://github.com/Hamza-Ahmed96)
# Fraud-Detection-and-Machine-Learning-Evaluation
