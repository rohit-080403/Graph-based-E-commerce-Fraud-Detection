# Model Comparison: XGBoost Baseline vs. Graph-Based GCN

| Metric | XGBoost (row-based) | GCN (graph-based) |
|---|---|---|
| Precision | 0.2894 | 0.2368 |
| Recall | 0.5420 | 0.5063 |
| F1 | 0.3773 | 0.3227 |
| ROC-AUC | 0.8576 | 0.7730 |
| Fraud-Ring Recall | N/A (no relational signal available) | 0.8913 |

## Interpretation

The XGBoost baseline outperforms the GCN on aggregate precision, F1, and ROC-AUC when evaluated on the full test set. Training curves for the GCN showed clear convergence (loss plateaued after learning-rate decay), ruling out undertraining as the cause — the gap instead points to the limited feature richness on entity nodes (Card/Address/Device/EmailDomain currently carry only degree and synthetic-flag features).

The GCN's key strength is its **89.1% recall on synthetically injected fraud rings** — transactions connected only through a shared device, which the XGBoost baseline has no mechanism to detect at all, since it evaluates each transaction independently. This is the core empirical evidence for the project's central claim: relational structure surfaces fraud patterns invisible to row-based models.
