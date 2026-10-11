# Clean MLP intrusion detector (CICIDS2017)

Baseline (un-poisoned) intrusion detector trained on the preprocessed CICIDS2017 splits.

- **Run:** `python -m src.models.train_clean --config configs/model.yaml`
- **Measured:** clean test-set performance (accuracy, per-class precision/recall/F1, ROC-AUC, attack PR-AUC, confusion matrix).

## Architecture
- Input features: **69**
- Hidden layers: **[128, 64, 32]** (ReLU, dropout=0.3, batchnorm=True)
- Output: **2-class softmax** (logits)
- Trainable parameters: **19,810**

## Training
- Optimizer: Adam (lr=0.001, weight_decay=0.0)
- Batch size: 4096, max epochs: 30, patience: 5
- Class weights: True (inverse frequency, from train split)
- Early-stopping monitor: `val_pr_auc` (best at epoch 13, score=0.9914)
- Seed: 42

## Test metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9691 |
| **Attack recall** | **0.9890** |
| **Attack F1** | **0.9152** |
| **PR-AUC (attack)** | **0.9914** |
| ROC-AUC | 0.9980 |
| Macro F1 | 0.9482 |
| Macro precision | 0.9247 |
| Macro recall | 0.9770 |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| benign | 0.9977 | 0.9650 | 0.9811 | 314,259 |
| attack | 0.8517 | 0.9890 | 0.9152 | 63,861 |

### Confusion matrix (test)

Rows = true, columns = predicted (`[benign, attack]`).

```
[[303266  10993]
 [   704  63157]]
```

## Artifacts
- `metrics.json`, `history.json`
- `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
- Checkpoint: `results/models/clean_mlp.pt` (outside git)

*Attack recall, attack F1, and PR-AUC are emphasized because the dataset is imbalanced (~16.9% attack), so accuracy alone is not informative.*