# Amazon Review Rating Prediction with DistilBERT

Fine-tuning `distilbert-base-uncased` as a regression head to predict
star ratings from Amazon review text.

## Dataset
[Amazon Reviews (FastText format)](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
— 300K reviews, binary labels (1★ negative / 5★ positive).

## Results
| Metric | Constant Baseline | DistilBERT |
|--------|------------------|------------|
| MAE    | 2.000            | 0.279      |
| RMSE   | 2.000            | 0.818      |
| Macro F1 | —              | 0.95       |

- **86% MAE reduction** and **59% RMSE reduction** over baseline
- **95% macro-F1** on binary sentiment classification
- **93.3%** rounded-star exact-match accuracy on 45K test samples

## Model Architecture

Input tokens (max_len=128)
│
DistilBERT encoder (6 transformer layers, 66M params)
│
[CLS] hidden state (768-dim)
│
Dropout(0.3) → Linear(768→1)
│
Scalar rating ∈ ℝ

## Training Setup
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Scheduler: Linear warm-up + linear decay
- Mixed precision: PyTorch AMP
- Early stopping: patience=3 (stopped at epoch 6)
- Split: 210K train / 45K val / 45K test

## How to Run
This notebook is designed to run on **Kaggle**.
1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the [Amazon Reviews dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
3. Upload `MLProject_Enhanced.ipynb` and run all cells

## Author
Muhammad Musa — [github.com/musarashid49](https://github.com/musarashid49)
Areesha Saqib - [github.com/Areesha-008](http://github.com/Areesha-008)
