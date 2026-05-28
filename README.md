# stock-price-gat-forecasting

> Graph Attention Network (GAT) for stock price forecasting using a multi-stock correlation graph structure, implemented as part of a broader comparative deep learning study.

---

## About

This repository contains my individual contribution to a comparative deep learning study on stock price forecasting. The full study evaluated four architectures — LSTM with Temporal Attention, GRU with Multi-Head Attention, Graph Attention Network (GAT), and a Transformer — for predicting NVIDIA (NVDA) stock prices using historical data from 9 correlated stocks.

This repository focuses specifically on the **GAT implementation**, which was my responsibility in the project. The GAT model treats stocks as nodes in a dynamic graph, where edges represent correlations between stocks, allowing the model to learn relational patterns across the market rather than treating each stock in isolation.

The other models in the comparative study (LSTM, GRU, Transformer) were implemented by teammates and are not included here.

---

## Dataset

- **Target stock:** NVIDIA (NVDA)
- **Related stocks used as graph nodes:** NVDA, AAPL, AMZN, MSFT, GOOGL, META, TSM, AMD, INTC
- **Features per stock:** Open, High, Low, Close, Volume
- **Date range:** January 2018 — December 2023
- **Source:** Yahoo Finance via yfinance

---

## Model Architecture — GAT

The GAT model builds a dynamic graph where each stock is a node and edges are weighted by rolling Pearson correlation between closing prices. The model uses GATv2Conv layers to attend over neighboring stocks and learn which relationships are most predictive of NVDA's next-day closing price.

**Key design choices:**
- Graph constructed using a 30-day rolling correlation window
- Edges included only when correlation exceeds a threshold of 0.3
- GATv2Conv with multi-head attention to capture diverse relational patterns
- Sequence length of 10 trading days as input window
- Hyperparameter optimization using Optuna (350 trials)

---

## Hyperparameter Tuning

Optuna was used to optimize the following:

| Hyperparameter | Search Range |
|---|---|
| Number of GAT layers | 1 — 3 |
| Hidden dimension | 16 — 128 |
| Number of attention heads | 1 — 8 |
| Dropout rate | 0.1 — 0.5 |
| Learning rate | 1e-4 — 1e-2 |
| Sequence length | 5 — 20 |

---

## Results

| Metric | GAT |
|---|---|
| R² | 0.9391 |
| RMSE | $3.645 |
| MAE | $2.831 |
| Directional Accuracy | 93.33% |

The GAT model achieved the best performance among all four architectures in the comparative study, demonstrating that modeling inter-stock relational structure provides meaningful predictive signal beyond individual time series patterns.

---

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, PyTorch Geometric
- **GAT Layer:** GATv2Conv
- **Hyperparameter Tuning:** Optuna
- **Data:** yfinance, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab

---

## Notebooks

| File | Description |
|---|---|
| `gat_stock_forecasting.ipynb` | Full GAT implementation including graph construction, model architecture, Optuna tuning, training, and evaluation with cell outputs |

---

## Context

This GAT implementation was part of a larger comparative study evaluating four deep learning architectures for stock price forecasting. The other models (LSTM with Temporal Attention, GRU with Multi-Head Attention, Transformer) were developed by other team members and are not included in this repository.
