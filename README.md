## System Architecture

```
ML-Final-Project/
├── data/
│   ├── raw/              # Original datasets (final_dataset.csv, etc.)
│   └── processed/        # Data after feature engineering
├── src/
│   ├── data_loader.py    # Load and validate raw data
│   ├── feature_engine.py # Technical indicators calculation
│   ├── feature_select.py # Random Forest feature selection
│   ├── models/           # Model implementations
│   │   ├── __init__.py
│   │   ├── linear_models.py     # Linear Regression, SVM
│   │   ├── neural_models.py     # LSTM
│   │   └── evaluation.py        # Metrics, cross-validation
│   └── pipeline.py       # Orchestrate full workflow
├── notebooks/
│   ├── El_parsero.ipynb         # Data exploration
├── config.py             # Hyperparameters, paths, constants
├── model_evaluation.ipynb              # Entry point
└── requirements.txt
```

**Module Breakdown:**
- **data_loader.py** — Read CSV, validate schema, handle missing values
- **feature_engine.py** — Calculate 22 technical indicators (stockstats-based)
- **feature_select.py** — Random Forest feature ranking & selection
- **linear_models.py** — Linear Regression and SVM model implementations
- **neural_models.py** — LSTM model implementation
- **evaluation.py** — Shared metrics, cross-validation, performance comparison
- **pipeline.py** — Orchestrate workflow: load → engineer → select → train → evaluate
- **model_evaluation.ipynb** — Single entry point that calls pipeline and creates visualizations comparing model performance
- **config.py** — Centralized hyperparameters, file paths, constants

## Data Collection

## Technical Indicators
Created using the Stockstats Python library

We cannot re-create all 42 indicators in the Hybrid-LSTM report because we don't have Volume, Open, High, and Low data. The following are a subset of 22 indicators that we are capable of re-creating using our data.

### Trend-Following
Aim to identify the direction of price trends
- SMA: Smooths data based on mean of past prices
- EMA: Weights recent prices higher in an SMA
- LRMA: Uses linear regression to fit a trend line 
- KAMA: Adjusts smoothing based on volatility
- CTI: Asseses strength and consistency of trends using correlations

### Momentum
Aim to measure the speed and rate of price movements
- RSI: Compares magnitudes of recent gains and losses
- MACD: Plots relationship between two EMAs
- TRIX: Percentage change of a triple smoothed EMA
- PPO: Percentage difference of two EMAs as a percentage of the longer of the two
- Stochastic RSI: Applies a stochastic oscillator formula to RSI values
- RoC: Percentage change over a period
- Coppock Curve: Rated sum of RoC, used to identify long-term opportunities
- KST: Combination of smoothed ROC indicators over multiple different timeframes
- PGO: Percentage difference between two moving averages
- PSL: Ratio of rising periods to total periods based on price direction

### Volatility
Aim to assess the degree of price movements
- MSTD: Dispersion of price data around its moving average
- MVAR: Squared deviation of price from its moving average
- Bollinger Band: Price bands based on standard deviations from moving average
- KER: Quantifies noise by measuring efficiency of price movement over a period

### Other
- DMA: Difference between moving averages of different periods
- MAD: Average absolute difference from the mean to measure volatility
- BIAS: Percentage difference between current closing price and moving average

## Feature Selection with Random Forest

## Data Split

### Sliding 30-day Windows

## Hyperparameter Tuning