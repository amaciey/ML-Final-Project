# Import models

# Load raw data from sources (./data/raw/)

# Handle missing values and validate data quality

# Calculate or integrate technical indicators
# (Reference technical_indicators.py output)

# Normalize features using StandardScaler

# Apply feature selection (Random Forest ranking)

# Create windowed tf.data.Dataset with different lookahead values
# Lookahead options: 1 day, 10 days, 20 days
# Handle temporal integrity (no data leakage between train/val/test)

# Split windowed data into train/validation/test sets
# Maintain chronological order (time series best practice)
# Implement walk-forward cross-validation

# Train LSTM model

# Train GRU model

# Train Hybrid LSTM-GRU model

# Evaluate all models on test set and collect metrics

# Compare performance across LSTM, GRU, and Hybrid

# Visualize results and create comparison plots

# Generate performance summary (metrics table, rankings)

# Save trained models for reproducibility and inference