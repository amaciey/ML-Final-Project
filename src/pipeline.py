# Import other libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

# Import modules from this project
from technical_indicators import technical_indicator_creation
    # Import models

# Load raw data from sources (./data/raw/)

# Handle missing values and validate data quality

# Calculate or integrate technical indicators
# (Reference technical_indicators.py output)

# Data Scaling and Normalization
def data_scaler(scaler_type):
    """
    Function to scale cleaned data prior to any 
    """

    data = technical_indicator_creation()

    # Normalize features using StandardScaler
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Apply StandardScaler to the dataset
    data_standardized = pd.DataFrame(
        standard_scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

    # Apply MinMaxScaler to the dataset
    data_minmax = pd.DataFrame(
        minmax_scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

    if scaler_type == 'MinMax':
        return data_minmax

    else:
        return data_standardized

# Apply feature selection (Random Forest ranking)

# Create windowed tf.data.Dataset with different lookahead values
# Lookahead options: 1 day, 10 days, 20 days
# Handle temporal integrity (no data leakage between train/val/test)
def window_creator(dataset, window_size, lookahead_value, batch_size):
    windowed_data = tf.keras.utils.timeseries_dataset_from_array(
        dataset.to_numpy(),
        targets = dataset.loc[(window_size+lookahead_value-1):, 'Ucome_fob_ARA'],
        sequence_length = window_size,
        batch_size = batch_size,
        shuffle=True,
        seed=42
        )
    
    return windowed_data

test_windowed_data = window_creator(data_scaler('MinMax'), window_size=30, lookahead_value=10, batch_size=2)

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