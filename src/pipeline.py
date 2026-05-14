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
def window_creator(dataset, window_size, lookahead_value, batch_size):
    windowed_data = tf.keras.utils.timeseries_dataset_from_array(
        dataset.to_numpy(),
        targets = dataset.loc[(window_size+lookahead_value-1):, 'Ucome_fob_ARA'],
        sequence_length = window_size,
        batch_size = batch_size,
        shuffle=False,
        seed=42
        )
    
    return windowed_data

test_windowed_data = window_creator(data_scaler('MinMax'), window_size=30, lookahead_value=10, batch_size=2)

# Split windowed data into train/validation/test sets
# Maintain chronological order (time series best practice)
# Handle temporal integrity (no data leakage between train/val/test)
def train_test_split(windowed_data, window_size, lookahead_value):
    """
    Split windowed time series data into train (80%), and test (20%)
    while maintaining chronological order and ensuring no data leakage.
    
    By splitting after windowing and maintaining strict sequential order,
    we prevent information from future periods (val/test) from influencing training.
    window_size and lookahead_value define the temporal constraints of each sample.
    This assumes that windows were created sequentially with targets following after the end of the window.
    
    Args:
        windowed_data: tf.data.Dataset output from window_creator function
        window_size: Number of historical timesteps in each window (e.g., 30 days)
        lookahead_value: Number of steps ahead for prediction target (e.g., 10 days)
    
    Returns:
        tuple: (train_data, test_data) - two tf.data.Dataset objects
    """
    
    # Count total samples in the windowed dataset
    total_samples = sum(1 for _ in windowed_data)
    
    # Calculate split indices maintaining 80/20 ratio and buffer to avoid data leakage
    train_size = int(0.8 * total_samples) - (lookahead_value + window_size)
    
    # Training data: first 80% of windows (exclusive of buffer windows defined above)
    train_data = windowed_data.take(train_size)
    
    # Skip training data and take only test (20%)
    test_data = windowed_data.skip(train_size)
    
    return train_data, test_data


# Implement walk-forward cross-validation

# Train LSTM model

# Train GRU model

# Train Hybrid LSTM-GRU model

# Evaluate all models on test set and collect metrics

# Compare performance across LSTM, GRU, and Hybrid

# Visualize results and create comparison plots

# Generate performance summary (metrics table, rankings)

# Save trained models for reproducibility and inference