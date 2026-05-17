# Import other libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from skopt import forest_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Import modules from this project
from src.feature_engine import technical_indicator_creation
from src.models.neural_models import *
from src.feature_select import feature_selection

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



# Split windowed data into train/validation/test sets
# Maintain chronological order (time series best practice)
# Handle temporal integrity (no data leakage between train/val/test)
def train_test_split(dataset):
    """
    Split scaled data into train (80%), and test (20%)
    while maintaining chronological order and ensuring no data leakage.
    
    By splitting before windowing and maintaining strict sequential order,
    we prevent information from future periods (val/test) from influencing training.
    window_size and lookahead_value define the temporal constraints of each sample.
    This assumes that windows were created sequentially with targets following after the end of the window.
    
    Args:
        dataset: DataFrame output from data_scaler function
    
    Returns:
        tuple: (train_data, test_data) - two tf.data.Dataset objects
    """
    
    # Count total samples in the windowed dataset
    total_samples = len(dataset)
    
    # Calculate split indices maintaining 80/20 ratio and buffer to avoid data leakage
    train_size = int(0.8 * total_samples)
    
    # Training data: first 80% of windows (exclusive of buffer windows defined above)
    train_data = dataset.iloc[:train_size,:].reset_index(drop=True)
    
    # Skip training data and take only test (20%)
    test_data = dataset.iloc[train_size:,:].reset_index(drop=True)

    return train_data, test_data

# Create windowed tf.data.Dataset with different lookahead values
# Lookahead options: 1 day, 10 days, 20 days
def window_creator(dataset, window_size, lookahead_value, batch_size):
    """
    Create windowed dataset with properly scaled targets.
    Targets (Ucome_fob_ARA) are already scaled in the dataset.
    Extract targets at the appropriate lookahead index.
    """
    # Extract scaled target column at lookahead positions
    target_column = dataset.loc[(window_size+lookahead_value-1):, 'Ucome_fob_ARA'].values
    
    windowed_data = tf.keras.utils.timeseries_dataset_from_array(
        dataset.to_numpy(),
        targets=target_column,
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    
    return windowed_data

# Implement walk-forward cross-validation

# Implement hyperparameter tuning
def optimize_hyperparameters(train_df, model_type, window_size, lookahead_value, batch_size, n_features):
    print("\n--- Starting Hyperparameter Optimization ---")
    print(f"Model Type: {model_type}")
    # 1. Define the Search Space
    base_space = [
        Real(0.1, 0.5, name='dropout'),
        Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate')
    ]

    if model_type == 'lstm':
        space = [Integer(20, 100, name='lstm_units')]+base_space
    elif model_type == 'gru':
        space = [Integer(20, 100, name='gru_units')]+base_space
    elif model_type == 'hybrid':
        space = [Integer(20, 100, name='gru_units'), Integer(20, 100, name='lstm_units')]+base_space
    else:
        raise ValueError("Invalid model_type. Choose from 'lstm', 'gru', or 'hybrid'.")

    
    # 2. Define the Objective Function
    @use_named_args(space)
    def objective(gru_units = None, lstm_units = None, dropout = None, learning_rate = None):
        # Walk-Forward Cross Validation
        tscv = TimeSeriesSplit(n_splits=3)
        fold_val_losses = []
        
        # Split the base DataFrame chronologically
        for train_idx, val_idx in tscv.split(train_df):
            
            # Extract train/val splits
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Convert DataFrames to Windowed tf.data.Datasets
            fold_train_data = window_creator(fold_train_df, window_size, lookahead_value, batch_size)
            fold_val_data = window_creator(fold_val_df, window_size, lookahead_value, batch_size)
            
            # Skip if the fold is too small to create windows
            if len(fold_train_data) == 0 or len(fold_val_data) == 0:
                continue

            # Instantiate Model with current hyperparameters
            if model_type == 'lstm':
                model = LSTMModel(
                    lstm_units=int(lstm_units),
                    dropout=float(dropout),
                    learning_rate=float(learning_rate),
                    window_size=window_size,
                    n_features=n_features
                )
            elif model_type == 'gru':
                model = GRUModel(
                    gru_units=int(gru_units),
                    dropout=float(dropout),
                    learning_rate=float(learning_rate),
                    window_size=window_size,
                    n_features=n_features
                )
            elif model_type == 'hybrid':
                model = HybridGRU_LSTM(
                    gru_units=int(gru_units),
                    lstm_units=int(lstm_units),
                    dropout=float(dropout),
                    learning_rate=float(learning_rate),
                    window_size=window_size,
                    n_features=n_features
                )
            
            # Early stopping to prevent wasting time on bad configs
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
            
            # Train model on this fold
            history = model.train(
                fold_train_data, 
                val_data=fold_val_data, 
                epochs=15, 
                callbacks=[early_stop]
            )
            
            # Get the best validation loss for this fold
            best_val_loss = min(history.history['val_loss'])
            fold_val_losses.append(best_val_loss)
            
        # The objective is the average validation loss across all time-folds
        mean_val_loss = np.mean(fold_val_losses)
        print(f"Tested: GRU={gru_units}, LSTM={lstm_units}, Drop={dropout:.2f}, LR={learning_rate:.4f}, Window={window_size} --> Mean Val Loss: {mean_val_loss:.6f}")
        
        return mean_val_loss

    # 3. Run the Tree-Based Optimization
    res = forest_minimize(
        func=objective,
        dimensions=space,
        base_estimator="RF",  # Uses a standard Random Forest Regressor
        n_calls=25,
        n_initial_points=5,
        random_state=42
    )
    
    # 4. Extract and return the best parameters
    best_params = {dim.name: res.x[i] for i, dim in enumerate(space)}

    best_params['window_size'] = window_size
    
    print("\nOptimal Hyperparameters Found:")
    print(best_params)
    return best_params
# Train LSTM model

# Train GRU model

# Train Hybrid LSTM-GRU model

# Evaluate all models on test set and collect metrics

# Compare performance across LSTM, GRU, and Hybrid

# Visualize results and create comparison plots

# Generate performance summary (metrics table, rankings)

# Save trained models for reproducibility and inference


if __name__ == "__main__":
    """
    Main orchestration: Load data, create windows, split train/test,
    instantiate model, train, and evaluate.
    """
    
    # Step 1: Load and scale data
    print("Loading and scaling data...")
    scaled_data = data_scaler('MinMax')
    
    # Check target column stats
    target_col = scaled_data['Ucome_fob_ARA']
    print(f"\nTarget column (Ucome_fob_ARA) statistics:")
    print(f"  Min: {target_col.min():.6f}, Max: {target_col.max():.6f}")
    print(f"  Mean: {target_col.mean():.6f}, Std: {target_col.std():.6f}")
    
    # Step 2: Select Features
    print("Selecting Features using Random Forest...")
    scaled_data, selected_cols = select_features_rf(scaled_data, top_n=40)
    n_features = scaled_data.shape[1]  # Number of features (price + indicators)

    # Step 3: Split into train/test maintaining chronological order
    print("Splitting into train/test sets...")
    train_data, test_data = train_test_split(scaled_data)

    print("Training Data: " + str(train_data))
    print("Testing Data: " + str(test_data))
    
    # Step 4: Create windowed dataset (targets automatically extracted)
    print("\nCreating windowed dataset...")
    window_size = 30
    lookahead_value = 10
    batch_size = 32

    best_params = optimize_hyperparameters(
        train_df=train_data, 
        lookahead_value=lookahead_value, 
        batch_size=batch_size, 
        n_features=n_features
    )
    
    windowed_train = window_creator(train_data, best_params['window_size'], lookahead_value, batch_size)

    windowed_test = window_creator(test_data, best_params['window_size'], lookahead_value, batch_size)
    
    # Step 4: Instantiate the model
    print("Building HybridGRU_LSTM model...")
    model = HybridGRU_LSTM(
    gru_units=int(best_params['gru_units']),
    lstm_units=int(best_params['lstm_units']),
    dropout=float(best_params['dropout']),
    learning_rate=float(best_params['learning_rate']),
    window_size=best_params['window_size'],
    n_features=n_features
    )
    
    # Step 5: Train the model
    print("Training model on training set...")
    history = model.train(windowed_train)  # train_data contains (X, y) pairs internally
    print("Training complete!\n")
    
    # Step 6: Evaluate on test set
    print("Evaluating model on test set...")
    test_metrics = model.evaluate(windowed_test)  # test_data contains (X, y) pairs internally
    print(f"Test Loss (MSE): {test_metrics[0]:.6f}")
    print(f"Test MAE: {test_metrics[1]:.6f}\n")
    
    # Step 7: Generate predictions on test set
    print("Generating predictions on test set...")
    predictions = model.predict(windowed_test)
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 10 predictions:\n{predictions[:10].flatten()}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print("\nModel training and evaluation complete!")