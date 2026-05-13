"""
linear_models.py

Linear Regression, Ridge, Lasso and Support Vector Regression scaffold
for the ML Final Project.

Main design choices:
- Time-based splitting, no random shuffle.
- Median imputation inside scikit-learn pipelines.
- Feature scaling with StandardScaler for Linear/Ridge/Lasso/SVR.
- Optional target scaling for SVR through TransformedTargetRegressor.
- Naive baseline included for comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ---------------------------------------------------------------------
# 1. Default configuration
# ---------------------------------------------------------------------

DEFAULT_TARGET_COL = "Ucome_fob_ARA"

DEFAULT_FEATURE_COLS = [
    "HVO_class_II_fob_ARA",
    "Ucome_fob_ARA",
    "UCO_exw_ARA",
    "UCO_fob_ARA",
    "UCO_cif_ARA",
    "EU_BRENT_CRUDE",
    "LSMGO_Rotterdam",
    "EUR_USD_rate",
    "FED_Funds_Rate",
    "FuelEU_MGO_penalty",
]

DEFAULT_LAGS = [1, 7, 30]
DEFAULT_ROLLING_WINDOWS = [7, 30]


# ---------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------

def load_dataset(
    data_path: str,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Load the processed dataset.

    Parameters
    ----------
    data_path:
        Path to the processed CSV file.
    date_col:
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Loaded and date-sorted dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset path does not exist.
    ValueError
        If the date column is missing.
    """

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}. "
            "Please check that the processed CSV exists before running the model."
        )

    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(
            f"Expected date column '{date_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    return df


# ---------------------------------------------------------------------
# 3. Preprocessing and feature engineering
# ---------------------------------------------------------------------

def prepare_model_dataset(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    feature_cols: Optional[List[str]] = None,
    date_col: str = "Date",
    start_date: str = "2023-01-01",
    horizon: int = 1,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    target_mode: str = "level",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare a supervised learning dataset for price prediction.

    Target modes
    ------------
    level:
        Predict future price level.
        Example: target = HVO[t + horizon]

    change:
        Predict future price change.
        Example: target = HVO[t + horizon] - HVO[t]

    return:
        Predict future percentage return.
        Example: target = (HVO[t + horizon] - HVO[t]) / HVO[t]

    Parameters
    ----------
    df:
        Raw or processed merged dataset.
    target_col:
        Column to predict.
    feature_cols:
        List of explanatory variables.
    date_col:
        Date column.
    start_date:
        Start date used for filtering.
    horizon:
        Forecast horizon. Example: 1 = next day, 7 = seven days ahead.
    lags:
        Lagged feature windows.
    rolling_windows:
        Rolling mean/std windows.
    target_mode:
        "level", "change", or "return".

    Returns
    -------
    X, y
        Feature matrix and target vector.
    """

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    if target_mode not in ["level", "change", "return"]:
        raise ValueError("target_mode must be one of: 'level', 'change', 'return'.")

    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    if lags is None:
        lags = DEFAULT_LAGS

    if rolling_windows is None:
        rolling_windows = DEFAULT_ROLLING_WINDOWS

    df = df.copy()

    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' not found.")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only available feature columns.
    feature_cols = [col for col in feature_cols if col in df.columns]

    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found in the dataset.")

    # Ensure current target price is available as a feature.
    # This is not leakage because the target is shifted into the future.
    if target_col not in feature_cols:
        feature_cols = [target_col] + feature_cols

    # Sort, filter, and index by date.
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[df[date_col] >= start_date].copy()
    df = df.set_index(date_col)

    if df.empty:
        raise ValueError(
            f"No observations remain after filtering from start_date={start_date}."
        )

    # Forward-fill time-ordered variables using only past available values.
    # This is mainly useful for low-frequency variables such as interest rates.
    # Remaining missing values are handled inside the model pipelines.
    df = df.ffill()

    model_df = df[feature_cols].copy()

    # Create target.
    if target_mode == "level":
        model_df["target"] = model_df[target_col].shift(-horizon)

    elif target_mode == "change":
        model_df["target"] = model_df[target_col].shift(-horizon) - model_df[target_col]

    elif target_mode == "return":
        model_df["target"] = (
            model_df[target_col].shift(-horizon) - model_df[target_col]
        ) / model_df[target_col]

    # Lagged features.
    for col in feature_cols:
        for lag in lags:
            model_df[f"{col}_lag_{lag}"] = model_df[col].shift(lag)

    # Rolling features.
    for col in feature_cols:
        for window in rolling_windows:
            model_df[f"{col}_rolling_mean_{window}"] = (
                model_df[col].rolling(window).mean()
            )
            model_df[f"{col}_rolling_std_{window}"] = (
                model_df[col].rolling(window).std()
            )

    # Spread features.
    if (
        "HVO_class_II_fob_ARA" in model_df.columns
        and "LSMGO_Rotterdam" in model_df.columns
    ):
        model_df["HVO_minus_LSMGO"] = (
            model_df["HVO_class_II_fob_ARA"] - model_df["LSMGO_Rotterdam"]
        )

    if "Ucome_fob_ARA" in model_df.columns and "UCO_fob_ARA" in model_df.columns:
        model_df["UCOME_minus_UCO"] = (
            model_df["Ucome_fob_ARA"] - model_df["UCO_fob_ARA"]
        )

    # Drop rows where the target is missing.
    # Remaining feature NaNs are handled inside model pipelines.
    model_df = model_df.dropna(subset=["target"])

    X = model_df.drop(columns=["target"])
    y = model_df["target"]

    if len(X) == 0:
        raise ValueError(
            "No rows available after target creation. "
            "Try a shorter horizon or check missing values in the target column."
        )

    print(f"Prepared model dataset: X={X.shape}, y={y.shape}")
    print(f"Target mode: {target_mode}, horizon: {horizon}")
    print(f"Feature count: {X.shape[1]}")

    return X, y


# ---------------------------------------------------------------------
# 4. Time-based splitting
# ---------------------------------------------------------------------

def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Chronological train / validation / test split.

    Time-series data should not be randomly shuffled.
    """

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1.")

    n = len(X)

    if n < 10:
        raise ValueError("Not enough rows to create train/validation/test splits.")

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    split_data = {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
    }

    print("Train:", split_data["X_train"].shape)
    print("Validation:", split_data["X_val"].shape)
    print("Test:", split_data["X_test"].shape)

    print(
        "Train dates:",
        split_data["X_train"].index.min(),
        "to",
        split_data["X_train"].index.max(),
    )
    print(
        "Validation dates:",
        split_data["X_val"].index.min(),
        "to",
        split_data["X_val"].index.max(),
    )
    print(
        "Test dates:",
        split_data["X_test"].index.min(),
        "to",
        split_data["X_test"].index.max(),
    )

    return split_data


# ---------------------------------------------------------------------
# 5. Model builders
# ---------------------------------------------------------------------

def build_linear_regression() -> Pipeline:
    """
    Plain Linear Regression with imputation and feature scaling.

    Scaling is included for consistency with Ridge, Lasso and SVR,
    and to make coefficients more comparable.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def build_ridge(alpha: float = 1.0) -> Pipeline:
    """
    Ridge Regression.

    Ridge is useful when the engineered dataset contains many correlated features.
    Feature scaling is important because the L2 penalty is scale-sensitive.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def build_lasso(alpha: float = 0.01, max_iter: int = 10_000) -> Pipeline:
    """
    Lasso Regression.

    Lasso can shrink some coefficients to zero and may be useful for feature selection.
    Feature scaling is important because the L1 penalty is scale-sensitive.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=max_iter)),
    ])


def build_svr(
    kernel: str = "rbf",
    C: float = 10.0,
    epsilon: float = 0.1,
    gamma: str = "scale",
    scale_y: bool = True,
):
    """
    Support Vector Regression.

    SVR is sensitive to feature scale, so X is standardized.
    The target y can optionally be standardized using TransformedTargetRegressor.

    scale_y=True is useful because HVO prices can be much larger than variables
    such as Brent, EUR/USD or interest rates.
    """

    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)),
    ])

    if scale_y:
        return TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=StandardScaler(),
        )

    return base_pipeline


# ---------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------

def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    current_price=None,
    target_mode: str = "level",
) -> Dict[str, Union[float, str]]:
    """
    Evaluate regression predictions.

    Directional accuracy is computed differently depending on target mode:

    - level:
        compares sign(y_future - current_price) with sign(y_pred - current_price)

    - change / return:
        compares sign(y_true) with sign(y_pred)
    """

    if target_mode not in ["level", "change", "return"]:
        raise ValueError("target_mode must be one of: 'level', 'change', 'return'.")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    if target_mode == "level" and current_price is not None:
        current_price = np.asarray(current_price).flatten()
        true_direction = np.sign(y_true - current_price)
        pred_direction = np.sign(y_pred - current_price)
        directional_accuracy = float(np.mean(true_direction == pred_direction))

    elif target_mode in ["change", "return"]:
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        directional_accuracy = float(np.mean(true_direction == pred_direction))

    else:
        directional_accuracy = np.nan

    return {
        "model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy,
    }


def evaluate_on_split(
    model,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    target_col: Optional[str] = None,
    target_mode: str = "level",
) -> Dict[str, Union[float, str]]:
    """
    Predict and evaluate a model on validation or test split.
    """

    pred = model.predict(X)

    current_price = None
    if target_mode == "level" and target_col is not None and target_col in X.columns:
        current_price = X[target_col].values

    return evaluate_predictions(
        model_name=model_name,
        y_true=y.values,
        y_pred=pred,
        current_price=current_price,
        target_mode=target_mode,
    )


# ---------------------------------------------------------------------
# 7. Main experiment runner
# ---------------------------------------------------------------------

def run_linear_models_experiment(
    data_path: str,
    target_col: str = DEFAULT_TARGET_COL,
    feature_cols: Optional[List[str]] = None,
    horizon: int = 1,
    target_mode: str = "level",
    start_date: str = "2023-01-01",
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    End-to-end experiment for Linear Regression, Ridge, Lasso and SVR.

    Requires a real dataset path. No mock dataset is created.
    """

    df = load_dataset(data_path=data_path)

    X, y = prepare_model_dataset(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        horizon=horizon,
        target_mode=target_mode,
        start_date=start_date,
    )

    split = time_based_split(
        X=X,
        y=y,
        train_size=train_size,
        val_size=val_size,
    )

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    models = {
        "Linear Regression": build_linear_regression(),
        "Ridge": build_ridge(alpha=1.0),
        "Lasso": build_lasso(alpha=0.01),
        "SVR RBF": build_svr(kernel="rbf", C=10.0, epsilon=0.1, scale_y=True),
    }

    validation_results = []
    test_results = []
    fitted_models = {}

    # Naive baseline.
    if target_mode == "level":
        val_baseline = X_val[target_col].values
        test_baseline = X_test[target_col].values
        val_current_price = X_val[target_col].values
        test_current_price = X_test[target_col].values

    elif target_mode in ["change", "return"]:
        # For change/return targets, naive baseline predicts no movement.
        val_baseline = np.zeros(len(y_val))
        test_baseline = np.zeros(len(y_test))
        val_current_price = None
        test_current_price = None

    validation_results.append(
        evaluate_predictions(
            model_name="Naive Baseline",
            y_true=y_val.values,
            y_pred=val_baseline,
            current_price=val_current_price,
            target_mode=target_mode,
        )
    )

    test_results.append(
        evaluate_predictions(
            model_name="Naive Baseline",
            y_true=y_test.values,
            y_pred=test_baseline,
            current_price=test_current_price,
            target_mode=target_mode,
        )
    )

    # Train and evaluate models.
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        validation_results.append(
            evaluate_on_split(
                model=model,
                model_name=name,
                X=X_val,
                y=y_val,
                target_col=target_col,
                target_mode=target_mode,
            )
        )

        test_results.append(
            evaluate_on_split(
                model=model,
                model_name=name,
                X=X_test,
                y=y_test,
                target_col=target_col,
                target_mode=target_mode,
            )
        )

    validation_df = pd.DataFrame(validation_results).assign(split="validation")
    test_df = pd.DataFrame(test_results).assign(split="test")

    results_df = pd.concat([validation_df, test_df], ignore_index=True)

    artifacts = {
        "raw_df": df,
        "X": X,
        "y": y,
        "split": split,
        "models": fitted_models,
        "config": {
            "data_path": data_path,
            "target_col": target_col,
            "horizon": horizon,
            "target_mode": target_mode,
            "start_date": start_date,
            "train_size": train_size,
            "val_size": val_size,
        },
    }

    return results_df, artifacts