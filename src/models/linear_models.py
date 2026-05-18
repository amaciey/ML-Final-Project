"""Linear models experiment utilities for time-series regression."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TARGET_COL = "Ucome_fob_ARA"
DEFAULT_LAGS = [1, 7, 30]
DEFAULT_ROLLING_WINDOWS = [7, 30]


 # Load and validate a time-indexed CSV so downstream steps always start from
 # a consistent, date-sorted DataFrame.
def load_dataset(data_path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load CSV dataset and return date-sorted rows."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}.")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' not found.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    return df


 # Auto-discover numeric candidate predictors when a manual feature list is
 # not provided, excluding target/date to prevent accidental leakage.
def infer_feature_columns(df: pd.DataFrame, target_col: str, date_col: str = "Date") -> List[str]:
    """Infer numeric feature columns excluding target/date."""
    excluded = {target_col, date_col}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


 # Convert raw time-series data into supervised-learning inputs/targets by
 # creating future targets plus lag/rolling features needed by linear models.
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
    """Build supervised X/y with shifted target and simple time features."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if target_mode not in ["level", "change", "return"]:
        raise ValueError("target_mode must be one of: 'level', 'change', 'return'.")

    lags = DEFAULT_LAGS if lags is None else lags
    rolling_windows = DEFAULT_ROLLING_WINDOWS if rolling_windows is None else rolling_windows

    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' not found.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if feature_cols is None:
        feature_cols = infer_feature_columns(df=df, target_col=target_col, date_col=date_col)
    else:
        feature_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("No valid feature columns found in the dataset.")
    if target_col not in feature_cols:
        feature_cols = [target_col] + feature_cols

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[df[date_col] >= start_date].copy()
    df = df.set_index(date_col)
    if df.empty:
        raise ValueError(f"No observations remain after filtering from start_date={start_date}.")

    df = df.ffill()
    model_df = df[feature_cols].copy()

    if target_mode == "level":
        model_df["target"] = model_df[target_col].shift(-horizon)
    elif target_mode == "change":
        model_df["target"] = model_df[target_col].shift(-horizon) - model_df[target_col]
    else:
        model_df["target"] = (model_df[target_col].shift(-horizon) - model_df[target_col]) / model_df[target_col]

    for col in feature_cols:
        for lag in lags:
            model_df[f"{col}_lag_{lag}"] = model_df[col].shift(lag)
    for col in feature_cols:
        for window in rolling_windows:
            model_df[f"{col}_rolling_mean_{window}"] = model_df[col].rolling(window).mean()
            model_df[f"{col}_rolling_std_{window}"] = model_df[col].rolling(window).std()

    if "HVO_class_II_fob_ARA" in model_df.columns and "LSMGO_Rotterdam" in model_df.columns:
        model_df["HVO_minus_LSMGO"] = model_df["HVO_class_II_fob_ARA"] - model_df["LSMGO_Rotterdam"]

    model_df = model_df.dropna(subset=["target"])
    X = model_df.drop(columns=["target"])
    y = model_df["target"]

    if len(X) == 0:
        raise ValueError("No rows available after target creation. Try a shorter horizon.")

    print(f"Prepared model dataset: X={X.shape}, y={y.shape}")
    print(f"Target mode: {target_mode}, horizon: {horizon}")
    print(f"Feature count: {X.shape[1]}")
    return X, y


 # Create chronological train/validation/test partitions for time series,
 # avoiding random shuffle that would break temporal integrity.
def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Chronological train/validation/test split."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1.")
    if len(X) < 10:
        raise ValueError("Not enough rows to create train/validation/test splits.")

    n = len(X)
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
    print("Train dates:", split_data["X_train"].index.min(), "to", split_data["X_train"].index.max())
    print("Validation dates:", split_data["X_val"].index.min(), "to", split_data["X_val"].index.max())
    print("Test dates:", split_data["X_test"].index.min(), "to", split_data["X_test"].index.max())
    return split_data


 # Build a shared preprocessing pipeline so every linear model gets identical
 # imputation/scaling before fitting.
def _base_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


 # Plain OLS baseline for interpretable linear fit and benchmark comparison.
def build_linear_regression() -> Pipeline:
    return _base_pipeline(LinearRegression())


 # L2-regularized linear model to stabilize coefficients under collinearity.
def build_ridge(alpha: float = 1.0) -> Pipeline:
    return _base_pipeline(Ridge(alpha=alpha))


 # L1-regularized linear model to encourage sparse feature usage.
def build_lasso(alpha: float = 0.01, max_iter: int = 10_000) -> Pipeline:
    return _base_pipeline(Lasso(alpha=alpha, max_iter=max_iter))


 # Mixed L1/L2 regularization to balance sparsity and coefficient shrinkage.
def build_elastic_net(alpha: float = 0.01, l1_ratio: float = 0.5, max_iter: int = 10_000) -> Pipeline:
    return _base_pipeline(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter))


 # Compute standard regression metrics and directional accuracy so models can
 # be compared on both error magnitude and sign prediction.
def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    current_price=None,
    target_mode: str = "level",
) -> Dict[str, Union[float, str]]:
    """Compute MAE, MAPE, RMSE, R2, and directional accuracy."""
    if target_mode not in ["level", "change", "return"]:
        raise ValueError("target_mode must be one of: 'level', 'change', 'return'.")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if target_mode == "level" and current_price is not None:
        current_price = np.asarray(current_price).flatten()
        directional_accuracy = float(np.mean(np.sign(y_true - current_price) == np.sign(y_pred - current_price)))
    elif target_mode in ["change", "return"]:
        directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    else:
        directional_accuracy = np.nan

    return {
        "model": model_name,
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy,
    }


 # Run prediction on one split and route outputs through the shared metrics
 # function to keep evaluation logic consistent.
def evaluate_on_split(
    model,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    target_col: Optional[str] = None,
    target_mode: str = "level",
) -> Dict[str, Union[float, str]]:
    """Predict and evaluate a model on one split."""
    pred = model.predict(X)
    current_price = X[target_col].values if (target_mode == "level" and target_col in X.columns) else None
    return evaluate_predictions(model_name=model_name, y_true=y.values, y_pred=pred, current_price=current_price, target_mode=target_mode)


 # Normalize input source (path vs in-memory DataFrame) into one canonical
 # DataFrame format with a Date column for reproducible downstream processing.
def _resolve_input_dataframe(
    data_path: Optional[Union[str, pd.DataFrame]] = None,
    data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Normalize input and ensure a Date column exists."""
    if isinstance(data_path, pd.DataFrame) and data is None:
        data = data_path
        data_path = None

    if data is None and data_path is None:
        raise ValueError("Provide either 'data_path' or 'data'.")
    if data is not None and data_path is not None:
        raise ValueError("Provide only one of 'data_path' or 'data', not both.")

    if data is None:
        return load_dataset(data_path=data_path), data_path

    df = data.copy()
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        else:
            df = df.reset_index(drop=True).copy()
            df["Date"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df, None


 # Build a naive reference forecast (persistence for level, zeros otherwise)
 # to contextualize whether learned models add value.
def _naive_predictions(
    target_mode: str,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    target_col: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build naive baseline predictions for a split."""
    if target_mode == "level":
        baseline = X_split[target_col].values
        current_price = X_split[target_col].values
    else:
        baseline = np.zeros(len(y_split))
        current_price = None
    return baseline, current_price


 # End-to-end orchestration for data prep, splitting, model fitting, baseline
 # comparison, and artifact/result packaging for analysis/reporting.
def run_linear_models_experiment(
    data_path: Optional[Union[str, pd.DataFrame]] = None,
    data: Optional[pd.DataFrame] = None,
    target_col: str = DEFAULT_TARGET_COL,
    feature_cols: Optional[List[str]] = None,
    horizon: int = 1,
    target_mode: str = "level",
    start_date: str = "2023-01-01",
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run end-to-end linear experiment and return results/artifacts."""
    df, resolved_data_path = _resolve_input_dataframe(data_path=data_path, data=data)

    X, y = prepare_model_dataset(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        horizon=horizon,
        target_mode=target_mode,
        start_date=start_date,
    )

    split = time_based_split(X=X, y=y, train_size=train_size, val_size=val_size)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]

    models = {
        "Linear Regression": build_linear_regression(),
        "Ridge": build_ridge(alpha=1.0),
        "Lasso": build_lasso(alpha=0.01),
        "Elastic Net": build_elastic_net(alpha=0.01, l1_ratio=0.5),
    }

    validation_results: List[Dict[str, Union[float, str]]] = []
    test_results: List[Dict[str, Union[float, str]]] = []
    fitted_models: Dict[str, Pipeline] = {}

    val_baseline, val_current = _naive_predictions(target_mode, X_val, y_val, target_col)
    test_baseline, test_current = _naive_predictions(target_mode, X_test, y_test, target_col)

    validation_results.append(
        evaluate_predictions("Naive Baseline", y_val.values, val_baseline, val_current, target_mode)
    )
    test_results.append(
        evaluate_predictions("Naive Baseline", y_test.values, test_baseline, test_current, target_mode)
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        validation_results.append(evaluate_on_split(model, name, X_val, y_val, target_col, target_mode))
        test_results.append(evaluate_on_split(model, name, X_test, y_test, target_col, target_mode))

    results_df = pd.concat([
        pd.DataFrame(validation_results).assign(split="validation"),
        pd.DataFrame(test_results).assign(split="test"),
    ], ignore_index=True)

    artifacts = {
        "raw_df": df,
        "X": X,
        "y": y,
        "split": split,
        "models": fitted_models,
        "config": {
            "data_path": resolved_data_path,
            "target_col": target_col,
            "horizon": horizon,
            "target_mode": target_mode,
            "start_date": start_date,
            "train_size": train_size,
            "val_size": val_size,
        },
    }
    return results_df, artifacts
